package com.badu.ai.onnx.config;

import com.badu.ai.onnx.engine.ModelArchitecture;
import java.io.IOException;
import java.nio.file.FileVisitOption;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;

/**
 * Utility class for discovering ONNX model files in a directory structure.
 *
 * <p>This class handles the auto-discovery of encoder, decoder, and tokenizer files
 * based on model variant and optional flavour subdirectory.
 *
 * <p><b>Discovery Strategy:</b>
 * <ul>
 *   <li>Encoder/Decoder: Search root directory, then subdirectories (max depth 2)</li>
 *   <li>Tokenizer: Search from encoder directory upwards to root</li>
 *   <li>Flavour: Optional subdirectory filter to disambiguate multiple matches</li>
 * </ul>
 *
 * <p>Thread Safety: This class is stateless and thread-safe.
 *
 * <p>Usage example:
 * <pre>{@code
 * ModelFileDiscovery discovery = new ModelFileDiscovery();
 * ModelFiles files = discovery.discover(
 *     Paths.get("models/flan-t5-small-ONNX"),
 *     ModelVariant.INT8,
 *     null  // no flavour
 * );
 * }</pre>
 */
public class ModelFileDiscovery {

    /**
     * Discovers encoder, decoder, and tokenizer files from model directory.
     *
     * <p>Prefers pre-optimized models in *_optimized directories for 10x faster cold start.
     *
     * @param rootPath root model directory
     * @param variant model variant (FULL, FP16, Q4, INT8)
     * @param flavour optional subdirectory filter (null for auto-detect)
     * @return ModelFiles with discovered paths
     * @throws IllegalStateException if any required file is not found or multiple candidates found
     */
    public ModelFiles discover(Path rootPath, ModelVariant variant, String flavour) {
        // First, detect if this is a decoder-only model
        if (isDecoderOnlyModel(rootPath)) {
            return discoverDecoderOnlyModel(rootPath, variant, flavour);
        }

        // Otherwise, treat as encoder-decoder model
        // Discover encoder (prefer optimized version in _optimized directories)
        String encoderFileName = getEncoderFileName(variant);
        Path encoderFile = findModelFileWithOptimizedDirFallback(rootPath, encoderFileName, flavour);
        if (encoderFile == null) {
            throw new IllegalStateException(
                "Encoder model file not found: " + encoderFileName + " in " + rootPath +
                (flavour != null ? " (flavour: " + flavour + ")" : "") +
                ". Use ModelConfig.encoderPath() to specify custom location.");
        }

        // Discover decoder(s) - prefer dual-session for best KV-cache performance
        Path decoderFile = null;
        Path decoderWithPastFile = null;
        String attemptedFileName = null;

        // Strategy 1: Try dual-session (BEST - no If overhead, optimal KV-cache)
        // Look for both decoder_model.onnx and decoder_with_past_model.onnx (prefer optimized)
        String withPastDecoderFileName = getWithPastDecoderFileName(variant);
        String nonMergedDecoderFileName = getNonMergedDecoderFileName(variant);

        Path withPastPath = findModelFileWithOptimizedDirFallback(rootPath, withPastDecoderFileName, flavour);
        Path nonMergedPath = findModelFileWithOptimizedDirFallback(rootPath, nonMergedDecoderFileName, flavour);

        if (withPastPath != null && nonMergedPath != null) {
            // Dual-session mode: use decoder_model.onnx for first step, decoder_with_past_model.onnx for rest
            decoderFile = nonMergedPath;
            decoderWithPastFile = withPastPath;
            attemptedFileName = withPastDecoderFileName + " + " + nonMergedDecoderFileName;
        }
        // Strategy 2: Try with_past only (GOOD - KV-cache, but requires empty cache tensors on first step)
        else if (withPastPath != null) {
            decoderFile = withPastPath;
            decoderWithPastFile = null;  // Single-session mode with KV-cache
            attemptedFileName = withPastDecoderFileName;
        }
        // Strategy 3: Try merged decoder (OK - KV-cache support, but has If node overhead)
        else {
            String mergedDecoderFileName = getMergedDecoderFileName(variant);
            Path mergedPath = findModelFileWithOptimizedDirFallback(rootPath, mergedDecoderFileName, flavour);

            if (mergedPath != null) {
                decoderFile = mergedPath;
                decoderWithPastFile = null;  // Single-session mode
                attemptedFileName = mergedDecoderFileName;
            }
            // Strategy 4: Try non-merged decoder (FALLBACK - no KV-cache)
            else if (nonMergedPath != null) {
                decoderFile = nonMergedPath;
                decoderWithPastFile = null;  // Single-session mode without cache
                attemptedFileName = nonMergedDecoderFileName;
            }
        }

        if (decoderFile == null) {
            throw new IllegalStateException(
                "Decoder model file not found: " + attemptedFileName + " in " + rootPath +
                (flavour != null ? " (flavour: " + flavour + ")" : "") +
                ". Use ModelConfig.decoderPath() to specify custom location.");
        }

        // Discover tokenizer
        Path encoderDir = encoderFile.getParent();
        String tokenizerPathStr = findTokenizerPath(encoderDir, rootPath);
        if (tokenizerPathStr == null) {
            throw new IllegalStateException(
                "Tokenizer file (tokenizer.json) not found in " + rootPath +
                " or subdirectories. Use ModelConfig.tokenizerPath() to specify custom location.");
        }

        return new ModelFiles(encoderFile, decoderFile, decoderWithPastFile, Paths.get(tokenizerPathStr));
    }

    /**
     * Finds a model file with preference for optimized directories.
     *
     * <p>Search strategy:
     * <ol>
     *   <li>First check *_optimized directories for the file</li>
     *   <li>Then fall back to non-optimized directories</li>
     * </ol>
     *
     * @param rootPath root model directory
     * @param fileName model file name to find
     * @param flavour optional subdirectory filter
     * @return path to model file (optimized if available), or null if not found
     */
    private Path findModelFileWithOptimizedDirFallback(Path rootPath, String fileName, String flavour) {
        // First, try to find in _optimized directories
        Path optimizedPath = findModelFileInOptimizedDirs(rootPath, fileName, flavour);
        if (optimizedPath != null) {
            return optimizedPath;
        }

        // Fall back to regular search
        return findModelFile(rootPath, fileName, flavour);
    }

    /**
     * Finds a model file specifically in *_optimized directories.
     *
     * @param rootPath root model directory
     * @param fileName model file name to find
     * @param flavour optional subdirectory filter
     * @return path to model file in optimized directory, or null if not found
     */
    private Path findModelFileInOptimizedDirs(Path rootPath, String fileName, String flavour) {
        try {
            List<Path> candidates = new ArrayList<>();

            // Common optimized directory names to check
            String[] optimizedDirPatterns = {
                "onnx_optimized",
                "optimized",
                "fp16_optimized",
                "int8_optimized",
                "q4_optimized"
            };

            // Check root-level optimized directories
            for (String dirPattern : optimizedDirPatterns) {
                Path optimizedDir = rootPath.resolve(dirPattern);
                if (Files.isDirectory(optimizedDir)) {
                    Path filePath = optimizedDir.resolve(fileName);
                    if (Files.exists(filePath)) {
                        // If flavour specified and directory matches, prioritize it
                        if (flavour == null || dirPattern.contains(flavour)) {
                            candidates.add(filePath);
                        }
                    }
                }
            }

            // Also check subdirectories with _optimized suffix
            if (Files.isDirectory(rootPath)) {
                try (var stream = Files.walk(rootPath, 2, FileVisitOption.FOLLOW_LINKS)) {
                    stream.filter(Files::isDirectory)
                        .filter(p -> p.getFileName().toString().endsWith("_optimized"))
                        .forEach(dir -> {
                            Path filePath = dir.resolve(fileName);
                            if (Files.exists(filePath)) {
                                if (flavour == null || dir.toString().contains(flavour)) {
                                    candidates.add(filePath);
                                }
                            }
                        });
                }
            }

            if (candidates.isEmpty()) {
                return null;
            }

            // Return the first match (prioritizing root-level optimized dirs)
            return candidates.get(0);

        } catch (IOException e) {
            // If error, just return null to fall back to regular search
            return null;
        }
    }

    /**
     * Finds a model file (encoder or decoder) by searching from root directory.
     *
     * <p>Search strategy:
     * <ol>
     *   <li>First check root directory directly</li>
     *   <li>Then search subdirectories (max depth 2)</li>
     *   <li>If flavour specified, only check that subdirectory</li>
     *   <li>If multiple candidates found without flavour, throw exception</li>
     * </ol>
     *
     * @param rootPath root model directory
     * @param fileName model file name to find
     * @param flavour optional subdirectory filter
     * @return path to model file, or null if not found
     * @throws IllegalStateException if multiple candidates found without flavour
     */
    private Path findModelFile(Path rootPath, String fileName, String flavour) {
        try {
            List<Path> candidates = new ArrayList<>();

            // Check root directory first
            Path directFile = rootPath.resolve(fileName);
            if (Files.exists(directFile)) {
                candidates.add(directFile);
            }

            // Search subdirectories (max depth 2)
            if (Files.isDirectory(rootPath)) {
                try (var stream = Files.walk(rootPath, 2, FileVisitOption.FOLLOW_LINKS)) {
                    stream.filter(Files::isRegularFile)
                        .filter(p -> p.getFileName().toString().equals(fileName))
                        .filter(p -> !p.equals(directFile)) // Don't duplicate root match
                        .filter(p -> flavour == null || p.getParent().getFileName().toString().equals(flavour))
                        .forEach(candidates::add);
                }
            }

            if (candidates.isEmpty()) {
                return null;
            }

            if (candidates.size() == 1) {
                return candidates.get(0);
            }

            // Multiple candidates found
            throw new IllegalStateException(
                "Multiple " + fileName + " files found in " + rootPath + ". " +
                "Please specify 'flavour' parameter to disambiguate. Found: " +
                candidates.stream().map(p -> p.getParent().getFileName().toString()).distinct().toList());

        } catch (IOException e) {
            throw new IllegalStateException("Error searching for " + fileName + " in " + rootPath, e);
        }
    }

    /**
     * Finds tokenizer.json by searching from encoder directory up to root.
     *
     * <p>Search order:
     * <ol>
     *   <li>In the encoder directory itself</li>
     *   <li>In parent directories up to rootPath</li>
     *   <li>In rootPath itself</li>
     * </ol>
     *
     * @param startDir directory to start search from (encoder location)
     * @param rootPath root model directory (don't search above this)
     * @return absolute path to tokenizer.json, or null if not found
     */
    private String findTokenizerPath(Path startDir, Path rootPath) {
        Path currentDir = startDir.toAbsolutePath().normalize();
        Path rootNormalized = rootPath.toAbsolutePath().normalize();

        while (currentDir != null) {
            Path tokenizerFile = currentDir.resolve("tokenizer.json");
            if (Files.exists(tokenizerFile)) {
                return tokenizerFile.toString();
            }

            // Stop if we've reached the root model directory
            if (currentDir.equals(rootNormalized)) {
                break;
            }

            // Move to parent directory
            Path parent = currentDir.getParent();
            if (parent == null || parent.equals(currentDir)) {
                break;
            }

            // Don't search above root
            if (!parent.startsWith(rootNormalized)) {
                break;
            }

            currentDir = parent;
        }

        return null; // Not found
    }

    /**
     * Checks if the model directory contains a decoder-only model.
     *
     * @param rootPath root model directory
     * @return true if this appears to be a decoder-only model
     */
    private boolean isDecoderOnlyModel(Path rootPath) {
        // Check for config.json to determine architecture
        Path configPath = rootPath.resolve("config.json");
        if (Files.exists(configPath)) {
            try {
                ModelConfigParser parser = new ModelConfigParser(configPath);
                ModelArchitecture arch = parser.detectArchitecture();
                return arch == ModelArchitecture.DECODER_ONLY;
            } catch (IOException e) {
                // Fall through to file-based detection
            }
        }

        // File-based detection: Check for decoder-only model patterns
        // Look for model files without "encoder" or "decoder" prefix
        try {
            if (Files.exists(rootPath)) {
                // Check in root and subdirectories for decoder-only model patterns
                List<Path> modelFiles = findDecoderOnlyModelFiles(rootPath);
                boolean hasDecoderOnlyFiles = !modelFiles.isEmpty();

                // Check for absence of encoder/decoder files
                boolean hasEncoderFiles = Files.walk(rootPath, 2)
                    .filter(Files::isRegularFile)
                    .anyMatch(p -> p.getFileName().toString().startsWith("encoder_model"));

                boolean hasDecoderFiles = Files.walk(rootPath, 2)
                    .filter(Files::isRegularFile)
                    .anyMatch(p -> p.getFileName().toString().startsWith("decoder_model"));

                // If we have model files but no encoder/decoder files, it's decoder-only
                return hasDecoderOnlyFiles && !hasEncoderFiles && !hasDecoderFiles;
            }
        } catch (IOException e) {
            // Default to encoder-decoder if we can't determine
        }

        return false;
    }

    /**
     * Discovers model files for decoder-only architectures (Llama, Phi-3, Qwen).
     *
     * @param rootPath root model directory
     * @param variant model variant (FULL, FP16, Q4, INT8)
     * @param flavour optional subdirectory filter
     * @return ModelFiles with discovered paths
     */
    private ModelFiles discoverDecoderOnlyModel(Path rootPath, ModelVariant variant, String flavour) {
        // Find the model file (decoder-only models have a single model file)
        Path modelFile = findDecoderOnlyModelFile(rootPath, variant, flavour);

        if (modelFile == null) {
            String expectedFileName = getDecoderOnlyModelFileName(variant);
            throw new IllegalStateException(
                "Decoder-only model file not found for variant " + variant + " in " + rootPath +
                (flavour != null ? " (flavour: " + flavour + ")" : "") +
                ". Expected patterns: " + expectedFileName +
                ". Use ModelConfig.decoderPath() to specify custom location.");
        }

        // Find tokenizer
        Path tokenizerPath = findTokenizerInDirectory(modelFile.getParent(), rootPath);
        if (tokenizerPath == null) {
            throw new IllegalStateException(
                "Tokenizer file (tokenizer.json) not found in " + rootPath +
                " or subdirectories. Use ModelConfig.tokenizerPath() to specify custom location.");
        }

        // For decoder-only models:
        // - encoder is the model file (for compatibility)
        // - decoder is also the model file
        // - decoderWithPast is null (single model handles all cases)
        return new ModelFiles(modelFile, modelFile, null, tokenizerPath);
    }

    /**
     * Finds decoder-only model file with optimized directory preference.
     *
     * @param rootPath root model directory
     * @param variant model variant
     * @param flavour optional subdirectory filter
     * @return path to model file, or null if not found
     */
    private Path findDecoderOnlyModelFile(Path rootPath, ModelVariant variant, String flavour) {
        // First try optimized directories
        Path optimizedFile = findDecoderOnlyModelInOptimizedDirs(rootPath, variant, flavour);
        if (optimizedFile != null) {
            return optimizedFile;
        }

        // Fall back to regular search
        return findDecoderOnlyModelInRegularDirs(rootPath, variant, flavour);
    }

    /**
     * Finds decoder-only model file in optimized directories.
     */
    private Path findDecoderOnlyModelInOptimizedDirs(Path rootPath, ModelVariant variant, String flavour) {
        String[] optimizedDirPatterns = {
            "onnx_optimized",
            "optimized",
            variant.name().toLowerCase() + "_optimized",
            "fp16_optimized",
            "int8_optimized",
            "q4_optimized"
        };

        List<String> modelPatterns = getDecoderOnlyModelPatterns(variant);

        for (String dirPattern : optimizedDirPatterns) {
            Path optimizedDir = rootPath.resolve(dirPattern);
            if (Files.isDirectory(optimizedDir) && !isEmptyDirectory(optimizedDir)) {
                for (String pattern : modelPatterns) {
                    Path candidate = optimizedDir.resolve(pattern);
                    if (Files.exists(candidate)) {
                        return candidate;
                    }
                }
            }
        }

        return null;
    }

    /**
     * Checks if a directory is empty or contains only subdirectories.
     */
    private boolean isEmptyDirectory(Path dir) {
        try {
            return !Files.walk(dir, 1)
                .filter(p -> !p.equals(dir))
                .filter(Files::isRegularFile)
                .findAny()
                .isPresent();
        } catch (IOException e) {
            return true;
        }
    }

    /**
     * Finds decoder-only model file in regular directories.
     */
    private Path findDecoderOnlyModelInRegularDirs(Path rootPath, ModelVariant variant, String flavour) {
        List<String> modelPatterns = getDecoderOnlyModelPatterns(variant);

        try {
            // First check root directory
            for (String pattern : modelPatterns) {
                Path candidate = rootPath.resolve(pattern);
                if (Files.exists(candidate)) {
                    return candidate;
                }
            }

            // Then check subdirectories (especially "onnx" subdirectory)
            if (Files.isDirectory(rootPath)) {
                List<Path> candidates = new ArrayList<>();

                Files.walk(rootPath, 2, FileVisitOption.FOLLOW_LINKS)
                    .filter(Files::isRegularFile)
                    .forEach(path -> {
                        String fileName = path.getFileName().toString();
                        for (String pattern : modelPatterns) {
                            if (fileName.equals(pattern)) {
                                // Apply flavour filter if specified
                                if (flavour == null || path.getParent().getFileName().toString().contains(flavour)) {
                                    candidates.add(path);
                                }
                            }
                        }
                    });

                if (!candidates.isEmpty()) {
                    // Prefer files in "onnx" subdirectory
                    for (Path candidate : candidates) {
                        if (candidate.getParent().getFileName().toString().equals("onnx")) {
                            return candidate;
                        }
                    }
                    // Otherwise return first match
                    return candidates.get(0);
                }
            }
        } catch (IOException e) {
            // Ignore and return null
        }

        return null;
    }

    /**
     * Gets possible model file name patterns for decoder-only models.
     *
     * @param variant model variant
     * @return list of possible file names to check
     */
    private List<String> getDecoderOnlyModelPatterns(ModelVariant variant) {
        List<String> patterns = new ArrayList<>();

        switch (variant) {
            case FULL:
                patterns.add("model.onnx");
                patterns.add("model_full.onnx");
                break;
            case FP16:
                patterns.add("model_fp16.onnx");
                patterns.add("model.fp16.onnx");
                patterns.add("model_f16.onnx");
                break;
            case Q4:
                patterns.add("model_q4.onnx");
                patterns.add("model_q4f16.onnx");
                patterns.add("model_bnb4.onnx");
                patterns.add("model.q4.onnx");
                patterns.add("model-q4.onnx");
                break;
            case INT8:
                patterns.add("model_int8.onnx");
                patterns.add("model_uint8.onnx");
                patterns.add("model_quantized.onnx");
                patterns.add("model.int8.onnx");
                patterns.add("model-int8.onnx");
                break;
        }

        return patterns;
    }

    /**
     * Gets expected model file name for error messages.
     */
    private String getDecoderOnlyModelFileName(ModelVariant variant) {
        return switch (variant) {
            case FULL -> "model.onnx";
            case FP16 -> "model_fp16.onnx or model.fp16.onnx";
            case Q4 -> "model_q4.onnx, model_q4f16.onnx, or model_bnb4.onnx";
            case INT8 -> "model_int8.onnx, model_uint8.onnx, or model_quantized.onnx";
        };
    }

    /**
     * Finds all decoder-only model files in a directory.
     */
    private List<Path> findDecoderOnlyModelFiles(Path rootPath) {
        List<Path> modelFiles = new ArrayList<>();
        Pattern modelPattern = Pattern.compile("^model[_.-]?(fp16|f16|int8|uint8|quantized|q4|q4f16|bnb4)?\\.onnx$");

        try {
            Files.walk(rootPath, 2)
                .filter(Files::isRegularFile)
                .filter(p -> {
                    String fileName = p.getFileName().toString();
                    return fileName.equals("model.onnx") || modelPattern.matcher(fileName).matches();
                })
                .forEach(modelFiles::add);
        } catch (IOException e) {
            // Return empty list
        }

        return modelFiles;
    }

    /**
     * Finds tokenizer.json by searching from start directory up to root.
     */
    private Path findTokenizerInDirectory(Path startDir, Path rootPath) {
        Path currentDir = startDir.toAbsolutePath().normalize();
        Path rootNormalized = rootPath.toAbsolutePath().normalize();

        while (currentDir != null) {
            Path tokenizerFile = currentDir.resolve("tokenizer.json");
            if (Files.exists(tokenizerFile)) {
                return tokenizerFile;
            }

            // Stop if we've reached the root model directory
            if (currentDir.equals(rootNormalized)) {
                // Check one more time in root
                tokenizerFile = rootNormalized.resolve("tokenizer.json");
                if (Files.exists(tokenizerFile)) {
                    return tokenizerFile;
                }
                break;
            }

            // Move to parent directory
            Path parent = currentDir.getParent();
            if (parent == null || parent.equals(currentDir)) {
                break;
            }

            // Don't search above root
            if (!parent.startsWith(rootNormalized)) {
                break;
            }

            currentDir = parent;
        }

        return null; // Not found
    }

    /**
     * Gets the encoder model file name based on variant.
     *
     * @param variant model variant
     * @return encoder file name
     */
    private String getEncoderFileName(ModelVariant variant) {
        return switch (variant) {
            case FULL -> "encoder_model.onnx";
            case FP16 -> "encoder_model_fp16.onnx";
            case Q4 -> "encoder_model_q4.onnx";
            case INT8 -> "encoder_model_int8.onnx";
        };
    }

    /**
     * Gets the merged decoder model file name based on variant.
     *
     * @param variant model variant
     * @return merged decoder file name
     */
    private String getMergedDecoderFileName(ModelVariant variant) {
        return switch (variant) {
            case FULL -> "decoder_model_merged.onnx";
            case FP16 -> "decoder_model_merged_fp16.onnx";
            case Q4 -> "decoder_model_merged_q4.onnx";
            case INT8 -> "decoder_model_merged_int8.onnx";
        };
    }

    /**
     * Gets the with_past decoder model file name based on variant.
     * These models have KV-cache support for 3-5x faster inference.
     *
     * @param variant model variant
     * @return with_past decoder file name
     */
    private String getWithPastDecoderFileName(ModelVariant variant) {
        return switch (variant) {
            case FULL -> "decoder_with_past_model.onnx";
            case FP16 -> "decoder_with_past_model_fp16.onnx";
            case Q4 -> "decoder_with_past_model_q4.onnx";
            case INT8 -> "decoder_with_past_model_int8.onnx";
        };
    }

    /**
     * Gets the non-merged decoder model file name based on variant.
     *
     * @param variant model variant
     * @return non-merged decoder file name
     */
    private String getNonMergedDecoderFileName(ModelVariant variant) {
        return switch (variant) {
            case FULL -> "decoder_model.onnx";
            case FP16 -> "decoder_model_fp16.onnx";
            case Q4 -> "decoder_model_q4.onnx";
            case INT8 -> "decoder_model_int8.onnx";
        };
    }

}
