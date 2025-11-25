package com.badu.ai.onnx.engine;

import com.badu.ai.onnx.config.ModelConfig;
import com.badu.ai.onnx.config.ModelConfigParser;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Factory for creating appropriate inference engine based on model architecture.
 *
 * <p>This factory provides two creation methods:
 * <ul>
 *   <li>{@link #createEngine(ModelArchitecture)}: Explicit engine selection</li>
 *   <li>{@link #autoDetect(ModelConfig)}: Auto-detect based on model files present</li>
 * </ul>
 *
 * <p><b>Auto-detection logic:</b>
 * <ol>
 *   <li>If encoder AND decoder models present → T5EncoderDecoderEngine</li>
 *   <li>Otherwise → SimpleGenAIEngine (legacy fallback)</li>
 * </ol>
 *
 * <p>Usage example:
 * <pre>{@code
 * // Explicit engine selection
 * InferenceEngine engine = InferenceEngineFactory.createEngine(ModelArchitecture.T5_ENCODER_DECODER);
 * engine.initialize(modelConfig, genConfig);
 *
 * // Auto-detection based on model files
 * InferenceEngine engine = InferenceEngineFactory.autoDetect(modelConfig);
 * engine.initialize(modelConfig, genConfig);
 * }</pre>
 */
public class InferenceEngineFactory {

    private static final Logger logger = LoggerFactory.getLogger(InferenceEngineFactory.class);

    /**
     * Creates an inference engine for the specified architecture.
     *
     * @param architecture model architecture type
     * @return new inference engine instance (not initialized)
     * @throws IllegalArgumentException if architecture is null or unsupported
     */
    public static InferenceEngine createEngine(ModelArchitecture architecture) {
        if (architecture == null) {
            throw new IllegalArgumentException("ModelArchitecture cannot be null");
        }

        return switch (architecture) {
            case T5_ENCODER_DECODER -> {
                logger.info("Creating T5 encoder-decoder engine");
                yield createEncoderDecoderEngine();
            }
            case DECODER_ONLY -> {
                logger.info("Creating Decoder-only engine (Phi-3/Llama/Qwen)");
                yield createDecoderOnlyEngine();
            }
        };
    }

    /**
     * Auto-detects model architecture based on config.json or files present in model directory.
     *
     * <p>Detection logic:
     * <ol>
     *   <li><b>Primary:</b> Parse config.json if present and use ModelConfigParser to detect architecture</li>
     *   <li><b>Fallback:</b> File-based detection:
     *     <ul>
     *       <li>If encoder and decoder models exist → T5EncoderDecoderEngine</li>
     *       <li>Otherwise → SimpleGenAIEngine (legacy fallback)</li>
     *     </ul>
     *   </li>
     * </ol>
     *
     * <p>File patterns checked (fallback only):
     * <ul>
     *   <li>Encoder: encoder_model*.onnx</li>
     *   <li>Decoder: decoder_model*.onnx</li>
     * </ul>
     *
     * @param config model configuration with model path
     * @return new inference engine instance (not initialized)
     * @throws IllegalArgumentException if config is null or model path invalid
     */
    public static InferenceEngine autoDetect(ModelConfig config) {
        if (config == null) {
            throw new IllegalArgumentException("ModelConfig cannot be null");
        }

        String modelPath = config.getModelPath();
        if (modelPath == null || modelPath.trim().isEmpty()) {
            throw new IllegalArgumentException("Model path cannot be null or empty");
        }

        Path modelDir = Paths.get(modelPath);

        if (!Files.exists(modelDir)) {
            logger.warn("Model directory does not exist: {}. Falling back to SimpleGenAI engine.", modelPath);
            return createSimpleGenAIEngine();
        }

        if (!Files.isDirectory(modelDir)) {
            logger.warn("Model path is not a directory: {}. Falling back to SimpleGenAI engine.", modelPath);
            return createSimpleGenAIEngine();
        }

        // Primary detection: Try config.json parsing
        Path configPath = modelDir.resolve("config.json");
        if (Files.exists(configPath)) {
            try {
                ModelConfigParser parser = new ModelConfigParser(configPath);
                ModelArchitecture arch = parser.detectArchitecture();
                logger.info("Detected architecture from config.json: {} -> {}", modelPath, arch);
                return createEngine(arch);
            } catch (IOException e) {
                logger.warn("Failed to parse config.json, falling back to file-based detection: {}", e.getMessage());
            }
        } else {
            logger.debug("No config.json found at {}, using file-based detection", configPath);
        }

        // Fallback: File-based detection
        boolean hasEncoder = hasEncoderModel(config);
        boolean hasDecoder = hasDecoderModel(config);

        if (hasEncoder && hasDecoder) {
            logger.trace("File-based detection: T5 encoder-decoder architecture at {}", modelPath);
            return createEncoderDecoderEngine();
        }

        logger.info("File-based detection: SimpleGenAI architecture at {} (encoder: {}, decoder: {})",
            modelPath, hasEncoder, hasDecoder);
        return createSimpleGenAIEngine();
    }

    private static InferenceEngine createSimpleGenAIEngine() {
        throw new IllegalArgumentException("SimpleGenAIEngine is not implemented"); 
    }

    private static InferenceEngine createEncoderDecoderEngine() {
        return new EncoderDecoderEngine();
    }

    private static InferenceEngine createDecoderOnlyEngine() {
        return new DecoderOnlyEngine();
    }

    /**
     * Checks if encoder model file exists for the given configuration.
     */
    private static boolean hasEncoderModel(ModelConfig config) {
        Path encoderPath = config.getEncoderPath();
        if (encoderPath != null) {
            // Explicit encoder path provided
            return Files.exists(encoderPath);
        }

        // Check for encoder model in model directory
        Path modelDir = Paths.get(config.getModelPath());
        String variant = config.getVariant().name().toLowerCase();
        String flavour = config.getFlavour();

        // Check multiple patterns
        String[] patterns = {
            "encoder_model.onnx",
            "encoder_model_" + variant + ".onnx",
            "encoder_model." + flavour,
            "encoder." + flavour
        };

        for (String pattern : patterns) {
            Path candidatePath = modelDir.resolve(pattern);
            if (Files.exists(candidatePath)) {
                logger.debug("Found encoder model: {}", candidatePath);
                return true;
            }
        }

        return false;
    }

    /**
     * Checks if decoder model file exists for the given configuration.
     */
    private static boolean hasDecoderModel(ModelConfig config) {
        Path decoderPath = config.getDecoderPath();
        if (decoderPath != null) {
            // Explicit decoder path provided
            return Files.exists(decoderPath);
        }

        // Check for decoder model in model directory
        Path modelDir = Paths.get(config.getModelPath());
        String variant = config.getVariant().name().toLowerCase();
        String flavour = config.getFlavour();

        // Check multiple patterns
        String[] patterns = {
            "decoder_model.onnx",
            "decoder_model_" + variant + ".onnx",
            "decoder_model." + flavour,
            "decoder." + flavour
        };

        for (String pattern : patterns) {
            Path candidatePath = modelDir.resolve(pattern);
            if (Files.exists(candidatePath)) {
                logger.debug("Found decoder model: {}", candidatePath);
                return true;
            }
        }

        return false;
    }
}
