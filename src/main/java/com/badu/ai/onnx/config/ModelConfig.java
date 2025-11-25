package com.badu.ai.onnx.config;

import com.badu.ai.onnx.utils.ModelUtils;
import lombok.Builder;
import lombok.Value;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Immutable configuration for ONNX model loading and initialization.
 *
 * <p>Specifies model paths, quantization variant, tokenizer location, and token limits.
 *
 * <p>Builder pattern usage:
 * <pre>{@code
 * // CPU inference with INT8
 * ModelConfig config = ModelConfig.builder()
 *     .modelPath("models/flan-t5-small-ONNX")
 *     .variant(ModelVariant.INT8)
 *     .maxInputTokens(8192)
 *     .build();
 *
 * // GPU inference with FP16 (recommended for 2-3x speedup)
 * ModelConfig gpuConfig = ModelConfig.builder()
 *     .modelPath("models/flan-t5-small-ONNX/fp16")
 *     .variant(ModelVariant.FP16)
 *     .deviceType(DeviceType.GPU)
 *     .gpuDeviceId(0)  // Use first GPU (default)
 *     .maxInputTokens(8192)
 *     .build();
 *
 * // Multi-GPU: Specify which GPU to use
 * ModelConfig multiGpuConfig = ModelConfig.builder()
 *     .modelPath("models/flan-t5-small-ONNX/fp16")
 *     .variant(ModelVariant.FP16)
 *     .deviceType(DeviceType.GPU)
 *     .gpuDeviceId(1)  // Use second GPU
 *     .build();
 * }</pre>
 *
 * @see ModelVariant
 * @see DeviceType
 */
@Value
@Builder
public class ModelConfig {

  /**
   * Root directory containing model files.
   * For T5 models: encoder_model.onnx and decoder_model.onnx may be in subdirectories.
   * Example: "models/flan-t5-small-ONNX"
   */
  String modelPath;

  /**
   * Quantization variant (FULL, FP16, Q4, INT8).
   * Default: INT8 (8-bit quantization for good balance).
   * Recommended: FP16 for GPU inference with Tensor Cores (2-3x speedup).
   */
  @Builder.Default
  ModelVariant variant = ModelVariant.INT8;

  /**
   * Optional subdirectory flavour for model files.
   * Used when multiple model variants exist in subdirectories (e.g., "onnx", "cpu", "gpu").
   * If null, searches all subdirectories. If multiple candidates found, throws exception.
   */
  String flavour;

  /**
   * Absolute path to encoder ONNX model file.
   * Discovered during build() by searching from modelPath.
   */
  String encoderPath;

  /**
   * Absolute path to decoder ONNX model file (first step).
   * Discovered during build() by searching from modelPath.
   */
  String decoderPath;

  /**
   * Absolute path to decoder with past ONNX model file (subsequent steps with KV-cache).
   * Optional - if null, decoderPath is used for all steps.
   * Discovered during build() if available (dual-session mode for optimal KV-cache performance).
   */
  String decoderWithPastPath;

  /**
   * Path to HuggingFace tokenizer.json file.
   * If not specified, discovered from modelPath or parent directories.
   */
  String tokenizerPath;

  /**
   * Maximum input tokens (reject longer inputs).
   * Default: 40960 (from T5 max_position_embeddings)
   */
  @Builder.Default
  int maxInputTokens = 40960;

  /**
   * Maximum output tokens to generate.
   * Default: 512 (suitable for summarization)
   */
  @Builder.Default
  int maxOutputTokens = 512;

  /**
   * Execution device (CPU or GPU).
   * Default: CPU (works everywhere, no special dependencies)
   */
  @Builder.Default
  DeviceType deviceType = DeviceType.CPU;

  /**
   * GPU device ID for multi-GPU systems.
   * Specifies which GPU to use when deviceType=GPU.
   * Default: 0 (first GPU)
   *
   * <p>For NVIDIA CUDA: Use nvidia-smi to list available GPUs
   * <p>For AMD ROCm: Set HIP_VISIBLE_DEVICES environment variable
   * <p>For Apple Silicon: Always 0 (integrated GPU)
   */
  @Builder.Default
  int gpuDeviceId = 0;

  @Builder.Default
  int threadsCount = ModelUtils.getOptimalThreadCount();

  /**
   * Enable ONNX Runtime profiling for performance debugging.
   * When enabled, generates profiling JSON files with detailed timing information.
   * Default: false (minimal overhead for production use)
   *
   * <p>Profiling output location: onnx_profile_{timestamp}.json
   */
  @Builder.Default
  boolean enableProfiling = false;

  /**
   * Enable detailed diagnostic logging for KV-cache and tensor flow debugging.
   * When enabled, logs detailed information about decoder inputs/outputs, cache usage, and tensor names.
   * Default: false (minimal logging for production use)
   *
   * <p>Diagnostic logs prefixed with: [DIAG]
   */
  @Builder.Default
  boolean enableDiagnostics = false;

  /**
   * Enable IO Bindings for GPU inference optimization.
   * When enabled, pre-allocates tensors on GPU memory to eliminate CPU↔GPU memory copies.
   * Requires deviceType=GPU. Expected performance improvement: +20-30% GPU throughput.
   * Default: false (standard mode, works with both CPU and GPU)
   *
   * <p>IO Bindings are automatically disabled if:
   * <ul>
   *   <li>deviceType is CPU (IO Bindings only benefit GPU)</li>
   *   <li>ONNX Runtime GPU provider is not available</li>
   * </ul>
   *
   * <p>Recommended for GPU inference with FP16 models on NVIDIA GPUs with Tensor Cores.
   */
  @Builder.Default
  boolean enableIoBindings = false;

  /**
   * Enable memory arena optimization for reduced memory fragmentation.
   * When enabled, uses device allocator for initializers and enables arena shrinkage.
   * Expected impact: 10-15% memory reduction.
   * Default: true (recommended for production use)
   *
   * <p>Memory arena helps reduce memory fragmentation by:
   * <ul>
   *   <li>Using device allocator for model initializers (weights)</li>
   *   <li>Enabling arena shrinkage to release unused memory</li>
   *   <li>Controlling arena growth strategy</li>
   * </ul>
   */
  @Builder.Default
  boolean enableMemoryArena = true;

  /**
   * Memory arena size in bytes (0 = auto-calculate based on model size).
   * Default: 256MB (268435456 bytes)
   *
   * <p>Recommended values by model size:
   * <ul>
   *   <li>Small models (Flan-T5-Small, &lt;1GB): 256MB</li>
   *   <li>Medium models (1-3GB): 512MB</li>
   *   <li>Large models (&gt;3GB): 1GB+</li>
   * </ul>
   *
   * <p>Set to 0 to let ONNX Runtime automatically determine the size.
   */
  @Builder.Default
  long memoryArenaSize = 256 * 1024 * 1024; // 256MB

  /**
   * Number of key-value heads for grouped query attention (decoder-only models).
   * Used to initialize KV-cache tensors with correct shape.
   *
   * <p>For decoder-only models (Llama, Phi-3):
   * <ul>
   *   <li>Llama 3.2 1B: 8 key-value heads (with 32 attention heads)</li>
   *   <li>Phi-3 Mini: varies by model size</li>
   * </ul>
   *
   * <p>If null, value will be parsed from genai_config.json during initialization.
   * For T5/BART encoder-decoder models, this field is not used.
   */
  Integer numKeyValueHeads;

  /**
   * Dimension per attention head (decoder-only models).
   * Used to initialize KV-cache tensors with correct shape.
   *
   * <p>Common values:
   * <ul>
   *   <li>Llama 3.2 1B: 64</li>
   *   <li>Phi-3 Mini: 96</li>
   * </ul>
   *
   * <p>If null, value will be parsed from genai_config.json during initialization.
   * For T5/BART encoder-decoder models, this field is not used.
   */
  Integer headSize;

  /**
   * Custom builder with validation logic.
   */
  public static class ModelConfigBuilder {
    /**
     * Builds the ModelConfig with validation.
     *
     * @return validated ModelConfig instance
     * @throws IllegalStateException if validation fails
     */
    public ModelConfig build() {
      // Required field validation
      validateRequired();

      // Model directory must exist
      Path rootPath = Paths.get(modelPath);
      validateRootPath(rootPath);

      // Discover or validate model files
      ModelFiles files = discoverModelFiles(rootPath);

      // Set defaults and validate token limits
      applyDefaults();
      validateTokenLimits();

      return new ModelConfig(modelPath, this.variant$value, this.flavour,
          files.encoder().toString(), files.decoder().toString(),
          files.decoderWithPast() != null ? files.decoderWithPast().toString() : null,
          files.tokenizer().toString(),
          this.maxInputTokens$value, this.maxOutputTokens$value, this.deviceType$value,
          this.gpuDeviceId$value, this.threadsCount$value, this.enableProfiling$value,
          this.enableDiagnostics$value, this.enableIoBindings$value,
          this.enableMemoryArena$value, this.memoryArenaSize$value,
          this.numKeyValueHeads, this.headSize);
    }

    private void validateRequired() {
      if (modelPath == null || modelPath.trim().isEmpty()) {
        throw new IllegalStateException("modelPath is required");
      }

      if (this.variant$value == null) {
        this.variant$value = ModelVariant.INT8;
        this.variant$set = true;
      }
    }

    private void validateRootPath(Path rootPath) {
      if (!Files.exists(rootPath)) {
        throw new IllegalStateException("modelPath does not exist: " + modelPath);
      }
      if (!Files.isDirectory(rootPath)) {
        throw new IllegalStateException("modelPath must be a directory: " + modelPath);
      }
    }

    private ModelFiles discoverModelFiles(Path rootPath) {
      boolean hasExplicitPaths = (this.encoderPath != null && !this.encoderPath.trim().isEmpty())
          || (this.decoderPath != null && !this.decoderPath.trim().isEmpty())
          || (this.tokenizerPath != null && !this.tokenizerPath.trim().isEmpty());

      if (hasExplicitPaths) {
        // Use explicit paths with validation
        return useExplicitPaths(rootPath);
      } else {
        // Auto-discover files
        ModelFileDiscovery discovery = new ModelFileDiscovery();
        return discovery.discover(rootPath, this.variant$value, this.flavour);
      }
    }

    private ModelFiles useExplicitPaths(Path rootPath) {
      Path encoderFile;
      Path decoderFile;
      Path tokenizerFile;

      // Handle encoder path
      if (this.encoderPath != null && !this.encoderPath.trim().isEmpty()) {
        encoderFile = Paths.get(this.encoderPath);
        if (!Files.exists(encoderFile)) {
          throw new IllegalStateException(
              "Explicitly provided encoder path does not exist: " + this.encoderPath);
        }
      } else {
        // Auto-discover encoder only
        ModelFileDiscovery discovery = new ModelFileDiscovery();
        ModelFiles discovered = discovery.discover(rootPath, this.variant$value, this.flavour);
        encoderFile = discovered.encoder();
      }

      // Handle decoder path
      if (this.decoderPath != null && !this.decoderPath.trim().isEmpty()) {
        decoderFile = Paths.get(this.decoderPath);
        if (!Files.exists(decoderFile)) {
          throw new IllegalStateException(
              "Explicitly provided decoder path does not exist: " + this.decoderPath);
        }
      } else {
        // Auto-discover decoder only
        ModelFileDiscovery discovery = new ModelFileDiscovery();
        ModelFiles discovered = discovery.discover(rootPath, this.variant$value, this.flavour);
        decoderFile = discovered.decoder();
      }

      // Handle tokenizer path
      if (this.tokenizerPath != null && !this.tokenizerPath.trim().isEmpty()) {
        tokenizerFile = Paths.get(this.tokenizerPath);
        if (!Files.exists(tokenizerFile)) {
          throw new IllegalStateException(
              "Tokenizer file not found: " + this.tokenizerPath +
              ". Use ModelConfig.tokenizerPath() to specify custom location.");
        }
      } else {
        // Auto-discover tokenizer only (search from encoder directory upwards)
        Path encoderDir = encoderFile.getParent();
        tokenizerFile = findTokenizerInDirectory(encoderDir, rootPath);
        if (tokenizerFile == null) {
          throw new IllegalStateException(
              "Tokenizer file (tokenizer.json) not found in " + rootPath +
              " or subdirectories. Use ModelConfig.tokenizerPath() to specify custom location.");
        }
      }

      // For backward compatibility when using explicit paths, decoderWithPast is null
      return new ModelFiles(encoderFile, decoderFile, null, tokenizerFile);
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

    private void applyDefaults() {
      if (!this.maxInputTokens$set) {
        this.maxInputTokens$value = 40960;
        this.maxInputTokens$set = true;
      }

      if (!this.maxOutputTokens$set) {
        this.maxOutputTokens$value = 512;
        this.maxOutputTokens$set = true;
      }

      if (!this.deviceType$set) {
        this.deviceType$value = DeviceType.CPU;
        this.deviceType$set = true;
      }

      if (!this.gpuDeviceId$set) {
        this.gpuDeviceId$value = 0;
        this.gpuDeviceId$set = true;
      }

      if (!this.enableMemoryArena$set) {
        this.enableMemoryArena$value = true;
        this.enableMemoryArena$set = true;
      }

      if (!this.memoryArenaSize$set) {
        this.memoryArenaSize$value = 256 * 1024 * 1024; // 256MB
        this.memoryArenaSize$set = true;
      }
    }

    private void validateTokenLimits() {
      if (this.maxInputTokens$value <= 0) {
        throw new IllegalStateException("maxInputTokens must be positive");
      }
      if (this.maxInputTokens$value > 40960) {
        throw new IllegalStateException("maxInputTokens exceeds model maximum: 40960");
      }

      if (this.maxOutputTokens$value <= 0) {
        throw new IllegalStateException("maxOutputTokens must be positive");
      }
      if (this.maxOutputTokens$value > 4096) {
        throw new IllegalStateException("maxOutputTokens exceeds maximum: 4096");
      }
    }

  }

  /**
   * Gets the encoder model file path.
   * Path is discovered during ModelConfig.build() by searching from modelPath.
   *
   * @return absolute path to encoder model file
   */
  public Path getEncoderPath() {
    return Paths.get(encoderPath);
  }

  /**
   * Gets the decoder model file path.
   * Path is discovered during ModelConfig.build() by searching from modelPath.
   *
   * @return absolute path to decoder model file
   */
  public Path getDecoderPath() {
    return Paths.get(decoderPath);
  }

  /**
   * Gets the decoder with past model file path.
   * Optional - if null, decoderPath is used for all steps (single-session mode).
   *
   * @return absolute path to decoder with past model file, or null if not available
   */
  public Path getDecoderWithPastPath() {
    return decoderWithPastPath != null ? Paths.get(decoderWithPastPath) : null;
  }

  /**
   * Gets the tokenizer file path.
   *
   * @return absolute path to tokenizer.json file
   */
  public Path getTokenizerPathAsPath() {
    return Paths.get(tokenizerPath);
  }

  /**
   * Checks if dual-session decoder mode is enabled.
   * Dual-session mode uses separate decoder models for first step (decoder_model.onnx)
   * and subsequent steps (decoder_with_past_model.onnx) for optimal KV-cache performance.
   *
   * @return true if decoderWithPastPath is not null (dual-session), false otherwise (single-session)
   */
  public boolean hasDualDecoder() {
    return decoderWithPastPath != null;
  }
}
