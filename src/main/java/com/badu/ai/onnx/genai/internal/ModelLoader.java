package com.badu.ai.onnx.genai.internal;

import ai.onnxruntime.*;
import com.badu.ai.onnx.config.ModelConfig;
import com.badu.ai.onnx.utils.ModelUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Path;
import java.util.Set;

/**
 * Loads ONNX model sessions for T5 encoder-decoder architecture.
 *
 * <p>Manages ONNX Runtime environment and creates sessions for:
 * <ul>
 *   <li>Encoder model (encoder_model.onnx or quantized variants)</li>
 *   <li>Decoder model (decoder_model.onnx or quantized variants)</li>
 * </ul>
 *
 * <p>Supports:
 * <ul>
 *   <li>CPU and GPU execution providers</li>
 *   <li>Multiple quantization levels (FULL, Q4, INT8)</li>
 *   <li>KV-cache detection for optimized decoding</li>
 * </ul>
 *
 * <p>Usage:
 * <pre>{@code
 * ModelConfig config = ModelConfig.builder()
 *     .modelPath("models/flan-t5-small-ONNX")
 *     .variant(ModelVariant.INT8)
 *     .build();
 *
 * try (ModelLoader loader = new ModelLoader(config)) {
 *     OrtSession encoder = loader.getEncoderSession();
 *     OrtSession decoder = loader.getDecoderSession();
 *     // Use sessions for inference
 * }
 * }</pre>
 */
public class ModelLoader implements AutoCloseable {

  private static final Logger logger = LoggerFactory.getLogger(ModelLoader.class);

  // Singleton ORT environment (shared across all model loaders)
  private static OrtEnvironment environment;
  private static final Object ENV_LOCK = new Object();

  private final ModelConfig config;
  private final OrtSession.SessionOptions encoderOptions;
  private final OrtSession.SessionOptions decoderOptions;
  private OrtSession encoderSession;
  private OrtSession decoderSession;
  private OrtSession decoderWithPastSession;  // Optional: for dual-session mode
  private boolean hasKVCache;
  private boolean hasMergedDecoder;
  private boolean hasDualDecoder;
  private boolean hasEncoderHiddenStatesInput;  // True if decoder_with_past expects encoder_hidden_states as separate input
  private boolean hasPositionIdsInput;  // True if decoder requires position_ids (Qwen models)

  /**
   * Creates a ModelLoader with the specified configuration.
   *
   * @param config Model configuration
   * @throws OrtException if model loading fails
   */
  public ModelLoader(ModelConfig config) throws OrtException {
    this.config = config;

    logger.info("Loading ONNX models from: {}, variant: {}, device: {}",
        config.getModelPath(), config.getVariant(), config.getDeviceType());

    // Initialize ORT environment (thread-safe singleton)
    initializeEnvironment();

    // Create separate session options for encoder and decoder
    // Encoder benefits from parallel execution (many attention heads)
    // Decoder benefits from sequential execution (auto-regressive generation)
    this.encoderOptions = ModelUtils.createEncoderOptions(config);
    this.decoderOptions = ModelUtils.createDecoderOptions(config);

    // Load encoder and decoder sessions
    try {
      loadEncoderSession();
      loadDecoderSession();
      loadDecoderWithPastSession();  // Load dual decoder if available
      detectKVCache();
    } catch (OrtException e) {
      // Clean up on failure
      close();
      throw e;
    }

    if (hasDualDecoder) {
      logger.info("Model loaded successfully with DUAL-SESSION decoder (optimal KV-cache performance)");
      logger.info("  - Decoder (first step): {}", config.getDecoderPath().getFileName());
      logger.info("  - Decoder with past (cached steps): {}", config.getDecoderWithPastPath().getFileName());
    } else {
      logger.debug("Model loaded successfully. KV-cache supported: {}", hasKVCache);
    }
  }

  /**
   * Initializes the ONNX Runtime environment (singleton).
   */
  private static void initializeEnvironment() {
    synchronized (ENV_LOCK) {
      if (environment == null) {
        environment = OrtEnvironment.getEnvironment();
        logger.info("ONNX Runtime environment initialized");
      }
    }
  }

  /**
   * Loads the encoder ONNX session.
   */
  private void loadEncoderSession() throws OrtException {
    Path encoderPath = config.getEncoderPath();

    logger.debug("Loading encoder from: {}", encoderPath);

    try {
      this.encoderSession = environment.createSession(
          encoderPath.toString(),
          encoderOptions  // Use encoder-specific options (parallel execution)
      );

      logger.trace("Encoder loaded: {} inputs, {} outputs",
          encoderSession.getInputNames().size(),
          encoderSession.getOutputNames().size());

    } catch (OrtException e) {
      throw new OrtException("Failed to load encoder model from: " + encoderPath + ". " + e.getMessage());
    }
  }

  /**
   * Loads the decoder ONNX session.
   */
  private void loadDecoderSession() throws OrtException {
    Path decoderPath = config.getDecoderPath();

    logger.debug("Loading decoder from: {}", decoderPath);

    try {
      this.decoderSession = environment.createSession(
          decoderPath.toString(),
          decoderOptions  // Use decoder-specific options (sequential execution)
      );

      logger.trace("Decoder loaded: {} inputs, {} outputs",
          decoderSession.getInputNames().size(),
          decoderSession.getOutputNames().size());

    } catch (OrtException e) {
      throw new OrtException("Failed to load decoder model from: " + decoderPath + ". " + e.getMessage());
    }
  }

  /**
   * Loads the decoder with past ONNX session (optional, for dual-session mode).
   * If config.getDecoderWithPastPath() is null, this is a no-op.
   */
  private void loadDecoderWithPastSession() throws OrtException {
    Path decoderWithPastPath = config.getDecoderWithPastPath();

    if (decoderWithPastPath == null) {
      this.decoderWithPastSession = null;
      this.hasDualDecoder = false;
      logger.debug("Dual-session decoder not available (using single decoder for all steps)");
      return;
    }

    logger.debug("Loading decoder with past from: {}", decoderWithPastPath);

    try {
      this.decoderWithPastSession = environment.createSession(
          decoderWithPastPath.toString(),
          decoderOptions  // Use decoder-specific options (sequential execution)
      );

      this.hasDualDecoder = true;

      logger.trace("Decoder with past loaded: {} inputs, {} outputs",
          decoderWithPastSession.getInputNames().size(),
          decoderWithPastSession.getOutputNames().size());

    } catch (OrtException e) {
      throw new OrtException("Failed to load decoder with past model from: " + decoderWithPastPath + ". " + e.getMessage());
    }
  }

  /**
   * Detects if the decoder model supports KV-cache (past_key_values) and merged decoder.
   *
   * <p>KV-cache support is detected by checking for inputs/outputs named:
   * <ul>
   *   <li>past_key_values (input)</li>
   *   <li>present_key_values (output)</li>
   * </ul>
   *
   * <p>Merged decoder is detected by checking for use_cache_branch input.
   * Merged decoders combine decoder_model and decoder_with_past_model into one,
   * using a boolean input to switch between first step (false) and subsequent steps (true).
   *
   * <p>For dual-session mode, we check the decoderWithPastSession for KV-cache support
   * since that's the session that will have cache inputs/outputs.
   */
  private void detectKVCache() {
    try {
      // Use decoderWithPastSession if available (dual-session mode), otherwise use decoderSession
      OrtSession sessionToCheck = hasDualDecoder ? decoderWithPastSession : decoderSession;
      Set<String> inputNames = sessionToCheck.getInputNames();
      Set<String> outputNames = sessionToCheck.getOutputNames();

      // Detect merged decoder (has use_cache_branch input)
      this.hasMergedDecoder = inputNames.stream()
          .anyMatch(name -> name.equals("use_cache_branch"));

      // Detect if decoder expects encoder_hidden_states as separate input
      // Old format (optimum-cli 1.x): encoder_hidden_states is separate input (35 inputs total)
      // New format (optimum-cli 2.0.0): encoder_hidden_states embedded in past_key_values (34 inputs total)
      this.hasEncoderHiddenStatesInput = inputNames.contains("encoder_hidden_states");

      // Detect if decoder requires position_ids (Qwen models)
      this.hasPositionIdsInput = inputNames.contains("position_ids");

      // Count actual KV-cache inputs (must start with "past_key_values.")
      long pastKVInputCount = inputNames.stream()
          .filter(name -> name.startsWith("past_key_values."))
          .count();

      // Count KV-cache outputs (must start with "present." or "past_key_values.")
      long presentKVOutputCount = outputNames.stream()
          .filter(name -> name.startsWith("present.") || name.startsWith("past_key_values."))
          .count();

      // KV-cache enabled if decoder has past/present inputs/outputs
      // Note: Non-merged decoders (decoder_model.onnx) don't have KV-cache
      // Non-merged with-past decoders (decoder_with_past_model.onnx) do have KV-cache
      this.hasKVCache = (pastKVInputCount > 0 && presentKVOutputCount > 0);

      if (hasMergedDecoder) {
        logger.trace("Merged decoder detected with use_cache_branch input");
        if (hasKVCache) {
          logger.trace("  - KV-cache enabled with {} past inputs, {} present outputs",
              pastKVInputCount, presentKVOutputCount);
        } else {
          logger.trace("  - No KV-cache support (merged decoder without past/present tensors)");
        }
      } else if (hasKVCache) {
        logger.trace("KV-cache detected: {} past inputs, {} present outputs",
            pastKVInputCount, presentKVOutputCount);
      } else {
        logger.trace("No KV-cache support (past inputs: {}, present outputs: {}) - will compute all keys/values each step",
            pastKVInputCount, presentKVOutputCount);
      }

      logger.trace("Decoder inputs: {}", inputNames);
      logger.trace("Decoder outputs: {}", outputNames);

      if (hasPositionIdsInput) {
        logger.info("position_ids input detected (Qwen model) - will provide position tensors");
      }

    } catch (Exception e) {
      logger.warn("Failed to detect KV-cache support: {}", e.getMessage());
      this.hasKVCache = false;
      this.hasMergedDecoder = false;
      this.hasPositionIdsInput = false;
    }
  }

  /**
   * Gets the encoder session.
   *
   * @return Encoder OrtSession
   */
  public OrtSession getEncoderSession() {
    return encoderSession;
  }

  /**
   * Gets the decoder session.
   *
   * @return Decoder OrtSession
   */
  public OrtSession getDecoderSession() {
    return decoderSession;
  }

  /**
   * Checks if the decoder supports KV-cache.
   *
   * @return true if KV-cache is supported
   */
  public boolean hasKVCacheSupport() {
    return hasKVCache;
  }

  /**
   * Gets the decoder with past session (for dual-session mode).
   *
   * @return Decoder with past OrtSession, or null if not available
   */
  public OrtSession getDecoderWithPastSession() {
    return decoderWithPastSession;
  }

  /**
   * Checks if the decoder is a merged decoder (has use_cache_branch input).
   *
   * @return true if decoder is merged
   */
  public boolean hasMergedDecoder() {
    return hasMergedDecoder;
  }

  /**
   * Checks if dual-session decoder mode is enabled.
   *
   * @return true if decoder with past session is loaded (dual-session mode)
   */
  public boolean hasDualDecoder() {
    return hasDualDecoder;
  }

  /**
   * Checks if decoder_with_past model expects encoder_hidden_states as a separate input.
   *
   * <p>This is a compatibility check for different model export formats:
   * <ul>
   *   <li><b>Old format (optimum-cli 1.x):</b> encoder_hidden_states is a separate input (35 inputs total)</li>
   *   <li><b>New format (optimum-cli 2.0.0+):</b> encoder_hidden_states embedded in past_key_values (34 inputs total)</li>
   * </ul>
   *
   * <p>The code should conditionally add encoder_hidden_states to the input map only if this returns true.
   *
   * @return true if model expects encoder_hidden_states as separate input, false if embedded in KV-cache
   */
  public boolean hasEncoderHiddenStatesInput() {
    return hasEncoderHiddenStatesInput;
  }

  /**
   * Checks if decoder requires position_ids input.
   *
   * <p>Some models (Qwen2, Qwen3) require position_ids as an input to track token positions
   * during auto-regressive generation. This method detects if the decoder ONNX model expects
   * this input.
   *
   * <p><b>Models requiring position_ids:</b>
   * <ul>
   *   <li>Qwen2 (all variants)</li>
   *   <li>Qwen3 (all variants)</li>
   * </ul>
   *
   * <p><b>Models NOT requiring position_ids:</b>
   * <ul>
   *   <li>Llama 2/3 (all variants)</li>
   *   <li>Phi-3 (all variants)</li>
   *   <li>T5, BART (encoder-decoder models)</li>
   * </ul>
   *
   * @return true if model requires position_ids input, false otherwise
   */
  public boolean hasPositionIdsInput() {
    return hasPositionIdsInput;
  }

  /**
   * Gets the ONNX Runtime environment.
   *
   * @return OrtEnvironment instance
   */
  public static OrtEnvironment getEnvironment() {
    initializeEnvironment();
    return environment;
  }

  /**
   * Closes the model loader and releases ONNX sessions.
   *
   * <p>Note: Does not close the global OrtEnvironment (shared across loaders).
   */
  @Override
  public void close() {
    // End profiling and write files if enabled
    if (config.isEnableProfiling()) {
      try {
        if (encoderSession != null) {
          String encoderProfile = encoderSession.endProfiling();
          logger.info("Encoder profiling data written to: {}", encoderProfile);
        }
        if (decoderSession != null) {
          String decoderProfile = decoderSession.endProfiling();
          logger.info("Decoder profiling data written to: {}", decoderProfile);
        }
        if (decoderWithPastSession != null) {
          String decoderWithPastProfile = decoderWithPastSession.endProfiling();
          logger.info("Decoder with past profiling data written to: {}", decoderWithPastProfile);
        }
      } catch (OrtException e) {
        logger.warn("Failed to end profiling: {}", e.getMessage());
      }
    }

    if (encoderSession != null) {
      try {
        encoderSession.close();
        logger.debug("Encoder session closed");
      } catch (IllegalStateException e) {
        logger.debug("Encoder session already closed");
      } catch (OrtException e) {
        logger.warn("Error closing encoder session: {}", e.getMessage());
      } finally {
        encoderSession = null;
      }
    }

    if (decoderSession != null) {
      try {
        decoderSession.close();
        logger.debug("Decoder session closed");
      } catch (IllegalStateException e) {
        logger.debug("Decoder session already closed");
      } catch (OrtException e) {
        logger.warn("Error closing decoder session: {}", e.getMessage());
      } finally {
        decoderSession = null;
      }
    }

    if (decoderWithPastSession != null) {
      try {
        decoderWithPastSession.close();
        logger.debug("Decoder with past session closed");
      } catch (IllegalStateException e) {
        logger.debug("Decoder with past session already closed");
      } catch (OrtException e) {
        logger.warn("Error closing decoder with past session: {}", e.getMessage());
      } finally {
        decoderWithPastSession = null;
      }
    }

    if (encoderOptions != null) {
      try {
        encoderOptions.close();
        logger.debug("Encoder options closed");
      } catch (IllegalStateException e) {
        logger.debug("Encoder options already closed");
      }
    }

    if (decoderOptions != null) {
      try {
        decoderOptions.close();
        logger.debug("Decoder options closed");
      } catch (IllegalStateException e) {
        logger.debug("Decoder options already closed");
      }
    }
  }
}
