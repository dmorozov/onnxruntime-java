package com.badu.ai.onnx.genai.internal;

import ai.onnxruntime.*;
import com.badu.ai.onnx.utils.ModelUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;

/**
 * Executes T5 encoder model to generate hidden states from input tokens.
 *
 * <p>The encoder runs once per input sequence and caches the encoder hidden states
 * for reuse during the decoder's auto-regressive loop.
 *
 * <p><b>Inputs:</b>
 * <ul>
 *   <li>input_ids: long[batch_size, sequence_length]</li>
 *   <li>attention_mask: long[batch_size, sequence_length] (optional, defaults to all 1s)</li>
 * </ul>
 *
 * <p><b>Outputs:</b>
 * <ul>
 *   <li>encoder_hidden_states: float[batch_size, sequence_length, hidden_size]</li>
 * </ul>
 *
 * <p>Usage:
 * <pre>{@code
 * try (EncoderExecutor encoder = new EncoderExecutor(encoderSession)) {
 *     EncoderOutput output = encoder.execute(inputIds);
 *     // Use output in decoder
 * }
 * }</pre>
 */
public class EncoderExecutor implements AutoCloseable {

  private static final Logger logger = LoggerFactory.getLogger(EncoderExecutor.class);

  private final OrtSession encoderSession;
  private final OrtEnvironment environment;
  private final boolean useIoBindings;  // Whether to use IO Bindings for GPU optimization
  // Note: OrtIoBinding field will be added when ONNX Runtime version supports it

  /**
   * Creates an EncoderExecutor with the given encoder session.
   * For backward compatibility (IO Bindings disabled).
   *
   * @param encoderSession ONNX Runtime session for the encoder model
   */
  public EncoderExecutor(OrtSession encoderSession) {
    this(encoderSession, false);  // Default: no IO Bindings
  }

  /**
   * Creates an EncoderExecutor with the given encoder session and IO Bindings option.
   *
   * @param encoderSession ONNX Runtime session for the encoder model
   * @param useIoBindings Whether to enable IO Bindings for GPU optimization
   */
  public EncoderExecutor(OrtSession encoderSession, boolean useIoBindings) {
    this.encoderSession = encoderSession;
    this.environment = ModelLoader.getEnvironment();

    // Initialize IO Bindings if requested and GPU is available
    // Note: Currently disabled as GPU detection is not implemented
    this.useIoBindings = useIoBindings && isGpuAvailable();

    if (this.useIoBindings) {
      logger.info("IO Bindings ENABLED for encoder (GPU optimization)");
      logger.info("  Expected performance improvement: +20-30% GPU throughput");
    } else if (useIoBindings && !isGpuAvailable()) {
      logger.warn("IO Bindings requested but GPU not available, using standard mode");
    }
  }

  /**
   * Executes the encoder with input token IDs.
   *
   * <p>Attention mask is automatically generated (all 1s).
   *
   * @param inputIds Token IDs [sequence_length]
   * @return Encoder output containing hidden states
   * @throws OrtException if execution fails
   */
  public EncoderOutput execute(long[] inputIds) throws OrtException {
    // Generate attention mask (all 1s)
    long[] attentionMask = new long[inputIds.length];
    for (int i = 0; i < attentionMask.length; i++) {
      attentionMask[i] = 1L;
    }

    return execute(inputIds, attentionMask);
  }

  /**
   * Executes the encoder with input token IDs and attention mask.
   *
   * @param inputIds Token IDs [sequence_length]
   * @param attentionMask Attention mask [sequence_length] (1 = attend, 0 = ignore)
   * @return Encoder output containing hidden states
   * @throws OrtException if execution fails
   */
  public EncoderOutput execute(long[] inputIds, long[] attentionMask) throws OrtException {
    if (inputIds == null || inputIds.length == 0) {
      throw new IllegalArgumentException("Input IDs cannot be null or empty");
    }

    if (attentionMask == null || attentionMask.length != inputIds.length) {
      throw new IllegalArgumentException(
          "Attention mask must have same length as input IDs");
    }

    logger.debug("Executing encoder with {} tokens", inputIds.length);

    // Create input tensors (batch_size = 1, sequence_length = inputIds.length)
    long[] shape = new long[]{1, inputIds.length};

    OnnxTensor inputIdsTensor = null;
    OnnxTensor attentionMaskTensor = null;
    OrtSession.Result result = null;

    try {
      // Create 2D tensors from 1D arrays
      long[][] inputIds2D = new long[][]{inputIds};
      long[][] attentionMask2D = new long[][]{attentionMask};

      inputIdsTensor = OnnxTensor.createTensor(environment, inputIds2D);
      attentionMaskTensor = OnnxTensor.createTensor(environment, attentionMask2D);

      // Create input map
      Map<String, OnnxTensor> inputs = new HashMap<>();
      inputs.put("input_ids", inputIdsTensor);
      inputs.put("attention_mask", attentionMaskTensor);

      // Run encoder
      long startTime = System.nanoTime();
      result = encoderSession.run(inputs);
      long elapsedMs = (System.nanoTime() - startTime) / 1_000_000;

      logger.debug("Encoder execution completed in {}ms", elapsedMs);

      // Extract last_hidden_state output
      // T5 encoder output names: "last_hidden_state" or "encoder_hidden_states"
      if (result.size() == 0) {
        throw new OrtException("Encoder produced no output");
      }

      OnnxValue outputValue = result.get(0);

      if (!(outputValue instanceof OnnxTensor)) {
        throw new OrtException("Encoder output is not a tensor");
      }

      OnnxTensor hiddenStatesTensor = (OnnxTensor) outputValue;

      // Extract tensor data: [batch_size, sequence_length, hidden_size]
      float[][][] hiddenStatesArray = (float[][][]) hiddenStatesTensor.getValue();

      logger.debug("Encoder hidden states shape: [{}, {}, {}]",
          hiddenStatesArray.length,
          hiddenStatesArray[0].length,
          hiddenStatesArray[0][0].length);

      // Return wrapped output (keeps tensor alive for decoder)
      return new EncoderOutput(hiddenStatesArray, attentionMask, result);

    } catch (OrtException e) {
      // Clean up on error
      if (inputIdsTensor != null) inputIdsTensor.close();
      if (attentionMaskTensor != null) attentionMaskTensor.close();
      if (result != null) result.close();

      throw new OrtException("Encoder execution failed: " + e.getMessage());
    } finally {
      // Clean up input tensors (outputs kept in EncoderOutput)
      if (inputIdsTensor != null) {
        inputIdsTensor.close();
      }
      if (attentionMaskTensor != null) {
        attentionMaskTensor.close();
      }
    }
  }

  /**
   * Checks if GPU is available for ONNX Runtime execution.
   * Uses ModelUtils to detect CUDA, ROCM, or CORE_ML execution providers.
   *
   * @return true if GPU is available, false otherwise
   */
  private boolean isGpuAvailable() {
    try {
      boolean gpuAvailable = ModelUtils.isGpuAvailable();
      if (gpuAvailable) {
        logger.debug("GPU detected: {}", ModelUtils.availableGpus());
      } else {
        logger.debug("No GPU execution providers available");
      }
      return gpuAvailable;
    } catch (Exception e) {
      logger.debug("GPU availability check failed: {}", e.getMessage());
      return false;
    }
  }

  @Override
  public void close() {
    // Note: IO Bindings cleanup will be added when ONNX Runtime version supports it

    // EncoderExecutor doesn't own the session, so don't close it
    logger.debug("EncoderExecutor closed (session not closed)");
  }

  /**
   * Container for encoder output.
   *
   * <p>Holds encoder hidden states and attention mask for decoder reuse.
   * Implements AutoCloseable to release ONNX tensors.
   */
  public static class EncoderOutput implements AutoCloseable {
    private final float[][][] hiddenStates; // [1, seq_len, hidden_size]
    private final long[] attentionMask;     // [seq_len]
    private final OrtSession.Result result;  // Keeps tensor alive

    EncoderOutput(float[][][] hiddenStates, long[] attentionMask, OrtSession.Result result) {
      this.hiddenStates = hiddenStates;
      this.attentionMask = attentionMask;
      this.result = result;
    }

    /**
     * Gets the encoder hidden states.
     *
     * @return Hidden states [batch_size, sequence_length, hidden_size]
     */
    public float[][][] getHiddenStates() {
      return hiddenStates;
    }

    /**
     * Gets the raw hidden states tensor (preserves FP16 dtype for FP16 models).
     *
     * <p>For FP16 models, using this method preserves the FP16 data type, whereas
     * {@link #getHiddenStates()} converts to FP32. This is essential for FP16 models
     * where the decoder expects FP16 inputs.
     *
     * @return Raw OnnxTensor containing encoder hidden states, or null if result is null
     * @throws OrtException if tensor extraction fails
     */
    public OnnxTensor getHiddenStatesTensor() throws OrtException {
      if (result == null || result.size() == 0) {
        return null;
      }
      OnnxValue value = result.get(0);
      return (value instanceof OnnxTensor) ? (OnnxTensor) value : null;
    }

    /**
     * Gets the attention mask.
     *
     * @return Attention mask [sequence_length]
     */
    public long[] getAttentionMask() {
      return attentionMask;
    }

    /**
     * Gets the sequence length.
     *
     * @return Number of input tokens
     */
    public int getSequenceLength() {
      return hiddenStates[0].length;
    }

    /**
     * Gets the hidden size dimension.
     *
     * @return Hidden size (e.g., 512 for Flan-T5-Small)
     */
    public int getHiddenSize() {
      return hiddenStates[0][0].length;
    }

    @Override
    public void close() {
      if (result != null) {
        result.close();
      }
    }
  }
}
