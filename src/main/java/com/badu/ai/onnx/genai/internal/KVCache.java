package com.badu.ai.onnx.genai.internal;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

/**
 * Manages Key-Value cache for decoder models to optimize auto-regressive generation.
 *
 * <p>KV-cache stores the key and value tensors from previous decoder steps,
 * eliminating redundant computation. This significantly speeds up generation:
 * <ul>
 *   <li>First step: Empty cache (compute all)</li>
 *   <li>Subsequent steps: Use cached past_key_values + compute new token only</li>
 * </ul>
 *
 * <p><b>Supported Model Types:</b>
 * <ul>
 *   <li><b>T5/BART (Encoder-Decoder):</b> past_key_values.{layer}.decoder.key, past_key_values.{layer}.encoder.key</li>
 *   <li><b>Llama/Phi-3 (Decoder-Only):</b> past_key_values.{layer}.key, past_key_values.{layer}.value</li>
 * </ul>
 *
 * <p><b>Cache Naming Patterns:</b>
 * <pre>
 * T5 Inputs:  past_key_values.{layer}.decoder.key, past_key_values.{layer}.encoder.key
 * T5 Outputs: present.{layer}.decoder.key, present.{layer}.encoder.key
 *
 * Llama Inputs:  past_key_values.{layer}.key, past_key_values.{layer}.value
 * Llama Outputs: present.{layer}.key, present.{layer}.value
 * </pre>
 *
 * <p><b>Performance Impact:</b>
 * <ul>
 *   <li>Without KV-cache: ~50-100 tokens/sec</li>
 *   <li>With KV-cache: ~200-500 tokens/sec (3-5x speedup)</li>
 * </ul>
 */
public class KVCache implements AutoCloseable {

  private static final Logger logger = LoggerFactory.getLogger(KVCache.class);

  // Cache storage: input_name -> tensor
  private final Map<String, OnnxTensor> cache;

  // Track which tensors we own (for cleanup)
  private final Set<String> ownedTensors;

  // Metadata
  private final int numLayers;
  private boolean initialized;
  private boolean isDecoderOnly; // true for Llama/Phi-3, false for T5/BART

  /**
   * Creates a KVCache for the specified number of decoder layers.
   *
   * @param numLayers Number of decoder layers in the model
   */
  public KVCache(int numLayers) {
    this(numLayers, false); // Default to T5 (encoder-decoder) for backward compatibility
  }

  /**
   * Creates a KVCache for the specified number of decoder layers with model type.
   *
   * @param numLayers Number of decoder layers in the model
   * @param isDecoderOnly true for Llama/Phi-3, false for T5/BART
   */
  public KVCache(int numLayers, boolean isDecoderOnly) {
    this.cache = new LinkedHashMap<>();
    this.ownedTensors = new HashSet<>();
    this.numLayers = numLayers;
    this.initialized = false;
    this.isDecoderOnly = isDecoderOnly;

    logger.debug("KVCache created for {} layers (decoder-only: {})", numLayers, isDecoderOnly);
  }

  /**
   * Initializes the cache with empty tensors for the first decoder step.
   *
   * <p>For decoder_with_past models, the first step requires past_key_values inputs
   * even though there is no history yet. This method creates zero-filled tensors with:
   * <ul>
   *   <li>Decoder KV: [1, num_heads, 0, head_dim] - empty sequence</li>
   *   <li>Encoder KV: [1, num_heads, encoder_seq_len, head_dim] - from encoder output</li>
   * </ul>
   *
   * @param environment ONNX Runtime environment for tensor creation
   * @param encoderSeqLen Sequence length from encoder output
   * @param numLayers Number of decoder layers (typically 8 for T5-small)
   * @param numHeads Number of attention heads (typically 8 for T5-small)
   * @param headDim Dimension per head (typically 64 for T5-small)
   * @throws OrtException if tensor creation fails
   */
  public void initializeEmptyCache(ai.onnxruntime.OrtEnvironment environment,
                                    int encoderSeqLen,
                                    int numLayers,
                                    int numHeads,
                                    int headDim) throws OrtException {
    // Clear existing cache
    clearOwnedTensors();
    cache.clear();

    // Create empty tensors for all 32 inputs (8 layers × 4 tensors)
    for (int layer = 0; layer < numLayers; layer++) {
      // Decoder self-attention KV (empty - no past decoder tokens yet)
      // Shape: [batch_size=1, num_heads, seq_len=0, head_dim]
      float[][][][] emptyDecoderKV = new float[1][numHeads][0][headDim];

      OnnxTensor decoderKey = OnnxTensor.createTensor(environment, emptyDecoderKV);
      OnnxTensor decoderValue = OnnxTensor.createTensor(environment, emptyDecoderKV);

      cache.put("past_key_values." + layer + ".decoder.key", decoderKey);
      cache.put("past_key_values." + layer + ".decoder.value", decoderValue);
      ownedTensors.add("past_key_values." + layer + ".decoder.key");
      ownedTensors.add("past_key_values." + layer + ".decoder.value");

      // Encoder cross-attention KV (empty - will be populated by decoder)
      // Shape: [batch_size=1, num_heads, encoder_seq_len, head_dim]
      float[][][][] emptyEncoderKV = new float[1][numHeads][0][headDim];

      OnnxTensor encoderKey = OnnxTensor.createTensor(environment, emptyEncoderKV);
      OnnxTensor encoderValue = OnnxTensor.createTensor(environment, emptyEncoderKV);

      cache.put("past_key_values." + layer + ".encoder.key", encoderKey);
      cache.put("past_key_values." + layer + ".encoder.value", encoderValue);
      ownedTensors.add("past_key_values." + layer + ".encoder.key");
      ownedTensors.add("past_key_values." + layer + ".encoder.value");
    }

    initialized = true;
    logger.debug("Initialized empty KV-cache with {} tensors ({} layers)", cache.size(), numLayers);
  }

  /**
   * Creates a KVCache by auto-detecting the number of layers from decoder outputs.
   *
   * @param decoderResult First decoder result (to inspect output names)
   * @return Initialized KVCache
   * @throws OrtException if inspection fails
   */
  public static KVCache fromDecoderResult(OrtSession.Result decoderResult) throws OrtException {
    // Count unique layer numbers in output names
    // Example: "present.0.decoder.key" -> layer 0
    Set<Integer> layers = new HashSet<>();

    // Iterate over result entries
    for (Map.Entry<String, OnnxValue> entry : decoderResult) {
      String outputName = entry.getKey();
      if (outputName.startsWith("present.")) {
        try {
          String[] parts = outputName.split("\\.");
          if (parts.length >= 2) {
            int layerNum = Integer.parseInt(parts[1]);
            layers.add(layerNum);
          }
        } catch (NumberFormatException e) {
          // Skip non-layer outputs
        }
      }
    }

    int numLayers = layers.isEmpty() ? 0 : Collections.max(layers) + 1;
    return new KVCache(numLayers);
  }

  /**
   * Initializes the cache with empty tensors for decoder-only models (Llama, Phi-3).
   *
   * <p>Creates empty KV-cache tensors with shape [batch_size, num_heads, 0, head_dim].
   * The sequence length dimension is 0 for the first step (no past).
   *
   * <p><b>Note:</b> ONNX Runtime doesn't support zero-sized tensors in some configurations.
   * Use {@link #initializePlaceholderCacheDecoderOnly(ai.onnxruntime.OrtEnvironment, int, int)}
   * if the model requires KV-cache inputs on first iteration.
   *
   * @param environment ONNX Runtime environment for tensor creation
   * @param numHeads Number of attention heads
   * @param headDim Dimension per head
   * @throws OrtException if tensor creation fails
   */
  public void initializeEmptyCacheDecoderOnly(ai.onnxruntime.OrtEnvironment environment,
                                               int numHeads,
                                               int headDim) throws OrtException {
    if (!isDecoderOnly) {
      throw new IllegalStateException("This method is for decoder-only models. Use initializeEmptyCache() for T5.");
    }

    clearOwnedTensors();
    cache.clear();

    // Create empty tensors for all layers: past_key_values.{layer}.{key|value}
    for (int layer = 0; layer < numLayers; layer++) {
      // Shape: [batch_size=1, num_heads, seq_len=0, head_dim]
      float[][][][] emptyKV = new float[1][numHeads][0][headDim];

      OnnxTensor keyTensor = OnnxTensor.createTensor(environment, emptyKV);
      OnnxTensor valueTensor = OnnxTensor.createTensor(environment, emptyKV);

      cache.put("past_key_values." + layer + ".key", keyTensor);
      cache.put("past_key_values." + layer + ".value", valueTensor);
      ownedTensors.add("past_key_values." + layer + ".key");
      ownedTensors.add("past_key_values." + layer + ".value");
    }

    initialized = false; // Not initialized with real values yet
    logger.debug("Initialized empty KV-cache for decoder-only with {} tensors ({} layers)",
        cache.size(), numLayers);
  }

  /**
   * Initializes the cache with minimal placeholder tensors for decoder-only models (Llama, Phi-3).
   *
   * <p>Creates placeholder KV-cache tensors with shape [batch_size, num_heads, 1, head_dim]
   * filled with zeros. This workaround is needed because ONNX Runtime doesn't support
   * zero-sized tensors, but some models (like Llama 3.2) require KV-cache inputs on every
   * iteration, including the first.
   *
   * <p>The placeholder tensors allow the model to run on first iteration. After the first
   * step, these will be replaced with real KV-cache values from the model's output.
   *
   * @param environment ONNX Runtime environment for tensor creation
   * @param numHeads Number of key-value heads (NOT attention heads - use num_key_value_heads for GQA)
   * @param headDim Dimension per head
   * @throws OrtException if tensor creation fails
   */
  public void initializePlaceholderCacheDecoderOnly(ai.onnxruntime.OrtEnvironment environment,
                                                     int numHeads,
                                                     int headDim) throws OrtException {
    if (!isDecoderOnly) {
      throw new IllegalStateException("This method is for decoder-only models. Use initializeEmptyCache() for T5.");
    }

    clearOwnedTensors();
    cache.clear();

    // Create placeholder tensors for all layers: past_key_values.{layer}.{key|value}
    for (int layer = 0; layer < numLayers; layer++) {
      // Shape: [batch_size=1, num_heads, seq_len=1, head_dim]
      // Filled with zeros as placeholder
      float[][][][] placeholderKV = new float[1][numHeads][1][headDim];
      // Arrays are initialized to zero by default in Java

      OnnxTensor keyTensor = OnnxTensor.createTensor(environment, placeholderKV);
      OnnxTensor valueTensor = OnnxTensor.createTensor(environment, placeholderKV);

      cache.put("past_key_values." + layer + ".key", keyTensor);
      cache.put("past_key_values." + layer + ".value", valueTensor);
      ownedTensors.add("past_key_values." + layer + ".key");
      ownedTensors.add("past_key_values." + layer + ".value");
    }

    initialized = false; // Not initialized with real values yet (just placeholders)
    logger.debug("Initialized placeholder KV-cache for decoder-only with {} tensors ({} layers, seq_len=1)",
        cache.size(), numLayers);
  }

  /**
   * Updates the cache from decoder output.
   * Extracts "present.*" tensors and stores them as "past_key_values.*" for next step.
   *
   * <p><b>For T5/BART:</b> Encoder cross-attention KV is cached ONCE (from first step) and
   * then kept constant. Decoder self-attention KV is updated every step (grows by 1 token).
   *
   * <p><b>For Llama/Phi-3:</b> All KV tensors are updated every step (decoder-only, no encoder).
   *
   * @param decoderResult Result from decoder execution
   * @throws OrtException if tensor extraction fails
   */
  public void updateFromDecoderOutput(OrtSession.Result decoderResult) throws OrtException {
    if (decoderResult == null) {
      throw new IllegalArgumentException("Decoder result cannot be null");
    }

    if (isDecoderOnly) {
      updateFromDecoderOutputDecoderOnly(decoderResult);
    } else {
      updateFromDecoderOutputT5(decoderResult);
    }
  }

  /**
   * Updates cache for decoder-only models (Llama, Phi-3).
   * Converts "present.{layer}.key" to "past_key_values.{layer}.key".
   */
  private void updateFromDecoderOutputDecoderOnly(OrtSession.Result decoderResult) throws OrtException {
    // Clear old cache
    clearOwnedTensors();
    cache.clear();

    int cacheCount = 0;

    for (Map.Entry<String, OnnxValue> entry : decoderResult) {
      String outputName = entry.getKey();
      OnnxValue value = entry.getValue();

      // Check if this is a KV-cache output: "present.{layer}.key" or "present.{layer}.value"
      if (outputName.startsWith("present.") && value instanceof OnnxTensor) {
        // Convert output name to input name
        // "present.0.key" -> "past_key_values.0.key"
        String inputName = outputName.replace("present.", "past_key_values.");

        // Clone the tensor so it survives Result.close()
        OnnxTensor cloned = cloneTensor((OnnxTensor) value);
        if (cloned != null) {
          cache.put(inputName, cloned);
          ownedTensors.add(inputName);
          cacheCount++;
        }
      }
    }

    initialized = (cacheCount > 0);

    if (initialized) {
      logger.debug("KV-cache updated (decoder-only): {} tensors", cacheCount);
    }
  }

  /**
   * Updates cache for T5/BART encoder-decoder models.
   * Handles ".decoder" and ".encoder" suffixes.
   */
  private void updateFromDecoderOutputT5(OrtSession.Result decoderResult) throws OrtException {
    // Clear old DECODER cache only
    // Keep encoder cache constant after first step
    clearDecoderCache();

    // Extract present_* outputs and convert to past_key_values_* inputs
    int decoderCacheCount = 0;
    int encoderCacheCount = 0;
    int encoderCacheReused = 0;

    for (Map.Entry<String, OnnxValue> entry : decoderResult) {
      String outputName = entry.getKey();
      OnnxValue value = entry.getValue();

      // Check if this is a KV-cache output
      if (outputName.startsWith("present.") && value instanceof OnnxTensor) {
        // Convert output name to input name
        // "present.0.decoder.key" -> "past_key_values.0.decoder.key"
        String inputName = outputName.replace("present.", "past_key_values.");

        if (inputName.contains(".decoder.")) {
          // Always update decoder self-attention cache (grows each step)
          // Clone the tensor so it survives Result.close()
          OnnxTensor cloned = cloneTensor((OnnxTensor) value);
          if (cloned != null) {
            // Close old tensor if we own it
            if (cache.containsKey(inputName)) {
              OnnxTensor old = cache.get(inputName);
              if (ownedTensors.contains(inputName)) {
                old.close();
              }
            }
            cache.put(inputName, cloned);
            ownedTensors.add(inputName); // We own the cloned tensor
            decoderCacheCount++;
          }
        } else if (inputName.contains(".encoder.")) {
          // Encoder cross-attention cache: only store on first update, then keep constant
          if (!cache.containsKey(inputName)) {
            // First time: clone encoder KV (computed from encoder_hidden_states)
            OnnxTensor cloned = cloneTensor((OnnxTensor) value);
            if (cloned != null) {
              cache.put(inputName, cloned);
              ownedTensors.add(inputName); // We own the cloned tensor
              encoderCacheCount++;
            }
          } else {
            // Subsequent steps: reuse existing encoder cache (don't update)
            encoderCacheReused++;
          }
        }
      }
    }

    initialized = (decoderCacheCount > 0);

    // Note: We do NOT close the decoder result here because we're holding references
    // to its tensors in the cache. The DecoderExecutor will close the result after
    // extracting logits and updating the cache.

    if (initialized) {
      if (encoderCacheReused > 0) {
        logger.debug("KV-cache updated: {} decoder tensors, {} encoder tensors reused",
            decoderCacheCount, encoderCacheReused);
      } else {
        logger.debug("KV-cache initialized: {} decoder tensors, {} encoder tensors cached",
            decoderCacheCount, encoderCacheCount);
      }
    }
  }

  /**
   * Gets the cache as a map of input tensors for the next decoder step.
   *
   * @return Map of past_key_values input names to tensors
   */
  public Map<String, OnnxTensor> getCacheInputs() {
    return new LinkedHashMap<>(cache);
  }

  /**
   * Checks if the cache is initialized (has cached values).
   *
   * @return true if cache contains values from previous step
   */
  public boolean isInitialized() {
    return initialized;
  }

  /**
   * Gets the number of cached tensors.
   *
   * @return Number of tensors in cache
   */
  public int size() {
    return cache.size();
  }

  /**
   * Checks if the cache is empty.
   *
   * @return true if no cached tensors
   */
  public boolean isEmpty() {
    return cache.isEmpty();
  }

  /**
   * Clears the cache and releases owned tensors.
   */
  public void clear() {
    clearOwnedTensors();
    cache.clear();
    initialized = false;
  }

  /**
   * Closes tensors that we own (not from decoder result).
   */
  private void clearOwnedTensors() {
    for (String tensorName : ownedTensors) {
      OnnxTensor tensor = cache.get(tensorName);
      if (tensor != null) {
        tensor.close();
      }
    }
    ownedTensors.clear();
  }

  /**
   * Clears ONLY decoder self-attention cache, keeping encoder cross-attention cache constant.
   * This is used during cache updates to refresh decoder KV while preserving encoder KV.
   */
  private void clearDecoderCache() {
    // Close and remove decoder cache entries (we own these cloned tensors)
    cache.entrySet().removeIf(entry -> {
      if (entry.getKey().contains(".decoder.")) {
        if (ownedTensors.contains(entry.getKey())) {
          entry.getValue().close();
          ownedTensors.remove(entry.getKey());
        }
        return true; // Remove from cache
      }
      return false; // Keep encoder cache
    });
  }

  /**
   * Clones an ONNX tensor by copying its data to a new tensor.
   * This allows the tensor to survive after the original Result is closed.
   *
   * @param source Source tensor to clone
   * @return Cloned tensor, or null if cloning fails
   */
  private OnnxTensor cloneTensor(OnnxTensor source) {
    try {
      // Get tensor info
      long[] shape = source.getInfo().getShape();

      // Get data and create new tensor based on type
      // KV-cache tensors are typically float32
      Object data = source.getValue();

      // ONNX Runtime requires specific typed arrays, not Object
      // KV-cache tensors are 4D float arrays in T5: [batch_size, num_heads, seq_len, head_dim]
      if (data instanceof float[][][][]) {
        return OnnxTensor.createTensor(ModelLoader.getEnvironment(), (float[][][][]) data);
      } else if (data instanceof float[][][]) {
        return OnnxTensor.createTensor(ModelLoader.getEnvironment(), (float[][][]) data);
      } else if (data instanceof float[][]) {
        return OnnxTensor.createTensor(ModelLoader.getEnvironment(), (float[][]) data);
      } else if (data instanceof float[]) {
        return OnnxTensor.createTensor(ModelLoader.getEnvironment(), (float[]) data);
      } else {
        logger.error("Unsupported tensor type for cloning: {}", data.getClass().getName());
        logger.error("Tensor shape: {}", java.util.Arrays.toString(shape));
        return null;
      }
    } catch (Exception e) {
      logger.error("Failed to clone tensor: {}", e.getMessage());
      return null;
    }
  }

  /**
   * Gets cache statistics for debugging.
   *
   * @return String with cache statistics
   */
  public String getStats() {
    return String.format("KVCache[layers=%d, initialized=%b, size=%d]",
        numLayers, initialized, cache.size());
  }

  @Override
  public void close() {
    // Close all owned tensors (cloned tensors)
    for (String tensorName : ownedTensors) {
      OnnxTensor tensor = cache.get(tensorName);
      if (tensor != null) {
        tensor.close();
      }
    }
    ownedTensors.clear();

    // Clear cache
    clear();
    logger.debug("KVCache closed");
  }

  @Override
  public String toString() {
    return getStats();
  }
}
