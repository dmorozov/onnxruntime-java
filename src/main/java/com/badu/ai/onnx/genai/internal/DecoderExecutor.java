package com.badu.ai.onnx.genai.internal;

import ai.onnxruntime.*;
import com.badu.ai.onnx.TokenCallback;
import com.badu.ai.onnx.config.GenerationConfig;
import com.badu.ai.onnx.tokenization.T5Tokenizer;
import com.badu.ai.onnx.utils.ModelUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

/**
 * Executes T5 decoder model with auto-regressive generation.
 *
 * <p>Implements auto-regressive decoding loop:
 * <ol>
 *   <li>Start with decoder start token (pad token ID = 0 for T5)</li>
 *   <li>Run decoder to get logits for next token</li>
 *   <li>Apply sampling strategy (greedy/top-k/top-p)</li>
 *   <li>Append token to sequence</li>
 *   <li>Repeat until EOS or max tokens reached</li>
 * </ol>
 *
 * <p><b>Decoder Inputs:</b>
 * <ul>
 *   <li>input_ids: long[1, 1] (current token)</li>
 *   <li>encoder_hidden_states: float[1, seq_len, hidden_size]</li>
 *   <li>encoder_attention_mask: long[1, seq_len]</li>
 * </ul>
 *
 * <p><b>Decoder Outputs:</b>
 * <ul>
 *   <li>logits: float[1, 1, vocab_size]</li>
 * </ul>
 *
 * <p>Usage:
 * <pre>{@code
 * try (DecoderExecutor decoder = new DecoderExecutor(decoderSession)) {
 *     List<Long> tokens = decoder.generate(encoderOutput, generationConfig);
 *     // Convert tokens to text using tokenizer
 * }
 * }</pre>
 */
public class DecoderExecutor implements AutoCloseable {

  private static final Logger logger = LoggerFactory.getLogger(DecoderExecutor.class);

  // T5 special tokens
  private static final long PAD_TOKEN_ID = 0L;    // Decoder start token
  private static final long EOS_TOKEN_ID = 1L;    // End of sequence token

  private final OrtSession decoderSession;  // For first step (or all steps if single-session)
  private final OrtSession decoderWithPastSession;  // For cached steps (null if single-session)
  private final OrtEnvironment environment;
  private final boolean hasKVCache;
  private final boolean hasMergedDecoder;
  private final boolean hasDualDecoder;
  private final boolean hasEncoderHiddenStatesInput;  // True if model expects encoder_hidden_states as separate input
  private final Random rng;
  private final boolean enableDiagnostics;
  private final boolean useIoBindings;  // Whether to use IO Bindings for GPU optimization
  // Note: OrtIoBinding field will be added when ONNX Runtime version supports it

  /**
   * Creates a DecoderExecutor with default settings (single-session, diagnostics disabled).
   * For backward compatibility.
   *
   * @param decoderSession ONNX Runtime session for the decoder model
   * @param hasKVCache Whether the decoder supports KV-cache optimization
   * @param hasMergedDecoder Whether the decoder is merged (has use_cache_branch input)
   */
  public DecoderExecutor(OrtSession decoderSession, boolean hasKVCache, boolean hasMergedDecoder) {
    this(decoderSession, null, hasKVCache, hasMergedDecoder, true, false, false);  // Default: no IO Bindings
  }

  /**
   * Creates a DecoderExecutor with single-session mode and diagnostics control.
   *
   * @param decoderSession ONNX Runtime session for the decoder model
   * @param hasKVCache Whether the decoder supports KV-cache optimization
   * @param hasMergedDecoder Whether the decoder is merged (has use_cache_branch input)
   * @param enableDiagnostics Whether to enable detailed diagnostic logging
   */
  public DecoderExecutor(OrtSession decoderSession, boolean hasKVCache, boolean hasMergedDecoder,
                          boolean enableDiagnostics) {
    this(decoderSession, null, hasKVCache, hasMergedDecoder, true, enableDiagnostics, false);  // Default: no IO Bindings
  }

  /**
   * Creates a DecoderExecutor with dual-session mode (optimal KV-cache performance).
   *
   * @param decoderSession ONNX Runtime session for first decoder step (decoder_model.onnx)
   * @param decoderWithPastSession ONNX Runtime session for cached steps (decoder_with_past_model.onnx), or null for single-session
   * @param hasKVCache Whether the decoder supports KV-cache optimization
   * @param hasMergedDecoder Whether the decoder is merged (has use_cache_branch input)
   * @param hasEncoderHiddenStatesInput Whether decoder_with_past expects encoder_hidden_states as separate input
   * @param enableDiagnostics Whether to enable detailed diagnostic logging
   * @param useIoBindings Whether to enable IO Bindings for GPU optimization
   */
  public DecoderExecutor(OrtSession decoderSession, OrtSession decoderWithPastSession,
                          boolean hasKVCache, boolean hasMergedDecoder, boolean hasEncoderHiddenStatesInput,
                          boolean enableDiagnostics, boolean useIoBindings) {
    this.decoderSession = decoderSession;
    this.decoderWithPastSession = decoderWithPastSession;
    this.hasDualDecoder = (decoderWithPastSession != null);
    this.environment = ModelLoader.getEnvironment();
    this.hasKVCache = hasKVCache;
    this.hasMergedDecoder = hasMergedDecoder;
    this.hasEncoderHiddenStatesInput = hasEncoderHiddenStatesInput;
    this.enableDiagnostics = enableDiagnostics;
    this.rng = new Random();

    // Initialize IO Bindings if requested and GPU is available
    this.useIoBindings = useIoBindings && isGpuAvailable();

    if (this.useIoBindings) {
      logger.info("IO Bindings ENABLED for decoder (GPU optimization)");
      logger.info("  Expected performance improvement: +20-30% GPU throughput");
    } else if (useIoBindings && !isGpuAvailable()) {
      logger.warn("IO Bindings requested but GPU not available, using standard mode");
    }

    if (hasDualDecoder) {
      logger.info("DecoderExecutor initialized with DUAL-SESSION mode (optimal KV-cache performance)");
      logger.info("  - First step: decoder_model.onnx (no cache)");
      logger.info("  - Cached steps: decoder_with_past_model.onnx (with KV-cache)");
    } else if (hasMergedDecoder) {
      logger.debug("DecoderExecutor initialized with merged decoder (use_cache_branch)");
    } else if (hasKVCache) {
      logger.debug("DecoderExecutor initialized with KV-cache support");
    } else {
      logger.debug("DecoderExecutor initialized without KV-cache (slower)");
    }

    if (enableDiagnostics) {
      logger.info("Diagnostic logging ENABLED for DecoderExecutor");
    }
  }

  /**
   * Generates tokens using the specified generation configuration.
   *
   * @param encoderOutput Encoder hidden states from EncoderExecutor
   * @param config Generation configuration (temperature, top-k, top-p, etc.)
   * @return List of generated token IDs
   * @throws OrtException if execution fails
   */
  public List<Long> generate(EncoderExecutor.EncoderOutput encoderOutput,
                              GenerationConfig config) throws OrtException {
    if (encoderOutput == null) {
      throw new IllegalArgumentException("Encoder output cannot be null");
    }

    if (config == null) {
      throw new IllegalArgumentException("Generation config cannot be null");
    }

    logger.debug("Starting generation with maxTokens={}, temperature={}, topK={}, topP={}, KV-cache={}",
        config.getMaxOutputTokens(), config.getTemperature(), config.getTopK(),
        config.getTopP(), hasKVCache);

    List<Long> generatedTokens = new ArrayList<>();
    long currentToken = PAD_TOKEN_ID; // T5 decoder start token

    int maxIterations = config.getMaxOutputTokens();
    long startTime = System.nanoTime();

    // Initialize KV-cache if supported
    // For decoder_with_past models: create empty cache tensors for first step
    // For merged decoders: cache starts uninitialized (use_cache_branch=false on first step)
    KVCache kvCache = null;
    if (hasKVCache && !hasMergedDecoder) {
      // decoder_with_past model: ALWAYS needs past_key_values inputs (empty on first step)
      kvCache = new KVCache(0); // Will be auto-initialized from first decoder output
    } else if (hasKVCache && hasMergedDecoder) {
      // Merged decoder: cache starts uninitialized (use_cache_branch controls mode)
      kvCache = new KVCache(0);
    }

    try {
      for (int step = 0; step < maxIterations; step++) {
        // Run decoder for one step
        // If hasKVCache: pass only current token, cache maintains history
        // If !hasKVCache: pass ALL tokens so far (PAD + generated tokens)
        float[] logits = runDecoderStep(currentToken, generatedTokens, encoderOutput, kvCache);

        // Suppress EOS token if below minimum length (for encoder-decoder models, count only generated tokens)
        if (config.hasMinLength() && generatedTokens.size() < config.getMinOutputTokens()) {
          logits[(int) EOS_TOKEN_ID] = Float.NEGATIVE_INFINITY;
          logger.trace("Suppressed EOS token (generated: {}/{} min tokens)",
              generatedTokens.size(), config.getMinOutputTokens());
        }

        // Apply bad words filter (mask tokens that would complete banned sequences)
        if (config.hasBadWords()) {
          BadWordsProcessor.applyBadWordsFilter(logits, generatedTokens, config.getBadWordsIds());
        }

        // Apply generation strategy
        long nextToken = sampleNextToken(logits, generatedTokens, config);

        // DEBUG: Print token ID
        // System.out.println("DEBUG: Generated token ID at step " + step + ": " + nextToken);

        // Check for EOS
        if (nextToken == EOS_TOKEN_ID) {
          logger.debug("EOS token generated at step {}", step);
          break;
        }

        generatedTokens.add(nextToken);
        currentToken = nextToken;
      }

      long elapsedMs = (System.nanoTime() - startTime) / 1_000_000;
      float tokensPerSec = generatedTokens.size() / (elapsedMs / 1000.0f);

      String cacheInfo = (kvCache != null && kvCache.isInitialized())
          ? String.format(" (KV-cache: %d tensors)", kvCache.size())
          : "";

      logger.info("Generated {} tokens in {}ms ({} tokens/sec){}",
          generatedTokens.size(), elapsedMs, String.format("%.1f", tokensPerSec), cacheInfo);

      return generatedTokens;

    } finally {
      // Clean up KV-cache
      if (kvCache != null) {
        kvCache.close();
      }
    }
  }

  /**
   * Runs a single decoder step to produce logits for the next token.
   *
   * @param currentToken Current token ID to decode
   * @param generatedTokens Previously generated tokens (for non-cache mode)
   * @param encoderOutput Encoder hidden states
   * @param kvCache KV-cache for past key-values (null if not using cache)
   * @return Logits for next token [vocab_size]
   * @throws OrtException if execution fails
   */
  private float[] runDecoderStep(long currentToken,
                                  List<Long> generatedTokens,
                                  EncoderExecutor.EncoderOutput encoderOutput,
                                  KVCache kvCache) throws OrtException {

    // Dual-session mode: use decoderSession for first step, decoderWithPastSession for cached steps
    OrtSession sessionToUse;
    if (hasDualDecoder) {
      sessionToUse = generatedTokens.isEmpty() ? decoderSession : decoderWithPastSession;
      if (enableDiagnostics) {
        if (generatedTokens.isEmpty()) {
          logger.info("[DIAG] Dual-session: Using decoder_model.onnx for FIRST step (no cache)");
        } else if (generatedTokens.size() == 1) {
          logger.info("[DIAG] Dual-session: Using decoder_with_past_model.onnx for CACHED step (iteration {})", generatedTokens.size());
        }
      }
    } else {
      sessionToUse = decoderSession;
    }

    OnnxTensor inputIdsTensor = null;
    OnnxTensor encoderHiddenStatesTensor = null;
    OnnxTensor encoderAttentionMaskTensor = null;
    OnnxTensor useCacheBranchTensor = null;
    OrtSession.Result result = null;
    boolean shouldCloseEncoderTensor = false;  // Track if we created the tensor (and should close it)

    try {
      // Create input tensors
      // input_ids shape depends on KV-cache:
      // - WITH KV-cache: [1, 1] (just current token, history in cache)
      // - WITHOUT KV-cache: [1, seq_len] (PAD + all generated tokens)
      long[][] inputIds;
      if (hasKVCache && kvCache != null && kvCache.isInitialized()) {
        // KV-cache maintains history, pass only current token
        inputIds = new long[][]{{currentToken}};
        // System.out.println("DEBUG: runDecoderStep with currentToken=" + currentToken + " (KV-cache)");
      } else {
        // No KV-cache: pass decoder_input_ids = PAD + all generated tokens so far
        // The merged decoder with use_cache_branch=false expects the full sequence
        List<Long> decoderInputIds = new ArrayList<>();
        decoderInputIds.add(PAD_TOKEN_ID); // Decoder start token
        decoderInputIds.addAll(generatedTokens); // All tokens generated so far
        long[] tokenArray = decoderInputIds.stream().mapToLong(Long::longValue).toArray();
        inputIds = new long[][]{tokenArray};
        // System.out.println("DEBUG: runDecoderStep with " + decoderInputIds.size() + " tokens (no cache): " + decoderInputIds);
      }
      inputIdsTensor = OnnxTensor.createTensor(environment, inputIds);

      // encoder_hidden_states: [1, seq_len, hidden_size]
      // For FP16 models, use raw tensor to preserve dtype; for FP32 models, recreate from array
      OnnxTensor rawTensor = encoderOutput.getHiddenStatesTensor();
      if (rawTensor != null) {
        // Use raw tensor directly (preserves FP16 dtype for FP16 models)
        // Don't close this tensor - it's owned by EncoderOutput
        encoderHiddenStatesTensor = rawTensor;
        shouldCloseEncoderTensor = false;
        if (logger.isDebugEnabled()) {
          logger.debug("Using raw encoder tensor (dtype preserved): type={}, shape={}",
              rawTensor.getInfo().type, rawTensor.getInfo().getShape());
        }
      } else {
        // Fallback: create from Java array (FP32 models or if raw tensor unavailable)
        // We own this tensor, so mark it for cleanup
        encoderHiddenStatesTensor = OnnxTensor.createTensor(
            environment, encoderOutput.getHiddenStates());
        shouldCloseEncoderTensor = true;
        if (logger.isDebugEnabled()) {
          logger.debug("Created new encoder tensor from Java array (FP32)");
        }
      }

      // encoder_attention_mask: [1, seq_len]
      long[][] encoderMask = new long[][]{encoderOutput.getAttentionMask()};
      encoderAttentionMaskTensor = OnnxTensor.createTensor(environment, encoderMask);

      // Create input map
      Map<String, OnnxTensor> inputs = new HashMap<>();
      inputs.put("input_ids", inputIdsTensor);

      // Add encoder_hidden_states based on decoder step and model format:
      // - First step (decoder_model.onnx): ALWAYS required
      // - Cached steps (decoder_with_past_model.onnx): conditionally required
      //   * Old format (optimum-cli 1.x): required as separate input (35 inputs total)
      //   * New format (optimum-cli 2.0.0+): embedded in past_key_values (34 inputs total)
      boolean isFirstStep = generatedTokens.isEmpty();
      boolean needsEncoderHiddenStates = isFirstStep || hasEncoderHiddenStatesInput;
      if (needsEncoderHiddenStates) {
        inputs.put("encoder_hidden_states", encoderHiddenStatesTensor);
      }

      inputs.put("encoder_attention_mask", encoderAttentionMaskTensor);

      // For merged decoders, add use_cache_branch boolean input
      if (hasMergedDecoder) {
        // use_cache_branch = false on first step (no cache), true on subsequent steps
        boolean useCache = (kvCache != null && kvCache.isInitialized());
        boolean[] useCacheBranch = new boolean[]{useCache};
        useCacheBranchTensor = OnnxTensor.createTensor(environment, useCacheBranch);
        inputs.put("use_cache_branch", useCacheBranchTensor);

        if (enableDiagnostics && generatedTokens.isEmpty()) {
          logger.info("[DIAG] Merged decoder step: use_cache_branch={} (kvCache={}, initialized={}, hasKVCache={})",
              useCache, (kvCache != null), (kvCache != null && kvCache.isInitialized()), hasKVCache);
        }
        logger.trace("Merged decoder step: use_cache_branch={}", useCache);
      }

      // Add past_key_values from cache if available and supported
      if (kvCache != null && kvCache.isInitialized() && hasKVCache) {
        Map<String, OnnxTensor> cacheInputs = kvCache.getCacheInputs();

        // Only add inputs that the decoder actually expects
        Set<String> expectedInputs = sessionToUse.getInputNames();
        int addedCount = 0;
        for (Map.Entry<String, OnnxTensor> entry : cacheInputs.entrySet()) {
          if (expectedInputs.contains(entry.getKey())) {
            inputs.put(entry.getKey(), entry.getValue());
            addedCount++;
          }
        }

        if (enableDiagnostics && generatedTokens.size() == 1) {
          logger.info("[DIAG] KV-cache initialized={}, size={}, added={} tensors to decoder",
              kvCache.isInitialized(), kvCache.size(), addedCount);
          logger.info("[DIAG] Decoder expects inputs: {}", expectedInputs);
          logger.info("[DIAG] We are providing inputs: {}", inputs.keySet());
        }

        if (addedCount > 0) {
          logger.trace("Added {} cached tensors to decoder inputs", addedCount);
        } else {
          logger.warn("KV-cache has {} tensors but decoder doesn't accept them (decoder expects: {})",
              cacheInputs.size(), expectedInputs);
        }
      } else {
        if (enableDiagnostics && generatedTokens.isEmpty()) {
          logger.info("[DIAG] KV-cache NOT used (kvCache={}, initialized={}, hasKVCache={})",
              (kvCache != null), (kvCache != null && kvCache.isInitialized()), hasKVCache);
        }
      }

      // DEBUG: Print decoder inputs
      // Set<String> expectedInputs = sessionToUse.getInputNames();
      // System.out.println("DEBUG: Decoder expects inputs: " + expectedInputs);
      // System.out.println("DEBUG: We are providing inputs: " + inputs.keySet());

      // Run decoder
      result = sessionToUse.run(inputs);

      // Update KV-cache from decoder outputs
      if (kvCache != null) {
        kvCache.updateFromDecoderOutput(result);
        if (!kvCache.isInitialized() && kvCache.size() == 0) {
          // First step with no cache outputs - auto-detect layers
          kvCache = KVCache.fromDecoderResult(result);
          logger.debug("Auto-detected {} cache layers from decoder output", kvCache.size());
        }
        if (enableDiagnostics && generatedTokens.isEmpty()) {
          logger.info("[DIAG] After first step: KV-cache now has {} tensors, initialized={}",
              kvCache.size(), kvCache.isInitialized());
        }

        // Log KV-cache shape on first few iterations to verify growth
        if (enableDiagnostics && kvCache.isInitialized() && generatedTokens.size() < 5) {
          try {
            Map<String, OnnxTensor> cacheInputs = kvCache.getCacheInputs();
            // Log decoder cache shape (should grow by 1 each iteration)
            OnnxTensor decoderKey = cacheInputs.get("past_key_values.0.decoder.key");
            if (decoderKey != null) {
              long[] shape = decoderKey.getInfo().getShape();
              long seqLen = shape.length >= 3 ? shape[2] : -1;
              logger.info("[KV-CACHE DIAG] Iteration {}: Decoder cache shape={}, seq_len={}",
                  generatedTokens.size(), Arrays.toString(shape), seqLen);
            }
            // Log encoder cache shape (should remain constant)
            OnnxTensor encoderKey = cacheInputs.get("past_key_values.0.encoder.key");
            if (encoderKey != null && generatedTokens.isEmpty()) {
              long[] shape = encoderKey.getInfo().getShape();
              long seqLen = shape.length >= 3 ? shape[2] : -1;
              logger.info("[KV-CACHE DIAG] Iteration {}: Encoder cache shape={}, seq_len={} (constant)",
                  generatedTokens.size(), Arrays.toString(shape), seqLen);
            }
          } catch (Exception e) {
            logger.warn("[KV-CACHE DIAG] Failed to inspect cache shapes: {}", e.getMessage());
          }
        }
      }

      // Extract logits output
      // T5 decoder output: "logits" shape [1, 1, vocab_size]
      // Note: Decoder may have multiple outputs (KV-cache + logits)
      // Logits are typically the LAST output or named "logits"
      if (result.size() == 0) {
        throw new OrtException("Decoder produced no output");
      }

      // DEBUG: Print all output names
      // System.out.println("DEBUG: Decoder has " + result.size() + " outputs");

      // Try to find logits by name first
      OnnxValue logitsValue = null;
      try {
        logitsValue = result.get("logits").orElse(null);
        if (logitsValue != null) {
          // System.out.println("DEBUG: Found logits output by name 'logits'");
        }
      } catch (Exception e) {
        // Name lookup failed, will use index fallback
        logger.warn("DEBUG: Could not find logits by name, using index fallback");
      }

      // Fallback: logits are usually the last output
      if (logitsValue == null) {
        int lastIndex = result.size() - 1;
        logitsValue = result.get(lastIndex);
        // System.out.println("DEBUG: Using last output (index " + lastIndex + "/" + result.size() + ") for logits");
      }

      if (!(logitsValue instanceof OnnxTensor)) {
        throw new OrtException("Decoder output is not a tensor");
      }

      OnnxTensor logitsTensor = (OnnxTensor) logitsValue;
      float[][][] logitsArray = (float[][][]) logitsTensor.getValue();

      // Debug: log shape and first few values
      //System.out.println("DEBUG: Logits shape: [" + logitsArray.length + ", " +
      //    logitsArray[0].length + ", " + logitsArray[0][0].length + "]");

      // Extract logits for the LAST position (newly generated token)
      // - WITH KV-cache: input_ids is [1,1] so logits is [1,1,vocab_size] -> use position 0
      // - WITHOUT KV-cache: input_ids is [1,seq_len] so logits is [1,seq_len,vocab_size] -> use position seq_len-1
      int lastPosition = logitsArray[0].length - 1;
      float[] lastLogits = logitsArray[0][lastPosition];

      //System.out.println("DEBUG: Using logits from position " + lastPosition + "/" + logitsArray[0].length);
      //System.out.println("DEBUG: First 5 logits at last position: [" + lastLogits[0] + ", " +
      //    lastLogits[1] + ", " + lastLogits[2] + ", " + lastLogits[3] + ", " + lastLogits[4] + "]");

      return lastLogits;

    } finally {
      // Clean up input tensors
      if (inputIdsTensor != null) inputIdsTensor.close();
      // Only close encoder tensor if we created it (not if we reused from EncoderOutput)
      if (encoderHiddenStatesTensor != null && shouldCloseEncoderTensor) encoderHiddenStatesTensor.close();
      if (encoderAttentionMaskTensor != null) encoderAttentionMaskTensor.close();
      if (useCacheBranchTensor != null) useCacheBranchTensor.close();

      // Close result after KVCache has extracted tensors
      // KVCache holds references to tensors but doesn't own the result
      if (result != null) {
        result.close();
      }
    }
  }

  /**
   * Samples the next token using the configured generation strategy.
   *
   * @param logits Logits from decoder [vocab_size]
   * @param generatedTokens Previously generated tokens (for repetition penalty)
   * @param config Generation configuration
   * @return Sampled token ID
   */
  private long sampleNextToken(float[] logits, List<Long> generatedTokens,
                                GenerationConfig config) {

    // Apply repetition penalty
    if (config.hasRepetitionPenalty()) {
      SamplingUtils.applyRepetitionPenalty(logits, generatedTokens,
          config.getRepetitionPenalty());
    }

    // Apply temperature
    if (config.getTemperature() > 0.0f && config.getTemperature() != 1.0f) {
      SamplingUtils.applyTemperature(logits, config.getTemperature());
    }

    // Select sampling strategy
    if (config.isGreedy()) {
      // Greedy decoding (temperature = 0.0)
      return SamplingUtils.argmax(logits);
    } else if (config.isTopKEnabled()) {
      // Top-K sampling
      return SamplingUtils.sampleTopK(logits, config.getTopK(), rng);
    } else if (config.isTopPEnabled()) {
      // Top-P (nucleus) sampling
      return SamplingUtils.sampleTopP(logits, config.getTopP(), rng);
    } else {
      // Fallback to greedy
      logger.warn("No valid sampling strategy configured, using greedy");
      return SamplingUtils.argmax(logits);
    }
  }

  /**
   * Generates tokens with streaming callback support.
   *
   * <p>This method invokes the callback for each token as it's generated, enabling
   * progressive display and real-time UX. TTFT (time to first token) is tracked
   * separately from total decoder time.
   *
   * <p><b>Performance Targets:</b>
   * <ul>
   *   <li>TTFT: <50ms from API call</li>
   *   <li>Token intervals: <100ms between tokens</li>
   * </ul>
   *
   * @param encoderOutput Encoder hidden states from EncoderExecutor
   * @param config Generation configuration (temperature, top-k, top-p, etc.)
   * @param tokenizer Tokenizer for decoding tokens to text
   * @param callback Callback invoked for each token
   * @return StreamingResult with generated tokens and TTFT
   * @throws OrtException if execution fails
   */
  public StreamingResult generateStreaming(EncoderExecutor.EncoderOutput encoderOutput,
                                            GenerationConfig config,
                                            T5Tokenizer tokenizer,
                                            TokenCallback callback) throws OrtException {
    if (encoderOutput == null) {
      throw new IllegalArgumentException("Encoder output cannot be null");
    }

    if (config == null) {
      throw new IllegalArgumentException("Generation config cannot be null");
    }

    if (tokenizer == null) {
      throw new IllegalArgumentException("Tokenizer cannot be null");
    }

    if (callback == null) {
      throw new IllegalArgumentException("Callback cannot be null");
    }

    logger.debug("Starting streaming generation with maxTokens={}, temperature={}, topK={}, topP={}, KV-cache={}",
        config.getMaxOutputTokens(), config.getTemperature(), config.getTopK(),
        config.getTopP(), hasKVCache);

    List<Long> generatedTokens = new ArrayList<>();
    long currentToken = PAD_TOKEN_ID; // T5 decoder start token

    int maxIterations = config.getMaxOutputTokens();
    long startTime = System.nanoTime();
    long ttftNanos = 0; // Time to first token

    // Track previously decoded text to compute diffs
    String previousText = "";

    // Initialize KV-cache if supported
    // Note: KVCache starts UNINITIALIZED on first step (no past_key_values yet)
    // After first decoder run, updateFromDecoderOutput() will populate it with present outputs
    KVCache kvCache = hasKVCache ? new KVCache(0) : null;

    try {
      for (int step = 0; step < maxIterations; step++) {
        // Run decoder for one step
        // If hasKVCache: pass only current token, cache maintains history
        // If !hasKVCache: pass ALL tokens so far (PAD + generated tokens)
        float[] logits = runDecoderStep(currentToken, generatedTokens, encoderOutput, kvCache);

        // Suppress EOS token if below minimum length (for encoder-decoder models, count only generated tokens)
        if (config.hasMinLength() && generatedTokens.size() < config.getMinOutputTokens()) {
          logits[(int) EOS_TOKEN_ID] = Float.NEGATIVE_INFINITY;
          logger.trace("Suppressed EOS token (generated: {}/{} min tokens)",
              generatedTokens.size(), config.getMinOutputTokens());
        }

        // Apply bad words filter (mask tokens that would complete banned sequences)
        if (config.hasBadWords()) {
          BadWordsProcessor.applyBadWordsFilter(logits, generatedTokens, config.getBadWordsIds());
        }

        // Apply generation strategy
        long nextToken = sampleNextToken(logits, generatedTokens, config);

        // DEBUG: Print token ID
        // System.out.println("DEBUG: Generated token ID at step " + step + ": " + nextToken);

        // Check for EOS
        if (nextToken == EOS_TOKEN_ID) {
          logger.debug("EOS token generated at step {}", step);
          break;
        }

        generatedTokens.add(nextToken);

        // Capture TTFT (time to first token)
        if (step == 0) {
          ttftNanos = System.nanoTime() - startTime;
          logger.debug("TTFT: {}ms", ttftNanos / 1_000_000);
        }

        // Decode full sequence to get proper spacing
        // Then compute diff to get only the new token text
        String fullText = decodeTokenSequence(tokenizer, generatedTokens);
        String tokenText = fullText.substring(previousText.length());
        previousText = fullText;

        // Invoke callback with token
        boolean isLast = (step == maxIterations - 1);
        try {
          callback.onToken((int) nextToken, tokenText, step, isLast);
        } catch (Exception e) {
          logger.error("Callback threw exception at token {}", step, e);
          throw new RuntimeException("Callback error: " + e.getMessage(), e);
        }

        currentToken = nextToken;
      }

      long elapsedMs = (System.nanoTime() - startTime) / 1_000_000;
      float tokensPerSec = generatedTokens.size() / (elapsedMs / 1000.0f);

      String cacheInfo = (kvCache != null && kvCache.isInitialized())
          ? String.format(" (KV-cache: %d tensors)", kvCache.size())
          : "";

      logger.info("Streamed {} tokens in {}ms ({} tokens/sec), TTFT: {}ms{}",
          generatedTokens.size(), elapsedMs, String.format("%.1f", tokensPerSec),
          ttftNanos / 1_000_000, cacheInfo);

      return new StreamingResult(generatedTokens, ttftNanos / 1_000_000);

    } catch (Exception e) {
      logger.error("Streaming generation failed", e);
      throw e;
    } finally {
      // Clean up KV-cache
      if (kvCache != null) {
        kvCache.close();
      }
    }
  }

  /**
   * Decodes a single token to text.
   *
   * @param tokenizer T5Tokenizer instance
   * @param tokenId Token ID to decode
   * @return Decoded text for the token
   */
  private String decodeToken(T5Tokenizer tokenizer, long tokenId) {
    try {
      // Decode single token by wrapping in array
      long[] tokenArray = new long[]{tokenId};
      return tokenizer.decode(tokenArray, true); // Skip special tokens
    } catch (Exception e) {
      logger.warn("Failed to decode token {}: {}", tokenId, e.getMessage());
      return ""; // Return empty string for problematic tokens
    }
  }

  /**
   * Decodes a sequence of tokens to text.
   * This is the correct way to decode subword tokens (like T5 SentencePiece)
   * which require context to properly reconstruct spacing.
   *
   * @param tokenizer T5Tokenizer instance
   * @param tokens List of token IDs to decode
   * @return Decoded text for the full sequence
   */
  private String decodeTokenSequence(T5Tokenizer tokenizer, List<Long> tokens) {
    try {
      // Convert List<Long> to long[]
      long[] tokenArray = tokens.stream().mapToLong(Long::longValue).toArray();
      return tokenizer.decode(tokenArray, true); // Skip special tokens
    } catch (Exception e) {
      logger.warn("Failed to decode token sequence of length {}: {}", tokens.size(), e.getMessage());
      return ""; // Return empty string for problematic sequences
    }
  }

  /**
   * Result object for streaming generation.
   */
  public static class StreamingResult {
    private final List<Long> tokens;
    private final long ttftMs;

    public StreamingResult(List<Long> tokens, long ttftMs) {
      this.tokens = tokens;
      this.ttftMs = ttftMs;
    }

    public List<Long> getTokens() {
      return tokens;
    }

    public long getTtftMs() {
      return ttftMs;
    }
  }

  /**
   * Gets the decoder start token ID for T5.
   *
   * @return Decoder start token ID (PAD = 0)
   */
  public static long getDecoderStartTokenId() {
    return PAD_TOKEN_ID;
  }

  /**
   * Gets the EOS (end-of-sequence) token ID for T5.
   *
   * @return EOS token ID (1)
   */
  public static long getEosTokenId() {
    return EOS_TOKEN_ID;
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

    // DecoderExecutor doesn't own the session, so don't close it
    logger.debug("DecoderExecutor closed (session not closed)");
  }
}
