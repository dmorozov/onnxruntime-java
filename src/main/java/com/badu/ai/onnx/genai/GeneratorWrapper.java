package com.badu.ai.onnx.genai;

import java.util.function.Consumer;
import java.util.concurrent.atomic.AtomicInteger;
import com.badu.ai.onnx.TokenCallback;
import com.badu.ai.onnx.config.GenerationConfig;
import com.badu.ai.onnx.genai.internal.GeneratorParams;

/**
 * Wrapper for text generation using SimpleGenAI. Supports both blocking and streaming modes via
 * token callbacks.
 * <p>
 * SimpleGenAI.generate() method handles: - Tokenization (input text → tokens) - Generation
 * (autoregressive token generation) - Decoding (tokens → output text) - Streaming (optional
 * token-by-token callbacks)
 */
public class GeneratorWrapper {

  private final ModelWrapper modelWrapper;
  private final GenerationConfig config;

  private GeneratorWrapper(ModelWrapper modelWrapper, GenerationConfig config) {
    this.modelWrapper = modelWrapper;
    this.config = config;
  }

  /**
   * Creates generator wrapper.
   *
   * @param modelWrapper model wrapper with SimpleGenAI
   * @param config generation configuration
   * @return initialized GeneratorWrapper
   */
  public static GeneratorWrapper create(ModelWrapper modelWrapper, GenerationConfig config) {
    if (modelWrapper == null) {
      throw new IllegalArgumentException("modelWrapper cannot be null");
    }
    if (config == null) {
      throw new IllegalArgumentException("config cannot be null");
    }
    return new GeneratorWrapper(modelWrapper, config);
  }

  /**
   * Generates text in blocking mode (no streaming). Waits for complete generation, then returns
   * full text.
   *
   * @param promptText formatted prompt text (includes chat template)
   * @return complete generated text
   * @throws RuntimeException if generation fails
   */
  public String generateBlocking(String promptText) {
    try {
      // Create generator params
      GeneratorParams params = modelWrapper.createGeneratorParams();

      // Apply generation configuration
      applyConfig(params, config);

      // Call SimpleGenAI.generate() without token listener (blocking mode)
      // The method returns complete generated text
      return modelWrapper.getSimpleGenAI().generate(params, promptText, null);

    } catch (Exception e) {
      throw new RuntimeException("Generation failed (blocking mode): " + e.getMessage(), e);
    }
  }

  /**
   * Generates text in streaming mode. Invokes callback for each generated token, then returns
   * complete text.
   *
   * @param promptText formatted prompt text (includes chat template)
   * @param callback token callback invoked for each generated token
   * @return complete generated text (after streaming completes)
   * @throws RuntimeException if generation fails
   */
  public String generateStreaming(String promptText, TokenCallback callback) {
    if (callback == null) {
      throw new IllegalArgumentException("callback cannot be null for streaming mode");
    }

    try {
      // Create generator params
      GeneratorParams params = modelWrapper.createGeneratorParams();

      // Apply generation configuration
      applyConfig(params, config);

      // Adapt public TokenCallback to Consumer<String> for SimpleGenAI
      // SimpleGenAI only provides token text, so we track position and use placeholder for tokenId
      AtomicInteger position = new AtomicInteger(0);
      Consumer<String> tokenListener = tokenText -> {
        int pos = position.getAndIncrement();
        // SimpleGenAI doesn't provide tokenId or isLast signal, use placeholders
        callback.onToken(-1, tokenText, pos, false);
      };

      // Call SimpleGenAI.generate() with token listener (streaming mode)
      // The method calls listener for each token, then returns complete text
      return modelWrapper.getSimpleGenAI().generate(params, promptText, tokenListener);

    } catch (Exception e) {
      // Notify callback of error
      try {
        callback.onError(e);
      } catch (Exception callbackError) {
        // Ignore callback errors during error handling
      }
      throw new RuntimeException("Generation failed (streaming mode): " + e.getMessage(), e);
    }
  }

  /**
   * Applies generation configuration to GeneratorParams.
   *
   * @param params generator params to configure
   * @param config generation configuration
   * @throws Exception if setting options fails
   */
  private void applyConfig(GeneratorParams params, GenerationConfig config) throws Exception {
    // Set max generation length
    params.setSearchOption("max_length", config.getMaxOutputTokens());

    // Set sampling parameters
    params.setSearchOption("temperature", config.getTemperature());
    params.setSearchOption("top_k", config.getTopK());
    params.setSearchOption("top_p", config.getTopP());
  }
}
