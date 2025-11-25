package com.badu.ai.onnx.config.templates;

import com.badu.ai.onnx.config.ChatTemplate;

/**
 * Qwen3 chat template implementation using ChatML format.
 * <p>
 * Format structure:
 * 
 * <pre>
 * &lt;|im_start|&gt;system
 * {system_prompt}&lt;|im_end|&gt;
 * &lt;|im_start|&gt;user
 * {user_prompt}&lt;|im_end|&gt;
 * &lt;|im_start|&gt;assistant
 * </pre>
 * <p>
 * The model generates text after the final "assistant" marker.
 */
public class Qwen3Template implements ChatTemplate {

  private static final String DEFAULT_SYSTEM = "You are a helpful assistant.";

  /**
   * Qwen3 EOS token ID for &lt;|im_end|&gt;.
   */
  private static final int EOS_TOKEN_ID = 151645;

  @Override
  public String formatPrompt(String systemPrompt, String userPrompt) {
    if (userPrompt == null || userPrompt.trim().isEmpty()) {
      throw new IllegalArgumentException("userPrompt cannot be null or empty");
    }

    String system =
        (systemPrompt != null && !systemPrompt.trim().isEmpty()) ? systemPrompt : DEFAULT_SYSTEM;

    return "<|im_start|>system\n" + system + "<|im_end|>\n" + "<|im_start|>user\n" + userPrompt
        + "<|im_end|>\n" + "<|im_start|>assistant\n";
  }

  @Override
  public int getEosTokenId() {
    return EOS_TOKEN_ID;
  }

  @Override
  public String getModelType() {
    return "Qwen3";
  }
}
