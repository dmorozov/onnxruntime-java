package com.badu.ai.onnx.config.templates;

import com.badu.ai.onnx.config.ChatTemplate;

/**
 * LLama 3.2 chat template implementation using header format.
 * <p>
 * Format structure:
 * 
 * <pre>
 * &lt;|begin_of_text|&gt;&lt;|start_header_id|&gt;system&lt;|end_header_id|&gt;
 * {system_prompt}&lt;|eot_id|&gt;
 * &lt;|start_header_id|&gt;user&lt;|end_header_id|&gt;
 * {user_prompt}&lt;|eot_id|&gt;
 * &lt;|start_header_id|&gt;assistant&lt;|end_header_id|&gt;
 * </pre>
 * <p>
 * The model generates text after the final "assistant" header.
 */
public class LLama3Template implements ChatTemplate {

  private static final String DEFAULT_SYSTEM = "You are a helpful assistant.";

  /**
   * LLama 3 EOS token ID for &lt;|eot_id|&gt;.
   */
  private static final int EOS_TOKEN_ID = 128009;

  @Override
  public String formatPrompt(String systemPrompt, String userPrompt) {
    if (userPrompt == null || userPrompt.trim().isEmpty()) {
      throw new IllegalArgumentException("userPrompt cannot be null or empty");
    }

    String system =
        (systemPrompt != null && !systemPrompt.trim().isEmpty()) ? systemPrompt : DEFAULT_SYSTEM;

    return "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n" + system + "<|eot_id|>\n"
        + "<|start_header_id|>user<|end_header_id|>\n" + userPrompt + "<|eot_id|>\n"
        + "<|start_header_id|>assistant<|end_header_id|>\n";
  }

  @Override
  public int getEosTokenId() {
    return EOS_TOKEN_ID;
  }

  @Override
  public String getModelType() {
    return "LLama3.2";
  }
}
