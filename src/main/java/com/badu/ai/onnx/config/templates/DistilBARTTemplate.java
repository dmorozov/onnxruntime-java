package com.badu.ai.onnx.config.templates;

import com.badu.ai.onnx.config.ChatTemplate;

/**
 * DistilBART chat template implementation for summarization and text generation.
 * <p>
 * DistilBART-CNN-12-6 is a distilled version of BART specifically optimized for
 * summarization tasks. It's an encoder-decoder model similar to T5 but specialized
 * for summarization.
 * <p>
 * Format structure:
 * <pre>
 * Direct input:       "[text_to_summarize]"
 * With task prefix:   "summarize: [text]"
 * Q&A format:         "[question]"
 * </pre>
 * <p>
 * Like T5, DistilBART doesn't use chat-style markers. For general Q&A or chat
 * interactions, we pass the user prompt directly.
 *
 * @see <a href="https://huggingface.co/sshleifer/distilbart-cnn-12-6">DistilBART Model Card</a>
 */
public class DistilBARTTemplate implements ChatTemplate {

  /**
   * BART EOS token ID for &lt;/s&gt; (end of sequence).
   * BART uses token ID 2 for the end-of-sequence marker.
   */
  private static final int EOS_TOKEN_ID = 2;

  /**
   * Formats a prompt for DistilBART text generation.
   * <p>
   * DistilBART is primarily designed for summarization, but can handle other
   * text-to-text tasks. For Q&A or chat-like interactions, we format the prompt
   * simply with optional context from the system prompt.
   *
   * @param systemPrompt optional system context (can be null)
   * @param userPrompt the user's question or text to process (required)
   * @return formatted prompt for BART tokenization
   * @throws IllegalArgumentException if userPrompt is null or empty
   */
  @Override
  public String formatPrompt(String systemPrompt, String userPrompt) {
    if (userPrompt == null || userPrompt.trim().isEmpty()) {
      throw new IllegalArgumentException("userPrompt cannot be null or empty");
    }

    // DistilBART formatting is similar to T5:
    // 1. For summarization: can use "summarize: " prefix or direct text
    // 2. For Q&A: direct question
    // 3. With context: system prompt + user prompt

    if (systemPrompt != null && !systemPrompt.trim().isEmpty()) {
      // If system prompt provided, use it as context
      return systemPrompt.trim() + "\n\n" + userPrompt.trim();
    } else {
      // Direct format (works for both Q&A and summarization)
      return userPrompt.trim();
    }
  }

  /**
   * Returns the BART EOS (end-of-sequence) token ID.
   * <p>
   * BART/DistilBART uses token ID 2 for &lt;/s&gt; which signals the model to stop generation.
   *
   * @return the EOS token ID (2 for BART)
   */
  @Override
  public int getEosTokenId() {
    return EOS_TOKEN_ID;
  }

  /**
   * Returns the model type identifier.
   *
   * @return "DistilBART" as the model type
   */
  @Override
  public String getModelType() {
    return "DistilBART";
  }
}
