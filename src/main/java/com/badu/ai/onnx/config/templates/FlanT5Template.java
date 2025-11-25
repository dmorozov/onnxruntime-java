package com.badu.ai.onnx.config.templates;

import com.badu.ai.onnx.config.ChatTemplate;

/**
 * Flan-T5 chat template implementation for text-to-text generation.
 * <p>
 * Flan-T5 is an instruction-tuned variant of T5 that accepts task-prefixed inputs.
 * Unlike chat models (LLama3, Qwen3), T5 doesn't use special chat markers.
 * <p>
 * Format structure for different tasks:
 * <pre>
 * Question Answering: "[user_prompt]"
 * Summarization:      "summarize: [text]"
 * Translation:        "translate English to German: [text]"
 * </pre>
 * <p>
 * For general Q&A or chat-like interactions, we simply pass the user prompt directly,
 * optionally prefixed with the system prompt as context.
 *
 * @see <a href="https://huggingface.co/google/flan-t5-small">Flan-T5 Model Card</a>
 */
public class FlanT5Template implements ChatTemplate {

  /**
   * T5 EOS token ID for &lt;/s&gt; (end of sequence).
   */
  private static final int EOS_TOKEN_ID = 1;

  /**
   * Formats a prompt for Flan-T5 text-to-text generation.
   * <p>
   * For Flan-T5, we use a simple format since it's designed for instruction following.
   * If a system prompt is provided, we prepend it as context. For Q&A tasks, we format
   * it as a direct question.
   *
   * @param systemPrompt optional system context (can be null)
   * @param userPrompt the user's question or task (required)
   * @return formatted prompt for T5 tokenization
   * @throws IllegalArgumentException if userPrompt is null or empty
   */
  @Override
  public String formatPrompt(String systemPrompt, String userPrompt) {
    if (userPrompt == null || userPrompt.trim().isEmpty()) {
      throw new IllegalArgumentException("userPrompt cannot be null or empty");
    }

    // For Flan-T5, we can format prompts in different ways depending on the task:
    // 1. Direct question/answer: just the user prompt
    // 2. With context: system prompt + user prompt
    // 3. Task-specific: "summarize: ", "translate: ", etc.

    // For this implementation, we'll support both direct Q&A and context-based prompts
    if (systemPrompt != null && !systemPrompt.trim().isEmpty()) {
      // If system prompt provided, use it as context
      return systemPrompt.trim() + "\n\n" + userPrompt.trim();
    } else {
      // Direct question/answer format
      return userPrompt.trim();
    }
  }

  /**
   * Returns the T5 EOS (end-of-sequence) token ID.
   * <p>
   * T5 uses token ID 1 for &lt;/s&gt; which signals the model to stop generation.
   *
   * @return the EOS token ID (1 for T5)
   */
  @Override
  public int getEosTokenId() {
    return EOS_TOKEN_ID;
  }

  /**
   * Returns the model type identifier.
   *
   * @return "Flan-T5" as the model type
   */
  @Override
  public String getModelType() {
    return "Flan-T5";
  }
}
