package com.badu.ai.onnx.config;

/**
 * Strategy interface for model-specific chat prompt formatting. Different models (Qwen3, LLama 3.2,
 * Phi 3.5) use different chat template formats.
 * <p>
 * Implementations must format prompts with system/user/assistant message structure according to
 * each model's specific template requirements.
 *
 * @see Qwen3Template
 * @see LLama3Template
 * @see Phi3Template
 */
public interface ChatTemplate {

  /**
   * Formats a prompt with system and user messages according to the model's chat template.
   * <p>
   * The formatted prompt includes appropriate role markers and structure for the specific model
   * (e.g., &lt;|im_start|&gt; for Qwen3, &lt;|begin_of_text|&gt; for LLama3).
   *
   * @param systemPrompt the system instruction (nullable, uses default if null)
   * @param userPrompt the user query (non-null, non-empty)
   * @return the formatted prompt ready for tokenization
   * @throws IllegalArgumentException if userPrompt is null or empty
   */
  String formatPrompt(String systemPrompt, String userPrompt);

  /**
   * Returns the model-specific end-of-sequence (EOS) token ID. Used to signal the model to stop
   * generation.
   *
   * @return the EOS token ID (e.g., 151645 for Qwen3, 128009 for LLama3)
   */
  int getEosTokenId();

  /**
   * Returns the model type identifier for debugging and logging.
   *
   * @return the model type (e.g., "Qwen3", "LLama3.2", "Phi3.5")
   */
  String getModelType();
}
