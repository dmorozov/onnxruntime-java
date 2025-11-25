package com.badu.ai.onnx.processing;

import com.badu.ai.onnx.config.ChatTemplate;

/**
 * Utility class for formatting prompts before inference.
 *
 * <p>This class handles prompt formatting with support for:
 * <ul>
 *   <li>Chat templates (model-specific formatting)</li>
 *   <li>Natural language Q&A format (fallback)</li>
 *   <li>System prompts (context injection)</li>
 * </ul>
 *
 * <p>The natural language format is preferred over chat template markers
 * (like &lt;|im_start|&gt;) because SimpleGenAI doesn't tokenize special tokens.
 *
 * <p>Thread Safety: This class is stateless and thread-safe.
 *
 * <p>Usage example:
 * <pre>{@code
 * PromptFormatter formatter = new PromptFormatter();
 * String formatted = formatter.format(systemPrompt, userPrompt, chatTemplate);
 * }</pre>
 */
public class PromptFormatter {

    /**
     * Formats a prompt using chat template or natural language Q&A format.
     *
     * <p>If a chat template is provided, it's used for model-specific formatting.
     * Otherwise, falls back to natural language Q&A format which works well
     * across different models.
     *
     * @param systemPrompt optional system instruction (null for none)
     * @param userPrompt user's question or prompt (required)
     * @param template optional chat template for model-specific formatting (null for default)
     * @return formatted prompt ready for inference
     */
    public String format(String systemPrompt, String userPrompt, ChatTemplate template) {
        if (template != null) {
            return template.formatPrompt(systemPrompt, userPrompt);
        }
        return formatNaturally(systemPrompt, userPrompt);
    }

    /**
     * Formats a prompt using natural language Q&A format.
     *
     * <p>This method creates prompts in a simple Question/Answer format that
     * models can understand without special tokenization:
     *
     * <p>With system prompt:
     * <pre>
     * [system prompt]
     *
     * Question: [user prompt]
     * Answer:
     * </pre>
     *
     * <p>Without system prompt:
     * <pre>
     * Question: [user prompt]
     * Answer:
     * </pre>
     *
     * @param systemPrompt optional system instruction (null or empty for none)
     * @param userPrompt user's question or prompt
     * @return formatted prompt in Q&A format
     */
    public String formatNaturally(String systemPrompt, String userPrompt) {
        if (systemPrompt != null && !systemPrompt.trim().isEmpty()) {
            return systemPrompt.trim() + "\n\nQuestion: " + userPrompt.trim() + "\nAnswer:";
        } else {
            return "Question: " + userPrompt.trim() + "\nAnswer:";
        }
    }

    /**
     * Formats a prompt using chat template only (no fallback).
     *
     * <p>This method requires a chat template and throws if not provided.
     * Use {@link #format(String, String, ChatTemplate)} for automatic fallback.
     *
     * @param systemPrompt optional system instruction (null for none)
     * @param userPrompt user's question or prompt
     * @param template chat template for model-specific formatting (required)
     * @return formatted prompt using chat template
     * @throws IllegalArgumentException if template is null
     */
    public String formatWithTemplate(String systemPrompt, String userPrompt, ChatTemplate template) {
        if (template == null) {
            throw new IllegalArgumentException("Chat template cannot be null");
        }
        return template.formatPrompt(systemPrompt, userPrompt);
    }
}
