package com.badu.ai.onnx.config.templates;

import com.badu.ai.onnx.config.ChatTemplate;

/**
 * Phi 3.5 chat template implementation using role markers.
 * <p>
 * Format structure:
 * <pre>
 * &lt;|system|&gt;
 * {system_prompt}&lt;|end|&gt;
 * &lt;|user|&gt;
 * {user_prompt}&lt;|end|&gt;
 * &lt;|assistant|&gt;
 * </pre>
 * <p>
 * The model generates text after the final "assistant" marker.
 */
public class Phi3Template implements ChatTemplate {

    private static final String DEFAULT_SYSTEM = "You are a helpful assistant.";

    /**
     * Phi 3 EOS token ID for &lt;|end|&gt;.
     */
    private static final int EOS_TOKEN_ID = 32007;

    @Override
    public String formatPrompt(String systemPrompt, String userPrompt) {
        if (userPrompt == null || userPrompt.trim().isEmpty()) {
            throw new IllegalArgumentException("userPrompt cannot be null or empty");
        }

        String system = (systemPrompt != null && !systemPrompt.trim().isEmpty())
            ? systemPrompt
            : DEFAULT_SYSTEM;

        return "<|system|>\n" + system + "<|end|>\n" +
               "<|user|>\n" + userPrompt + "<|end|>\n" +
               "<|assistant|>\n";
    }

    @Override
    public int getEosTokenId() {
        return EOS_TOKEN_ID;
    }

    @Override
    public String getModelType() {
        return "Phi3.5";
    }
}
