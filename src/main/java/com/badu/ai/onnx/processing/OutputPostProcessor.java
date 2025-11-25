package com.badu.ai.onnx.processing;

/**
 * Utility class for post-processing model output to clean up artifacts.
 *
 * <p>This class handles common output cleaning tasks:
 * <ul>
 *   <li>Remove reasoning tags (e.g., &lt;think&gt;...&lt;/think&gt;)</li>
 *   <li>Extract answer after markers (e.g., "Answer:")</li>
 *   <li>Stop at repetition markers (e.g., "\nQuestion:")</li>
 *   <li>Trim and validate output</li>
 * </ul>
 *
 * <p>This is particularly useful for models that:
 * <ul>
 *   <li>Output reasoning steps before answers (e.g., Qwen3)</li>
 *   <li>Echo the prompt in their response</li>
 *   <li>Generate repetitive patterns</li>
 * </ul>
 *
 * <p>Thread Safety: This class is stateless and thread-safe.
 *
 * <p>Usage example:
 * <pre>{@code
 * OutputPostProcessor processor = new OutputPostProcessor();
 * String cleaned = processor.cleanOutput(rawOutput);
 * }</pre>
 */
public class OutputPostProcessor {

    /**
     * Cleans model output by removing artifacts and extracting the answer.
     *
     * <p>Processing steps:
     * <ol>
     *   <li>Remove reasoning tags (&lt;think&gt;...&lt;/think&gt;)</li>
     *   <li>Extract text after "Answer:" marker</li>
     *   <li>Stop at repetition markers ("\nQuestion:", "\nUser:")</li>
     *   <li>Trim and validate (return original if too short)</li>
     * </ol>
     *
     * @param rawOutput raw output from model
     * @return cleaned output text
     */
    public String cleanOutput(String rawOutput) {
        if (rawOutput == null || rawOutput.isEmpty()) {
            return rawOutput;
        }

        String cleaned = rawOutput;

        // Step 1: Remove reasoning tags <think>...</think> if present
        cleaned = removeReasoningTags(cleaned);

        // Step 2: Extract answer after "Answer:" marker if present
        cleaned = extractAnswerAfterMarker(cleaned);

        // Step 3: Stop at repetition markers
        cleaned = stopAtRepetitionMarkers(cleaned);

        // Step 4: Clean up and validate
        cleaned = cleaned.trim();

        // If cleaning resulted in empty or too short output, use original
        if (cleaned.isEmpty() || cleaned.length() < 2) {
            return rawOutput.trim();
        }

        return cleaned;
    }

    /**
     * Removes reasoning tags from output.
     *
     * <p>Some models (like Qwen3) output reasoning in &lt;think&gt; tags.
     * This method removes everything up to and including the closing tag.
     *
     * <p>Example:
     * <pre>
     * Input:  "&lt;think&gt;Let me calculate...&lt;/think&gt;The answer is 4"
     * Output: "The answer is 4"
     * </pre>
     *
     * @param output output text
     * @return text with reasoning tags removed
     */
    private String removeReasoningTags(String output) {
        if (output.contains("</think>")) {
            int thinkEndIndex = output.lastIndexOf("</think>");
            if (thinkEndIndex >= 0) {
                String afterThink = output.substring(thinkEndIndex + 8).trim();
                if (!afterThink.isEmpty()) {
                    return afterThink;
                }
            }
        }
        return output;
    }

    /**
     * Extracts text after "Answer:" marker.
     *
     * <p>Models may echo the prompt including "Answer:" marker.
     * This method extracts only the generated answer text.
     *
     * <p>Example:
     * <pre>
     * Input:  "Question: What is 2+2?\nAnswer: The answer is 4"
     * Output: "The answer is 4"
     * </pre>
     *
     * @param output output text
     * @return text after "Answer:" marker, or original if not found
     */
    private String extractAnswerAfterMarker(String output) {
        // Check for "\nAnswer:" (preferred)
        int answerMarkerIndex = output.lastIndexOf("\nAnswer:");
        if (answerMarkerIndex >= 0) {
            String afterAnswer = output.substring(answerMarkerIndex + 8).trim();
            if (!afterAnswer.isEmpty()) {
                return afterAnswer;
            }
        }

        // Check for "Answer:" at start
        if (output.startsWith("Answer:")) {
            String afterAnswer = output.substring(7).trim();
            if (!afterAnswer.isEmpty()) {
                return afterAnswer;
            }
        }

        return output;
    }

    /**
     * Stops output at repetition markers.
     *
     * <p>Some models repeat the Q&A pattern. This method stops at the first
     * occurrence of repetition markers like "\nQuestion:" or "\nUser:".
     *
     * <p>Example:
     * <pre>
     * Input:  "The answer is 4\n\nQuestion: What is 3+3?"
     * Output: "The answer is 4"
     * </pre>
     *
     * @param output output text
     * @return text before repetition marker, or original if not found
     */
    private String stopAtRepetitionMarkers(String output) {
        int[] stopPoints = {
            output.indexOf("\nQuestion:"),
            output.indexOf("\nUser:"),
            output.indexOf("\n\nQuestion:"),
            output.indexOf("\n\nUser:")
        };

        int earliestStop = Integer.MAX_VALUE;
        for (int stop : stopPoints) {
            if (stop > 0 && stop < earliestStop) {
                earliestStop = stop;
            }
        }

        if (earliestStop < Integer.MAX_VALUE) {
            return output.substring(0, earliestStop).trim();
        }

        return output;
    }

    /**
     * Removes prompt echo from output.
     *
     * <p>This method is useful when the model repeats the original prompt
     * in its response. Not currently used in cleanOutput but available
     * for future use.
     *
     * @param output output text
     * @param originalPrompt original prompt that may be echoed
     * @return output with prompt echo removed
     */
    public String removePromptEcho(String output, String originalPrompt) {
        if (originalPrompt == null || originalPrompt.isEmpty()) {
            return output;
        }

        // Remove exact prompt match at start
        if (output.startsWith(originalPrompt)) {
            return output.substring(originalPrompt.length()).trim();
        }

        // Remove prompt without "Answer:" suffix
        String promptWithoutAnswer = originalPrompt.replace("\nAnswer:", "").trim();
        if (output.startsWith(promptWithoutAnswer)) {
            return output.substring(promptWithoutAnswer.length()).trim();
        }

        return output;
    }
}
