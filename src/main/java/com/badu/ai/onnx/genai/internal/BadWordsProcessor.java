package com.badu.ai.onnx.genai.internal;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

/**
 * Utility class for filtering bad words during text generation.
 *
 * <p>Implements prefix-matching algorithm to prevent generation of banned token sequences.
 * Supports both single-token and multi-token bad words.
 *
 * <p><b>Algorithm:</b>
 * For each bad word sequence, check if the current generated tokens match the prefix:
 * <ol>
 *   <li>If generated tokens end with bad_word[:-1], mask bad_word[-1]</li>
 *   <li>Repeat for all bad word sequences</li>
 * </ol>
 *
 * <p><b>Example:</b>
 * <pre>{@code
 * // Bad word: "feck" → tokenizes to [12345, 67890]
 * // Generated tokens: [..., 12345]
 * // → Mask token 67890 to prevent "feck"
 *
 * List<Long> generatedTokens = List.of(100L, 200L, 12345L);
 * List<List<Long>> badWords = List.of(List.of(12345L, 67890L));
 * float[] logits = new float[32000];
 *
 * BadWordsProcessor.applyBadWordsFilter(logits, generatedTokens, badWords);
 * // → logits[67890] = Float.NEGATIVE_INFINITY
 * }</pre>
 *
 * <p><b>Performance Considerations:</b>
 * <ul>
 *   <li>Time complexity: O(B * M) where B = number of bad words, M = max bad word length</li>
 *   <li>Negligible overhead for small bad word lists (< 100 sequences)</li>
 *   <li>For large lists (> 1000 sequences), consider using Trie data structure</li>
 * </ul>
 *
 * @see com.badu.ai.onnx.config.GenerationConfig#getBadWordsIds()
 */
public class BadWordsProcessor {

    private static final Logger logger = LoggerFactory.getLogger(BadWordsProcessor.class);

    /**
     * Private constructor to prevent instantiation (utility class).
     */
    private BadWordsProcessor() {
        throw new UnsupportedOperationException("Utility class");
    }

    /**
     * Applies bad words filter to logits by masking tokens that would complete banned sequences.
     *
     * <p>Modifies logits in-place by setting probabilities to -infinity for tokens that
     * would complete a bad word sequence based on currently generated tokens.
     *
     * <p><b>Multi-Token Example:</b>
     * <pre>{@code
     * Bad word: [123, 456, 789] (3 tokens)
     * Generated: [..., 123, 456]
     * → Mask token 789 (would complete bad word)
     *
     * Bad word: [123, 456, 789]
     * Generated: [..., 123]
     * → No masking (prefix not complete yet)
     * }</pre>
     *
     * @param logits Logits array to modify (vocab_size)
     * @param generatedTokens Currently generated token IDs
     * @param badWordsIds List of bad word token sequences to filter
     */
    public static void applyBadWordsFilter(float[] logits, List<Long> generatedTokens,
                                            List<List<Long>> badWordsIds) {
        if (badWordsIds == null || badWordsIds.isEmpty()) {
            return;  // No filtering needed
        }

        if (generatedTokens == null || generatedTokens.isEmpty()) {
            return;  // No tokens generated yet, nothing to filter
        }

        int maskedCount = 0;

        // For each bad word sequence
        for (List<Long> badWordSeq : badWordsIds) {
            if (badWordSeq.size() == 1) {
                // Single-token bad word: Always mask it
                int tokenId = badWordSeq.get(0).intValue();
                if (tokenId >= 0 && tokenId < logits.length) {
                    logits[tokenId] = Float.NEGATIVE_INFINITY;
                    maskedCount++;
                }
            } else {
                // Multi-token bad word: Check if prefix matches
                int prefixLen = badWordSeq.size() - 1;
                int genLen = generatedTokens.size();

                // Check if we have enough generated tokens to match prefix
                if (genLen >= prefixLen) {
                    // Extract last N tokens from generated sequence
                    List<Long> suffix = generatedTokens.subList(genLen - prefixLen, genLen);

                    // Check if suffix matches bad word prefix (all tokens except last)
                    boolean prefixMatches = true;
                    for (int i = 0; i < prefixLen; i++) {
                        if (!suffix.get(i).equals(badWordSeq.get(i))) {
                            prefixMatches = false;
                            break;
                        }
                    }

                    // If prefix matches, mask the completion token
                    if (prefixMatches) {
                        int tokenToMask = badWordSeq.get(prefixLen).intValue();
                        if (tokenToMask >= 0 && tokenToMask < logits.length) {
                            logits[tokenToMask] = Float.NEGATIVE_INFINITY;
                            maskedCount++;
                            logger.trace("Masked token {} to prevent bad word completion: {}",
                                    tokenToMask, badWordSeq);
                        }
                    }
                }
            }
        }

        if (maskedCount > 0) {
            logger.trace("Bad words filter masked {} tokens", maskedCount);
        }
    }

    /**
     * Checks if the current token sequence would complete a bad word.
     *
     * <p>Useful for validation or debugging. Does not modify logits.
     *
     * @param generatedTokens Currently generated token IDs
     * @param badWordsIds List of bad word token sequences
     * @return true if adding any token would complete a bad word
     */
    public static boolean wouldCompleteBadWord(List<Long> generatedTokens, List<List<Long>> badWordsIds) {
        if (badWordsIds == null || badWordsIds.isEmpty() || generatedTokens == null) {
            return false;
        }

        for (List<Long> badWordSeq : badWordsIds) {
            if (badWordSeq.size() == 1) {
                // Single-token bad words are always masked
                continue;
            }

            int prefixLen = badWordSeq.size() - 1;
            int genLen = generatedTokens.size();

            if (genLen >= prefixLen) {
                List<Long> suffix = generatedTokens.subList(genLen - prefixLen, genLen);

                boolean prefixMatches = true;
                for (int i = 0; i < prefixLen; i++) {
                    if (!suffix.get(i).equals(badWordSeq.get(i))) {
                        prefixMatches = false;
                        break;
                    }
                }

                if (prefixMatches) {
                    return true;
                }
            }
        }

        return false;
    }
}
