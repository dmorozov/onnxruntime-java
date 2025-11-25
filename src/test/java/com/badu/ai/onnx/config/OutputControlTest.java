package com.badu.ai.onnx.config;

import com.badu.ai.onnx.genai.internal.BadWordsProcessor;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive tests for Output Control Features (minOutputTokens and badWordsIds).
 *
 * <p>Tests cover:
 * <ul>
 *   <li>GenerationConfig validation for minOutputTokens and badWordsIds</li>
 *   <li>BadWordsProcessor filtering logic (single-token and multi-token)</li>
 *   <li>Edge cases and error conditions</li>
 * </ul>
 */
@DisplayName("Output Control Features Tests")
class OutputControlTest {

    // ========================================================================
    // GenerationConfig - minOutputTokens Validation Tests
    // ========================================================================

    @Test
    @DisplayName("minOutputTokens: default value is 0 (disabled)")
    void minOutputTokens_defaultIsZero() {
        GenerationConfig config = GenerationConfig.builder().build();
        assertEquals(0, config.getMinOutputTokens());
        assertFalse(config.hasMinLength());
    }

    @Test
    @DisplayName("minOutputTokens: can be set to valid value")
    void minOutputTokens_canBeSet() {
        GenerationConfig config = GenerationConfig.builder()
                .minOutputTokens(20)
                .build();
        assertEquals(20, config.getMinOutputTokens());
        assertTrue(config.hasMinLength());
    }

    @Test
    @DisplayName("minOutputTokens: accepts zero (disabled)")
    void minOutputTokens_acceptsZero() {
        GenerationConfig config = GenerationConfig.builder()
                .minOutputTokens(0)
                .build();
        assertEquals(0, config.getMinOutputTokens());
        assertFalse(config.hasMinLength());
    }

    @Test
    @DisplayName("minOutputTokens: rejects negative values")
    void minOutputTokens_rejectsNegative() {
        IllegalStateException exception = assertThrows(IllegalStateException.class, () -> {
            GenerationConfig.builder()
                    .minOutputTokens(-1)
                    .build();
        });
        assertTrue(exception.getMessage().contains("minOutputTokens must be >= 0"));
    }

    @Test
    @DisplayName("minOutputTokens: cannot exceed maxOutputTokens")
    void minOutputTokens_cannotExceedMax() {
        IllegalStateException exception = assertThrows(IllegalStateException.class, () -> {
            GenerationConfig.builder()
                    .maxOutputTokens(100)
                    .minOutputTokens(150)
                    .build();
        });
        assertTrue(exception.getMessage().contains("must be <= maxOutputTokens"));
    }

    @Test
    @DisplayName("minOutputTokens: can equal maxOutputTokens")
    void minOutputTokens_canEqualMax() {
        GenerationConfig config = GenerationConfig.builder()
                .maxOutputTokens(100)
                .minOutputTokens(100)
                .build();
        assertEquals(100, config.getMinOutputTokens());
        assertEquals(100, config.getMaxOutputTokens());
    }

    // ========================================================================
    // GenerationConfig - badWordsIds Validation Tests
    // ========================================================================

    @Test
    @DisplayName("badWordsIds: default is empty list")
    void badWordsIds_defaultIsEmpty() {
        GenerationConfig config = GenerationConfig.builder().build();
        assertNotNull(config.getBadWordsIds());
        assertTrue(config.getBadWordsIds().isEmpty());
        assertFalse(config.hasBadWords());
    }

    @Test
    @DisplayName("badWordsIds: can be set with valid sequences")
    void badWordsIds_canBeSet() {
        List<List<Long>> badWords = Arrays.asList(
                Arrays.asList(123L, 456L, 789L),  // Multi-token
                Arrays.asList(999L)                // Single-token
        );

        GenerationConfig config = GenerationConfig.builder()
                .badWordsIds(badWords)
                .build();

        assertNotNull(config.getBadWordsIds());
        assertEquals(2, config.getBadWordsIds().size());
        assertTrue(config.hasBadWords());
    }

    @Test
    @DisplayName("badWordsIds: is immutable after build")
    void badWordsIds_isImmutable() {
        List<List<Long>> badWords = new ArrayList<>();
        badWords.add(new ArrayList<>(Arrays.asList(123L, 456L)));

        GenerationConfig config = GenerationConfig.builder()
                .badWordsIds(badWords)
                .build();

        // Try to modify original list
        badWords.add(Arrays.asList(999L));

        // Config should be unchanged
        assertEquals(1, config.getBadWordsIds().size());

        // Try to modify returned list (should throw UnsupportedOperationException)
        assertThrows(UnsupportedOperationException.class, () -> {
            config.getBadWordsIds().add(Arrays.asList(777L));
        });
    }

    @Test
    @DisplayName("badWordsIds: rejects null list")
    void badWordsIds_rejectsNull() {
        IllegalStateException exception = assertThrows(IllegalStateException.class, () -> {
            GenerationConfig.builder()
                    .badWordsIds(null)
                    .build();
        });
        assertTrue(exception.getMessage().contains("cannot be null"));
    }

    @Test
    @DisplayName("badWordsIds: rejects empty sequences")
    void badWordsIds_rejectsEmptySequences() {
        List<List<Long>> badWords = Arrays.asList(
                Arrays.asList(123L, 456L),
                Collections.emptyList()  // Empty sequence - invalid
        );

        IllegalStateException exception = assertThrows(IllegalStateException.class, () -> {
            GenerationConfig.builder()
                    .badWordsIds(badWords)
                    .build();
        });
        assertTrue(exception.getMessage().contains("cannot be null or empty"));
    }

    @Test
    @DisplayName("badWordsIds: rejects null sequences")
    void badWordsIds_rejectsNullSequences() {
        List<List<Long>> badWords = new ArrayList<>();
        badWords.add(Arrays.asList(123L));
        badWords.add(null);  // Null sequence - invalid

        IllegalStateException exception = assertThrows(IllegalStateException.class, () -> {
            GenerationConfig.builder()
                    .badWordsIds(badWords)
                    .build();
        });
        assertTrue(exception.getMessage().contains("cannot be null or empty"));
    }

    // ========================================================================
    // BadWordsProcessor - Single-Token Bad Words Tests
    // ========================================================================

    @Test
    @DisplayName("BadWordsProcessor: masks single-token bad word")
    void badWordsProcessor_masksSingleToken() {
        List<Long> generatedTokens = Arrays.asList(100L, 200L, 300L);
        List<List<Long>> badWords = Arrays.asList(
                Arrays.asList(777L)  // Single-token bad word
        );

        float[] logits = new float[1000];
        Arrays.fill(logits, 1.0f);

        BadWordsProcessor.applyBadWordsFilter(logits, generatedTokens, badWords);

        // Token 777 should be masked
        assertEquals(Float.NEGATIVE_INFINITY, logits[777]);

        // Other tokens should be unchanged
        assertEquals(1.0f, logits[0]);
        assertEquals(1.0f, logits[100]);
        assertEquals(1.0f, logits[999]);
    }

    @Test
    @DisplayName("BadWordsProcessor: masks multiple single-token bad words")
    void badWordsProcessor_masksMultipleSingleTokens() {
        List<Long> generatedTokens = Arrays.asList(100L);
        List<List<Long>> badWords = Arrays.asList(
                Arrays.asList(111L),
                Arrays.asList(222L),
                Arrays.asList(333L)
        );

        float[] logits = new float[1000];
        Arrays.fill(logits, 1.0f);

        BadWordsProcessor.applyBadWordsFilter(logits, generatedTokens, badWords);

        assertEquals(Float.NEGATIVE_INFINITY, logits[111]);
        assertEquals(Float.NEGATIVE_INFINITY, logits[222]);
        assertEquals(Float.NEGATIVE_INFINITY, logits[333]);
        assertEquals(1.0f, logits[0]);
    }

    // ========================================================================
    // BadWordsProcessor - Multi-Token Bad Words Tests
    // ========================================================================

    @Test
    @DisplayName("BadWordsProcessor: masks multi-token bad word when prefix matches")
    void badWordsProcessor_masksMultiTokenWhenPrefixMatches() {
        // Generated: [..., 123, 456] - prefix of [123, 456, 789]
        List<Long> generatedTokens = Arrays.asList(100L, 123L, 456L);
        List<List<Long>> badWords = Arrays.asList(
                Arrays.asList(123L, 456L, 789L)  // 3-token bad word
        );

        float[] logits = new float[1000];
        Arrays.fill(logits, 1.0f);

        BadWordsProcessor.applyBadWordsFilter(logits, generatedTokens, badWords);

        // Token 789 should be masked (would complete bad word)
        assertEquals(Float.NEGATIVE_INFINITY, logits[789]);

        // Prefix tokens should NOT be masked
        assertEquals(1.0f, logits[123]);
        assertEquals(1.0f, logits[456]);
    }

    @Test
    @DisplayName("BadWordsProcessor: does NOT mask when prefix doesn't match")
    void badWordsProcessor_doesNotMaskWhenPrefixDoesNotMatch() {
        // Generated: [..., 100, 200] - does NOT match prefix of [123, 456, 789]
        List<Long> generatedTokens = Arrays.asList(100L, 200L);
        List<List<Long>> badWords = Arrays.asList(
                Arrays.asList(123L, 456L, 789L)
        );

        float[] logits = new float[1000];
        Arrays.fill(logits, 1.0f);

        BadWordsProcessor.applyBadWordsFilter(logits, generatedTokens, badWords);

        // No tokens should be masked
        assertEquals(1.0f, logits[789]);
        assertEquals(1.0f, logits[123]);
        assertEquals(1.0f, logits[456]);
    }

    @Test
    @DisplayName("BadWordsProcessor: does NOT mask when not enough tokens for full prefix")
    void badWordsProcessor_doesNotMaskWhenNotEnoughTokensForFullPrefix() {
        // Generated: [123] - only first token of [123, 456, 789]
        // Prefix length needed: 2 tokens [123, 456]
        // Generated tokens: 1 token [123]
        // Result: Not enough tokens to match full prefix, so NO masking
        List<Long> generatedTokens = Arrays.asList(123L);
        List<List<Long>> badWords = Arrays.asList(
                Arrays.asList(123L, 456L, 789L)  // Need prefix [123, 456] before masking 789
        );

        float[] logits = new float[1000];
        Arrays.fill(logits, 1.0f);

        BadWordsProcessor.applyBadWordsFilter(logits, generatedTokens, badWords);

        // No tokens should be masked (not enough generated tokens to match prefix)
        assertEquals(1.0f, logits[456]);
        assertEquals(1.0f, logits[789]);
        assertEquals(1.0f, logits[123]);
    }

    @Test
    @DisplayName("BadWordsProcessor: handles exact prefix match")
    void badWordsProcessor_handlesExactPrefixMatch() {
        // Bad word: [AA, BB, CC]
        // Generated: [AA, BB] → should mask CC
        List<Long> generatedTokens = Arrays.asList(10L, 20L);
        List<List<Long>> badWords = Arrays.asList(
                Arrays.asList(10L, 20L, 30L)
        );

        float[] logits = new float[100];
        Arrays.fill(logits, 1.0f);

        BadWordsProcessor.applyBadWordsFilter(logits, generatedTokens, badWords);

        assertEquals(Float.NEGATIVE_INFINITY, logits[30]);
    }

    @Test
    @DisplayName("BadWordsProcessor: masks multiple completion tokens when multiple bad words match")
    void badWordsProcessor_masksMultipleCompletions() {
        // Generated: [10, 20]
        // Bad word 1: [10, 20, 30]
        // Bad word 2: [10, 20, 40]
        // Should mask both 30 and 40
        List<Long> generatedTokens = Arrays.asList(10L, 20L);
        List<List<Long>> badWords = Arrays.asList(
                Arrays.asList(10L, 20L, 30L),
                Arrays.asList(10L, 20L, 40L)
        );

        float[] logits = new float[100];
        Arrays.fill(logits, 1.0f);

        BadWordsProcessor.applyBadWordsFilter(logits, generatedTokens, badWords);

        assertEquals(Float.NEGATIVE_INFINITY, logits[30]);
        assertEquals(Float.NEGATIVE_INFINITY, logits[40]);
        assertEquals(1.0f, logits[50]);  // Unrelated token
    }

    // ========================================================================
    // BadWordsProcessor - Edge Cases
    // ========================================================================

    @Test
    @DisplayName("BadWordsProcessor: handles empty bad words list")
    void badWordsProcessor_handlesEmptyBadWordsList() {
        List<Long> generatedTokens = Arrays.asList(100L, 200L);
        List<List<Long>> badWords = Collections.emptyList();

        float[] logits = new float[1000];
        Arrays.fill(logits, 5.0f);

        BadWordsProcessor.applyBadWordsFilter(logits, generatedTokens, badWords);

        // No tokens should be masked
        assertEquals(5.0f, logits[0]);
        assertEquals(5.0f, logits[500]);
        assertEquals(5.0f, logits[999]);
    }

    @Test
    @DisplayName("BadWordsProcessor: handles null bad words list gracefully")
    void badWordsProcessor_handlesNullBadWordsList() {
        List<Long> generatedTokens = Arrays.asList(100L, 200L);

        float[] logits = new float[1000];
        Arrays.fill(logits, 5.0f);

        // Should not throw
        assertDoesNotThrow(() -> {
            BadWordsProcessor.applyBadWordsFilter(logits, generatedTokens, null);
        });

        // No tokens should be masked
        assertEquals(5.0f, logits[0]);
    }

    @Test
    @DisplayName("BadWordsProcessor: handles empty generated tokens")
    void badWordsProcessor_handlesEmptyGeneratedTokens() {
        List<Long> generatedTokens = Collections.emptyList();
        List<List<Long>> badWords = Arrays.asList(
                Arrays.asList(123L, 456L)
        );

        float[] logits = new float[1000];
        Arrays.fill(logits, 1.0f);

        BadWordsProcessor.applyBadWordsFilter(logits, generatedTokens, badWords);

        // No tokens should be masked (no prefix to match)
        assertEquals(1.0f, logits[123]);
        assertEquals(1.0f, logits[456]);
    }

    @Test
    @DisplayName("BadWordsProcessor: handles null generated tokens gracefully")
    void badWordsProcessor_handlesNullGeneratedTokens() {
        List<List<Long>> badWords = Arrays.asList(
                Arrays.asList(123L, 456L)
        );

        float[] logits = new float[1000];
        Arrays.fill(logits, 1.0f);

        assertDoesNotThrow(() -> {
            BadWordsProcessor.applyBadWordsFilter(logits, null, badWords);
        });
    }

    @Test
    @DisplayName("BadWordsProcessor: handles out-of-bounds token IDs gracefully")
    void badWordsProcessor_handlesOutOfBoundsTokenIds() {
        List<Long> generatedTokens = Arrays.asList(10L, 20L);
        List<List<Long>> badWords = Arrays.asList(
                Arrays.asList(10L, 20L, 999999L)  // Token ID way out of vocab range
        );

        float[] logits = new float[1000];  // vocab size = 1000
        Arrays.fill(logits, 1.0f);

        // Should not throw (gracefully ignores out-of-bounds)
        assertDoesNotThrow(() -> {
            BadWordsProcessor.applyBadWordsFilter(logits, generatedTokens, badWords);
        });
    }

    @Test
    @DisplayName("BadWordsProcessor: handles negative token IDs gracefully")
    void badWordsProcessor_handlesNegativeTokenIds() {
        List<Long> generatedTokens = Arrays.asList(10L, 20L);
        List<List<Long>> badWords = Arrays.asList(
                Arrays.asList(10L, 20L, -5L)  // Negative token ID
        );

        float[] logits = new float[1000];
        Arrays.fill(logits, 1.0f);

        // Should not throw (gracefully ignores negative IDs)
        assertDoesNotThrow(() -> {
            BadWordsProcessor.applyBadWordsFilter(logits, generatedTokens, badWords);
        });
    }

    // ========================================================================
    // BadWordsProcessor - wouldCompleteBadWord() Tests
    // ========================================================================

    @Test
    @DisplayName("wouldCompleteBadWord: returns true when prefix matches")
    void wouldCompleteBadWord_returnsTrueWhenPrefixMatches() {
        List<Long> generatedTokens = Arrays.asList(10L, 20L);
        List<List<Long>> badWords = Arrays.asList(
                Arrays.asList(10L, 20L, 30L)
        );

        assertTrue(BadWordsProcessor.wouldCompleteBadWord(generatedTokens, badWords));
    }

    @Test
    @DisplayName("wouldCompleteBadWord: returns false when prefix doesn't match")
    void wouldCompleteBadWord_returnsFalseWhenPrefixDoesNotMatch() {
        List<Long> generatedTokens = Arrays.asList(100L, 200L);
        List<List<Long>> badWords = Arrays.asList(
                Arrays.asList(10L, 20L, 30L)
        );

        assertFalse(BadWordsProcessor.wouldCompleteBadWord(generatedTokens, badWords));
    }

    @Test
    @DisplayName("wouldCompleteBadWord: handles single-token bad words (always false)")
    void wouldCompleteBadWord_handlesSingleTokenBadWords() {
        List<Long> generatedTokens = Arrays.asList(10L, 20L);
        List<List<Long>> badWords = Arrays.asList(
                Arrays.asList(777L)  // Single-token bad words don't have "completions"
        );

        assertFalse(BadWordsProcessor.wouldCompleteBadWord(generatedTokens, badWords));
    }

    @Test
    @DisplayName("wouldCompleteBadWord: handles empty lists")
    void wouldCompleteBadWord_handlesEmptyLists() {
        assertFalse(BadWordsProcessor.wouldCompleteBadWord(
                Collections.emptyList(), Arrays.asList(Arrays.asList(1L, 2L))));

        assertFalse(BadWordsProcessor.wouldCompleteBadWord(
                Arrays.asList(1L, 2L), Collections.emptyList()));

        assertFalse(BadWordsProcessor.wouldCompleteBadWord(
                Collections.emptyList(), Collections.emptyList()));
    }

    @Test
    @DisplayName("wouldCompleteBadWord: handles null inputs")
    void wouldCompleteBadWord_handlesNullInputs() {
        assertFalse(BadWordsProcessor.wouldCompleteBadWord(null, null));
        assertFalse(BadWordsProcessor.wouldCompleteBadWord(
                Arrays.asList(1L), null));
        assertFalse(BadWordsProcessor.wouldCompleteBadWord(
                null, Arrays.asList(Arrays.asList(1L, 2L))));
    }

    // ========================================================================
    // Combined Features Tests
    // ========================================================================

    @Test
    @DisplayName("Combined: minOutputTokens and badWordsIds can be used together")
    void combined_bothFeaturesCanBeUsedTogether() {
        List<List<Long>> badWords = Arrays.asList(
                Arrays.asList(123L, 456L)
        );

        GenerationConfig config = GenerationConfig.builder()
                .minOutputTokens(20)
                .maxOutputTokens(256)
                .badWordsIds(badWords)
                .build();

        assertTrue(config.hasMinLength());
        assertTrue(config.hasBadWords());
        assertEquals(20, config.getMinOutputTokens());
        assertEquals(1, config.getBadWordsIds().size());
    }

    @Test
    @DisplayName("Combined: both features disabled by default")
    void combined_bothDisabledByDefault() {
        GenerationConfig config = GenerationConfig.builder().build();

        assertFalse(config.hasMinLength());
        assertFalse(config.hasBadWords());
    }

    // ========================================================================
    // Complex Scenarios
    // ========================================================================

    @Test
    @DisplayName("Complex: long sequence with overlapping bad word patterns")
    void complex_longSequenceWithOverlappingPatterns() {
        // Generated: [1, 2, 3, 4]
        // Bad word 1: [2, 3, 4, 5]  - prefix [2,3,4] matches → mask 5
        // Bad word 2: [3, 4, 6]     - prefix [3,4] matches → mask 6
        // Bad word 3: [4, 7]        - prefix [4] matches → mask 7
        List<Long> generatedTokens = Arrays.asList(1L, 2L, 3L, 4L);
        List<List<Long>> badWords = Arrays.asList(
                Arrays.asList(2L, 3L, 4L, 5L),
                Arrays.asList(3L, 4L, 6L),
                Arrays.asList(4L, 7L)
        );

        float[] logits = new float[100];
        Arrays.fill(logits, 1.0f);

        BadWordsProcessor.applyBadWordsFilter(logits, generatedTokens, badWords);

        assertEquals(Float.NEGATIVE_INFINITY, logits[5]);
        assertEquals(Float.NEGATIVE_INFINITY, logits[6]);
        assertEquals(Float.NEGATIVE_INFINITY, logits[7]);
        assertEquals(1.0f, logits[8]);  // Unmasked
    }

    @Test
    @DisplayName("Complex: very long bad word sequence (10 tokens)")
    void complex_veryLongBadWordSequence() {
        // Test that algorithm handles long sequences efficiently
        List<Long> generatedTokens = Arrays.asList(1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L);
        List<List<Long>> badWords = Arrays.asList(
                Arrays.asList(1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L, 10L)  // 10-token bad word
        );

        float[] logits = new float[100];
        Arrays.fill(logits, 1.0f);

        BadWordsProcessor.applyBadWordsFilter(logits, generatedTokens, badWords);

        assertEquals(Float.NEGATIVE_INFINITY, logits[10]);
    }

    @Test
    @DisplayName("Complex: many bad words (100+)")
    void complex_manyBadWords() {
        List<Long> generatedTokens = Arrays.asList(1L, 2L);

        // Create 100 bad words
        List<List<Long>> badWords = new ArrayList<>();
        for (int i = 0; i < 100; i++) {
            badWords.add(Arrays.asList(1L, 2L, (long) (10 + i)));
        }

        float[] logits = new float[200];
        Arrays.fill(logits, 1.0f);

        // Should complete in reasonable time
        long startTime = System.currentTimeMillis();
        BadWordsProcessor.applyBadWordsFilter(logits, generatedTokens, badWords);
        long elapsed = System.currentTimeMillis() - startTime;

        // Verify masking
        for (int i = 0; i < 100; i++) {
            assertEquals(Float.NEGATIVE_INFINITY, logits[10 + i],
                    "Token " + (10 + i) + " should be masked");
        }

        // Performance check: should complete in < 100ms
        assertTrue(elapsed < 100, "Filter should complete in <100ms, took: " + elapsed + "ms");
    }
}
