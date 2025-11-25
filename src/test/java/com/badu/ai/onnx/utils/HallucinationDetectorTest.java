package com.badu.ai.onnx.utils;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for HallucinationDetector.
 */
@DisplayName("HallucinationDetector Tests")
class HallucinationDetectorTest {

    private HallucinationDetector detector;

    @BeforeEach
    void setUp() {
        // Create detector with sequence length 5 and threshold 3
        detector = new HallucinationDetector(5, 3);
    }

    @Test
    @DisplayName("No hallucination for unique tokens")
    void testNoHallucinationUniqueTokens() {
        List<Long> tokens = Arrays.asList(1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L, 10L);

        assertFalse(detector.isHallucinating(tokens),
                "Should not detect hallucination for unique tokens");
    }

    @Test
    @DisplayName("No hallucination for tokens below threshold")
    void testNoHallucinationBelowThreshold() {
        // Sequence [1,2,3,4,5] repeats twice (below threshold of 3)
        List<Long> tokens = new ArrayList<>(Arrays.asList(
                1L, 2L, 3L, 4L, 5L,  // First occurrence
                1L, 2L, 3L, 4L, 5L   // Second occurrence
        ));

        assertFalse(detector.isHallucinating(tokens),
                "Should not detect hallucination for 2 repetitions (threshold is 3)");
    }

    @Test
    @DisplayName("Detect hallucination for consecutive repetitions")
    void testDetectConsecutiveRepetitions() {
        // Sequence [1,2,3,4,5] repeats 4 times (above threshold of 3)
        List<Long> tokens = new ArrayList<>();
        for (int i = 0; i < 4; i++) {
            tokens.addAll(Arrays.asList(1L, 2L, 3L, 4L, 5L));
        }

        assertTrue(detector.isHallucinating(tokens),
                "Should detect hallucination for 4 consecutive repetitions");
    }

    @Test
    @DisplayName("Detect hallucination for scattered repetitions")
    void testDetectScatteredRepetitions() {
        // Sequence [1,2,3,4,5] appears multiple times with different sequences in between
        // The detector needs to be called incrementally to track sequences properly
        List<Long> tokens = new ArrayList<>();

        // Add first occurrence
        tokens.addAll(Arrays.asList(1L, 2L, 3L, 4L, 5L));
        assertFalse(detector.isHallucinating(tokens), "First occurrence should not trigger");

        // Add different sequence
        tokens.addAll(Arrays.asList(6L, 7L, 8L, 9L, 10L));

        // Add second occurrence
        tokens.addAll(Arrays.asList(1L, 2L, 3L, 4L, 5L));
        assertFalse(detector.isHallucinating(tokens), "Second occurrence should not trigger");

        // Add different sequence
        tokens.addAll(Arrays.asList(11L, 12L, 13L, 14L, 15L));

        // Add third occurrence
        tokens.addAll(Arrays.asList(1L, 2L, 3L, 4L, 5L));
        assertFalse(detector.isHallucinating(tokens), "Third occurrence should not trigger (at threshold)");

        // Add different sequence
        tokens.addAll(Arrays.asList(16L, 17L, 18L, 19L, 20L));

        // Add fourth occurrence - should trigger
        tokens.addAll(Arrays.asList(1L, 2L, 3L, 4L, 5L));
        assertTrue(detector.isHallucinating(tokens),
                "Fourth occurrence should trigger (above threshold)");
    }

    @Test
    @DisplayName("No hallucination for short token list")
    void testShortTokenList() {
        List<Long> tokens = Arrays.asList(1L, 2L, 3L);

        assertFalse(detector.isHallucinating(tokens),
                "Should not detect hallucination for token list shorter than sequence length");
    }

    @Test
    @DisplayName("Reset clears detection state")
    void testReset() {
        // Build up repetitions
        List<Long> tokens = new ArrayList<>();
        for (int i = 0; i < 4; i++) {
            tokens.addAll(Arrays.asList(1L, 2L, 3L, 4L, 5L));
        }

        assertTrue(detector.isHallucinating(tokens),
                "Should detect hallucination before reset");

        // Reset and check with new sequence
        detector.reset();
        List<Long> newTokens = Arrays.asList(6L, 7L, 8L, 9L, 10L);

        assertFalse(detector.isHallucinating(newTokens),
                "Should not detect hallucination after reset with new tokens");
    }

    @Test
    @DisplayName("Different sequence lengths")
    void testDifferentSequenceLengths() {
        // Test with sequence length 3
        HallucinationDetector detector3 = new HallucinationDetector(3, 3);

        List<Long> tokens = new ArrayList<>();
        for (int i = 0; i < 4; i++) {
            tokens.addAll(Arrays.asList(1L, 2L, 3L));
        }

        assertTrue(detector3.isHallucinating(tokens),
                "Should detect hallucination with sequence length 3");
    }

    @Test
    @DisplayName("Whisper-like hallucination pattern")
    void testWhisperLikePattern() {
        // Simulate Whisper hallucination: "Thank you for watching."
        // Token sequence: [1077, 345, 329, 4964, 13]
        List<Long> tokens = new ArrayList<>(Arrays.asList(
                50L, 51L, 52L  // Initial tokens (SOT, language, etc.)
        ));

        // Add hallucinating sequence 5 times
        for (int i = 0; i < 5; i++) {
            tokens.addAll(Arrays.asList(1077L, 345L, 329L, 4964L, 13L));
        }

        assertTrue(detector.isHallucinating(tokens),
                "Should detect Whisper-like hallucination pattern");
    }

    @Test
    @DisplayName("Partial repetition does not trigger")
    void testPartialRepetition() {
        // Sequence [1,2,3,4,5] then [1,2,3,4,6] (last token different)
        List<Long> tokens = Arrays.asList(
                1L, 2L, 3L, 4L, 5L,
                1L, 2L, 3L, 4L, 6L  // Last token different
        );

        assertFalse(detector.isHallucinating(tokens),
                "Should not detect hallucination for partial match");
    }

    @Test
    @DisplayName("Constructor validation")
    void testConstructorValidation() {
        assertThrows(IllegalArgumentException.class,
                () -> new HallucinationDetector(1, 3),
                "Should throw for sequence length < 2");

        assertThrows(IllegalArgumentException.class,
                () -> new HallucinationDetector(5, 1),
                "Should throw for repetition threshold < 2");
    }

    @Test
    @DisplayName("Getters return correct values")
    void testGetters() {
        assertEquals(5, detector.getSequenceLength(),
                "Sequence length should be 5");
        assertEquals(3, detector.getRepetitionThreshold(),
                "Repetition threshold should be 3");

        // Initially, no sequences detected
        assertEquals(0, detector.getUniqueSequenceCount(),
                "Should have 0 unique sequences initially");

        // Add some tokens
        List<Long> tokens = Arrays.asList(1L, 2L, 3L, 4L, 5L);
        detector.isHallucinating(tokens);

        assertTrue(detector.getUniqueSequenceCount() > 0,
                "Should have detected at least one unique sequence");
    }

    @Test
    @DisplayName("toString provides useful info")
    void testToString() {
        String str = detector.toString();

        assertTrue(str.contains("5"), "toString should contain sequence length");
        assertTrue(str.contains("3"), "toString should contain threshold");
    }

    @Test
    @DisplayName("Incremental token addition")
    void testIncrementalAddition() {
        // Simulate real-world usage: adding tokens one at a time
        List<Long> tokens = new ArrayList<>();

        // Add tokens incrementally
        for (int repeat = 0; repeat < 4; repeat++) {
            for (int i = 1; i <= 5; i++) {
                tokens.add((long) i);

                // Check after each addition
                if (repeat < 3) {
                    assertFalse(detector.isHallucinating(tokens),
                            "Should not detect before threshold at repeat " + repeat);
                }
            }
        }

        // After 4th repetition, should detect
        assertTrue(detector.isHallucinating(tokens),
                "Should detect hallucination after 4th repetition");
    }

    @Test
    @DisplayName("Exact threshold boundary")
    void testThresholdBoundary() {
        // Test exact threshold (3 repetitions)
        List<Long> tokens = new ArrayList<>();
        for (int i = 0; i < 3; i++) {
            tokens.addAll(Arrays.asList(1L, 2L, 3L, 4L, 5L));
        }

        // At threshold, should not trigger (threshold is EXCLUSIVE)
        assertFalse(detector.isHallucinating(tokens),
                "Should not trigger at exact threshold (3 repetitions)");

        // Add one more token to start 4th repetition
        tokens.add(1L);
        tokens.add(2L);
        tokens.add(3L);
        tokens.add(4L);
        tokens.add(5L);

        // Above threshold, should trigger
        assertTrue(detector.isHallucinating(tokens),
                "Should trigger above threshold (4 repetitions)");
    }
}
