package com.badu.ai.onnx.utils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Detects hallucination patterns in generated token sequences.
 *
 * <p>Whisper models can sometimes hallucinate by repeating the same token
 * sequence indefinitely. This detector identifies such patterns and signals
 * when generation should stop early.
 *
 * <p><strong>Detection Strategy:</strong>
 * <ol>
 *   <li>Track recent token sequences of fixed length</li>
 *   <li>Count how many times each sequence appears</li>
 *   <li>Trigger early stopping if a sequence repeats more than threshold times</li>
 * </ol>
 *
 * <p><strong>Example Hallucination:</strong>
 * <pre>
 * "Thank you for watching. Thank you for watching. Thank you for watching..."
 * Token sequence [1077, 345, 329, 4964] repeats 5+ times -> hallucination detected
 * </pre>
 *
 * <p><strong>Usage:</strong>
 * <pre>{@code
 * HallucinationDetector detector = new HallucinationDetector(5, 3);
 *
 * for (int step = 0; step < maxSteps; step++) {
 *     long nextToken = generateToken();
 *     generatedTokens.add(nextToken);
 *
 *     if (detector.isHallucinating(generatedTokens)) {
 *         logger.warn("Hallucination detected at step {}, stopping early", step);
 *         break;
 *     }
 * }
 * }</pre>
 *
 * @see com.badu.ai.onnx.engine.WhisperEngine
 */
public class HallucinationDetector {

    private static final Logger logger = LoggerFactory.getLogger(HallucinationDetector.class);

    private final int sequenceLength;
    private final int repetitionThreshold;
    private final Map<String, Integer> sequenceCounts;

    /**
     * Creates a hallucination detector.
     *
     * @param sequenceLength Length of token sequence to track (e.g., 5 tokens)
     * @param repetitionThreshold Maximum allowed repetitions before declaring hallucination
     */
    public HallucinationDetector(int sequenceLength, int repetitionThreshold) {
        if (sequenceLength < 2) {
            throw new IllegalArgumentException("Sequence length must be at least 2");
        }
        if (repetitionThreshold < 2) {
            throw new IllegalArgumentException("Repetition threshold must be at least 2");
        }

        this.sequenceLength = sequenceLength;
        this.repetitionThreshold = repetitionThreshold;
        this.sequenceCounts = new HashMap<>();

        logger.debug("HallucinationDetector initialized: seqLen={}, threshold={}",
                sequenceLength, repetitionThreshold);
    }

    /**
     * Checks if the generated token sequence contains hallucination patterns.
     *
     * <p>This method should be called after each new token is generated.
     *
     * @param generatedTokens List of all generated tokens so far
     * @return true if hallucination detected, false otherwise
     */
    public boolean isHallucinating(List<Long> generatedTokens) {
        int totalTokens = generatedTokens.size();

        // Need at least sequenceLength tokens to check
        if (totalTokens < sequenceLength) {
            return false;
        }

        // Extract most recent sequence
        String sequence = extractSequence(generatedTokens, totalTokens - sequenceLength, totalTokens);

        // Update sequence count
        int count = sequenceCounts.getOrDefault(sequence, 0) + 1;
        sequenceCounts.put(sequence, count);

        // Check if threshold exceeded
        if (count > repetitionThreshold) {
            logger.warn("Hallucination detected: sequence '{}' repeated {} times (threshold: {})",
                    sequence, count, repetitionThreshold);
            return true;
        }

        // Check for exact repeats in sliding window
        // (e.g., "A B C D E A B C D E" - same 5 tokens repeated immediately)
        if (totalTokens >= sequenceLength * 2) {
            String prevSequence = extractSequence(generatedTokens,
                    totalTokens - sequenceLength * 2,
                    totalTokens - sequenceLength);

            if (sequence.equals(prevSequence)) {
                logger.debug("Detected immediate repetition: sequence '{}' at positions {}-{}",
                        sequence, totalTokens - sequenceLength * 2, totalTokens);

                // Check if this pattern repeats multiple times
                int consecutiveRepeats = countConsecutiveRepeats(generatedTokens, sequence);
                if (consecutiveRepeats > repetitionThreshold) {
                    logger.warn("Hallucination detected: sequence '{}' repeated {} consecutive times",
                            sequence, consecutiveRepeats);
                    return true;
                }
            }
        }

        return false;
    }

    /**
     * Counts how many consecutive times a sequence appears at the end of the token list.
     *
     * @param tokens List of tokens
     * @param sequence Sequence string to count
     * @return Number of consecutive repetitions
     */
    private int countConsecutiveRepeats(List<Long> tokens, String sequence) {
        int count = 0;
        int totalTokens = tokens.size();

        // Work backwards from the end
        for (int i = totalTokens - sequenceLength; i >= 0; i -= sequenceLength) {
            String currentSequence = extractSequence(tokens, i, i + sequenceLength);
            if (currentSequence.equals(sequence)) {
                count++;
            } else {
                break;  // Stop when pattern breaks
            }
        }

        return count;
    }

    /**
     * Extracts a token sequence as a string key.
     *
     * @param tokens List of tokens
     * @param start Start index (inclusive)
     * @param end End index (exclusive)
     * @return Sequence as string (e.g., "1234,5678,9012")
     */
    private String extractSequence(List<Long> tokens, int start, int end) {
        StringBuilder sb = new StringBuilder();
        for (int i = start; i < end; i++) {
            if (i > start) {
                sb.append(",");
            }
            sb.append(tokens.get(i));
        }
        return sb.toString();
    }

    /**
     * Resets the detector state.
     *
     * <p>Call this when starting a new transcription.
     */
    public void reset() {
        sequenceCounts.clear();
        logger.debug("HallucinationDetector reset");
    }

    /**
     * Gets the sequence length being tracked.
     *
     * @return Sequence length in tokens
     */
    public int getSequenceLength() {
        return sequenceLength;
    }

    /**
     * Gets the repetition threshold.
     *
     * @return Maximum allowed repetitions
     */
    public int getRepetitionThreshold() {
        return repetitionThreshold;
    }

    /**
     * Gets the number of unique sequences detected so far.
     *
     * @return Number of unique sequences
     */
    public int getUniqueSequenceCount() {
        return sequenceCounts.size();
    }

    @Override
    public String toString() {
        return String.format("HallucinationDetector{seqLen=%d, threshold=%d, uniqueSeqs=%d}",
                sequenceLength, repetitionThreshold, sequenceCounts.size());
    }
}
