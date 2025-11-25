package com.badu.ai.onnx.utils;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Shared utility methods for ONNX inference tests.
 *
 * <p>This class provides common helper methods used across multiple test classes:
 * <ul>
 *   <li>Array operations (argmax)</li>
 *   <li>Token utilities (EOS detection)</li>
 *   <li>Path validation</li>
 *   <li>Common test assertions</li>
 * </ul>
 *
 * <p>Thread Safety: All methods are stateless and thread-safe.
 */
public class TestUtils {

    /**
     * Finds the index of the maximum value in an array.
     *
     * <p>Used for greedy decoding: select token with highest logit value.
     *
     * @param array float array (e.g., logits from model output)
     * @return index of maximum value
     * @throws IllegalArgumentException if array is null or empty
     */
    public static int argmax(float[] array) {
        if (array == null || array.length == 0) {
            throw new IllegalArgumentException("Array cannot be null or empty");
        }

        int maxIdx = 0;
        float maxVal = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] > maxVal) {
                maxVal = array[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }

    /**
     * Checks if a token ID is an end-of-sequence (EOS) token.
     *
     * <p>Different models use different EOS token IDs:
     * <ul>
     *   <li>T5: 1</li>
     *   <li>Phi-3: 32000, 32001, 32007</li>
     *   <li>GPT models: varies</li>
     * </ul>
     *
     * @param tokenId token ID to check
     * @param eosTokenIds array of valid EOS token IDs for the model
     * @return true if tokenId is in eosTokenIds array
     */
    public static boolean isEosToken(int tokenId, int... eosTokenIds) {
        for (int eosId : eosTokenIds) {
            if (tokenId == eosId) {
                return true;
            }
        }
        return false;
    }

    /**
     * Checks if a model file exists.
     *
     * @param modelPath path to model file or directory
     * @return true if file/directory exists
     */
    public static boolean modelExists(String modelPath) {
        return Files.exists(Paths.get(modelPath));
    }

    /**
     * Checks if a model file exists.
     *
     * @param modelPath path to model file or directory
     * @return true if file/directory exists
     */
    public static boolean modelExists(Path modelPath) {
        return Files.exists(modelPath);
    }

    /**
     * Validates that a path exists and is a file.
     *
     * @param path path to validate
     * @throws IllegalArgumentException if path doesn't exist or is not a file
     */
    public static void requireModelFile(Path path) {
        if (!Files.exists(path)) {
            throw new IllegalArgumentException("Model file not found: " + path);
        }
        if (!Files.isRegularFile(path)) {
            throw new IllegalArgumentException("Path is not a file: " + path);
        }
    }

    /**
     * Validates that a path exists and is a directory.
     *
     * @param path path to validate
     * @throws IllegalArgumentException if path doesn't exist or is not a directory
     */
    public static void requireModelDirectory(Path path) {
        if (!Files.exists(path)) {
            throw new IllegalArgumentException("Model directory not found: " + path);
        }
        if (!Files.isDirectory(path)) {
            throw new IllegalArgumentException("Path is not a directory: " + path);
        }
    }

    /**
     * Formats a duration in milliseconds for display.
     *
     * @param durationMs duration in milliseconds
     * @return formatted string (e.g., "1.23s", "456ms")
     */
    public static String formatDuration(long durationMs) {
        if (durationMs >= 1000) {
            return String.format("%.2fs", durationMs / 1000.0);
        } else {
            return durationMs + "ms";
        }
    }

    /**
     * Formats throughput (tokens per second).
     *
     * @param tokens number of tokens generated
     * @param durationMs time taken in milliseconds
     * @return formatted string (e.g., "123.4 tok/s")
     */
    public static String formatThroughput(int tokens, long durationMs) {
        if (durationMs == 0) {
            return "N/A";
        }
        double tokensPerSec = (tokens * 1000.0) / durationMs;
        return String.format("%.1f tok/s", tokensPerSec);
    }

    /**
     * Prints a separator line for test output.
     *
     * @param length length of separator
     */
    public static void printSeparator(int length) {
        System.out.println("=".repeat(length));
    }

    /**
     * Prints a section header for test output.
     *
     * @param title section title
     * @param length total width
     */
    public static void printHeader(String title, int length) {
        printSeparator(length);
        System.out.println(title);
        printSeparator(length);
    }

    /**
     * Truncates a string to max length with ellipsis.
     *
     * @param text text to truncate
     * @param maxLength maximum length
     * @return truncated string
     */
    public static String truncate(String text, int maxLength) {
        if (text == null) {
            return "";
        }
        if (text.length() <= maxLength) {
            return text;
        }
        return text.substring(0, maxLength - 3) + "...";
    }

    private TestUtils() {
        // Utility class - prevent instantiation
    }
}
