package com.badu.ai.onnx.validation;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;

/**
 * Validates generated output text for quality and correctness.
 *
 * <p>Detects potential issues like:
 * <ul>
 *   <li>Token ID leakage (raw token IDs in output)</li>
 *   <li>Invalid UTF-8 encoding</li>
 *   <li>Excessive control characters</li>
 *   <li>Empty or malformed output</li>
 *   <li>Low readability (less than 20% letters)</li>
 * </ul>
 *
 * @see ValidationResult
 */
public class OutputValidator {

  // Pattern for detecting potential token ID leakage (5+ consecutive digits)
  private static final Pattern TOKEN_ID_PATTERN = Pattern.compile("\\b\\d{5,}\\b");

  // Pattern for detecting excessive whitespace
  private static final Pattern EXCESSIVE_WHITESPACE = Pattern.compile("\\s{10,}");

  // Pattern for control characters (except common ones like newline, tab)
  private static final Pattern CONTROL_CHARS =
      Pattern.compile("[\\x00-\\x08\\x0B\\x0C\\x0E-\\x1F\\x7F]");

  // Readability threshold (minimum percentage of letters)
  private static final double MIN_LETTER_RATIO = 0.20;

  private final int minOutputLength;
  private final int maxOutputLength;
  private final boolean strictMode;

  /**
   * Creates validator with default settings.
   * <ul>
   *   <li>minOutputLength: 1</li>
   *   <li>maxOutputLength: 100000 (conservative)</li>
   *   <li>strictMode: false</li>
   * </ul>
   */
  public OutputValidator() {
    this(1, 100000, false);
  }

  /**
   * Creates validator with custom settings.
   *
   * @param minOutputLength minimum acceptable output length
   * @param maxOutputLength maximum acceptable output length
   * @param strictMode if true, applies stricter validation rules
   */
  public OutputValidator(int minOutputLength, int maxOutputLength, boolean strictMode) {
    this.minOutputLength = minOutputLength;
    this.maxOutputLength = maxOutputLength;
    this.strictMode = strictMode;
  }

  /**
   * Validates generated output text.
   *
   * @param output the generated text to validate
   * @return validation result with details
   */
  public ValidationResult validate(String output) {
    List<String> errors = new ArrayList<>();
    List<String> warnings = new ArrayList<>();

    // Check for null
    if (output == null) {
      return ValidationResult.invalid(List.of("Output is null"));
    }

    // Check for empty
    if (output.isEmpty()) {
      return ValidationResult.invalid(List.of("Output is empty"));
    }

    // Check if output is all whitespace
    if (output.trim().isEmpty()) {
      return ValidationResult.invalid(List.of("Output is whitespace-only"));
    }

    // Check minimum length
    if (output.length() < minOutputLength) {
      errors.add(String.format("Output too short: %d < %d (minimum)",
          output.length(), minOutputLength));
    }

    // Check maximum length
    if (output.length() > maxOutputLength) {
      errors.add(String.format("Output too long: %d > %d (maximum)",
          output.length(), maxOutputLength));
    }

    // Validate UTF-8 encoding
    if (!isValidUtf8(output)) {
      errors.add("Output contains invalid UTF-8 sequences");
    }

    // Check for control characters
    if (CONTROL_CHARS.matcher(output).find()) {
      if (strictMode) {
        errors.add("Output contains invalid control characters");
      } else {
        warnings.add("Output contains control characters (may be intentional)");
      }
    }

    // Check for potential token ID leakage
    if (TOKEN_ID_PATTERN.matcher(output).find()) {
      if (strictMode) {
        errors.add("Output contains sequences of 5+ digits (potential token ID leakage)");
      } else {
        warnings.add("Potential token ID leakage: 5+ consecutive digits detected");
      }
    }

    // Check for excessive whitespace
    if (EXCESSIVE_WHITESPACE.matcher(output).find()) {
      warnings.add("Excessive whitespace detected (10+ consecutive spaces)");
    }

    // Check readability (≥20% letters)
    if (!isHumanReadable(output)) {
      long letterCount = output.chars().filter(Character::isLetter).count();
      double letterRatio = (double) letterCount / output.length();
      warnings.add(String.format(
          "Output readability low: only %.1f%% letters (minimum %.0f%% recommended)",
          letterRatio * 100, MIN_LETTER_RATIO * 100));
    }

    // Return result
    if (!errors.isEmpty()) {
      return ValidationResult.invalid(errors, warnings);
    }

    if (!warnings.isEmpty()) {
      return ValidationResult.validWithWarnings(warnings);
    }

    return ValidationResult.valid();
  }

  /**
   * Validates UTF-8 encoding of output.
   *
   * @param text the text to validate
   * @return true if valid UTF-8, false otherwise
   */
  private boolean isValidUtf8(String text) {
    try {
      // Encode to UTF-8 and decode back
      byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
      String decoded = new String(bytes, StandardCharsets.UTF_8);

      // Check for replacement character (U+FFFD) which indicates encoding issues
      // But allow it if it was in the original text
      return !decoded.contains("\uFFFD") || text.contains("\uFFFD");
    } catch (Exception e) {
      return false;
    }
  }

  /**
   * Checks if output appears to be human-readable.
   * Basic heuristic: contains at least 20% letters.
   *
   * @param output the output text
   * @return true if appears human-readable
   */
  public boolean isHumanReadable(String output) {
    if (output == null || output.isEmpty()) {
      return false;
    }

    // Count letters
    long letterCount = output.chars().filter(Character::isLetter).count();

    // Human-readable text should have at least 20% letters
    return letterCount >= output.length() * MIN_LETTER_RATIO;
  }
}
