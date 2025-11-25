package com.badu.ai.onnx.validation;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Validates user input for text generation.
 *
 * <p>Provides validation for:
 * <ul>
 *   <li>Null/empty checks</li>
 *   <li>Length limits (characters and approximate tokens)</li>
 *   <li>UTF-8 encoding validity</li>
 *   <li>Control character detection</li>
 *   <li>Token count estimation</li>
 * </ul>
 *
 * @see ValidationResult
 */
public class InputValidator {

  // Approximate tokens per character ratio for estimation (conservative)
  private static final double TOKENS_PER_CHAR = 0.3;

  // Pattern for detecting control characters (except newline, tab, carriage return)
  private static final Pattern CONTROL_CHARS =
      Pattern.compile("[\\x00-\\x08\\x0B\\x0C\\x0E-\\x1F\\x7F]");

  // Pattern for excessive consecutive digits (potential token ID leakage)
  private static final Pattern EXCESSIVE_DIGITS = Pattern.compile("\\d{6,}");

  private final int maxInputTokens;
  private final int maxCharacters;
  private final boolean allowControlCharacters;

  /**
   * Creates validator with default settings.
   * <ul>
   *   <li>maxInputTokens: 40960 (from model config)</li>
   *   <li>maxCharacters: 150000 (conservative estimate ~45k tokens)</li>
   *   <li>allowControlCharacters: false</li>
   * </ul>
   */
  public InputValidator() {
    this(40960, 150000, false);
  }

  /**
   * Creates validator with custom settings.
   *
   * @param maxInputTokens maximum number of tokens allowed
   * @param maxCharacters maximum number of characters allowed
   * @param allowControlCharacters whether to allow control characters
   */
  public InputValidator(int maxInputTokens, int maxCharacters, boolean allowControlCharacters) {
    this.maxInputTokens = maxInputTokens;
    this.maxCharacters = maxCharacters;
    this.allowControlCharacters = allowControlCharacters;
  }

  /**
   * Validates input text with comprehensive checks.
   *
   * @param input the input text to validate
   * @return validation result with details
   */
  public ValidationResult validate(String input) {
    List<String> errors = new ArrayList<>();
    List<String> warnings = new ArrayList<>();

    // Check for null
    if (input == null) {
      return ValidationResult.invalid(List.of("Input cannot be null"));
    }

    // Check for empty
    if (input.isEmpty()) {
      return ValidationResult.invalid(List.of("Input cannot be empty"));
    }

    // Check for whitespace-only
    if (input.trim().isEmpty()) {
      return ValidationResult.invalid(List.of("Input cannot be whitespace-only"));
    }

    // Check character length
    int charLength = input.length();
    if (charLength > maxCharacters) {
      errors.add(String.format("Input exceeds maximum character limit: %d > %d characters",
          charLength, maxCharacters));
    }

    // Validate UTF-8 encoding
    Utf8ValidationResult utf8Result = validateUtf8Detailed(input);
    if (!utf8Result.isValid()) {
      if (utf8Result.byteOffset >= 0) {
        errors.add(String.format(
            "Invalid UTF-8 encoding at byte offset %d (character position ~%d). " +
            "Ensure input is valid UTF-8. Use UTF-8 encoding when reading files.",
            utf8Result.byteOffset, utf8Result.charPosition));
      } else {
        errors.add("Input contains invalid UTF-8 sequences. Ensure input is valid UTF-8.");
      }
    }

    // Check for control characters
    if (!allowControlCharacters) {
      Matcher controlMatcher = CONTROL_CHARS.matcher(input);
      if (controlMatcher.find()) {
        int position = controlMatcher.start();
        int codePoint = input.codePointAt(position);
        errors.add(String.format(
            "Control character detected at position %d (U+%04X). Use \\n for newlines, \\t for tabs.",
            position, codePoint));
      }
    }

    // Estimate token count
    int estimatedTokens = estimateTokenCount(input);
    if (estimatedTokens > maxInputTokens) {
      errors.add(String.format(
          "Input exceeds maximum token limit: estimated %d tokens > %d max. "
              + "Actual count may vary. Try reducing input length.",
          estimatedTokens, maxInputTokens));
    }

    // Check for potential token ID leakage (excessive digits) - warning only
    if (EXCESSIVE_DIGITS.matcher(input).find()) {
      warnings.add("Input contains sequences of 6+ consecutive digits. "
          + "This is valid but may indicate unintended token IDs.");
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
   * Validates input text against a specific token count (from tokenizer).
   *
   * @param input the input text
   * @param actualTokenCount the actual token count from tokenizer
   * @return validation result
   */
  public ValidationResult validateWithTokenCount(String input, int actualTokenCount) {
    // First do basic validation (except token estimation)
    ValidationResult basicValidation = validate(input);
    if (!basicValidation.isValid()) {
      return basicValidation;
    }

    // Check actual token count
    if (actualTokenCount > maxInputTokens) {
      List<String> errors = new ArrayList<>(basicValidation.getErrors());
      errors.add(String.format("Input exceeds maximum token limit: %d > %d tokens",
          actualTokenCount, maxInputTokens));
      return ValidationResult.invalid(errors, basicValidation.getWarnings());
    }

    return basicValidation;
  }

  /**
   * Estimates token count from character count.
   * Uses conservative ratio of ~0.3 tokens per character (3.3 characters per token).
   *
   * @param text the input text
   * @return estimated token count
   */
  public int estimateTokenCount(String text) {
    if (text == null || text.isEmpty()) {
      return 0;
    }

    // Use character count with conservative ratio
    // English text: ~4 chars/token, but we're conservative at ~3.3
    return (int) Math.ceil(text.length() * TOKENS_PER_CHAR);
  }

  /**
   * UTF-8 validation result with byte offset information.
   */
  private static class Utf8ValidationResult {
    final boolean valid;
    final int byteOffset;
    final int charPosition;

    Utf8ValidationResult(boolean valid, int byteOffset, int charPosition) {
      this.valid = valid;
      this.byteOffset = byteOffset;
      this.charPosition = charPosition;
    }

    boolean isValid() {
      return valid;
    }
  }

  /**
   * Validates UTF-8 encoding with detailed error location.
   *
   * @param text the text to validate
   * @return validation result with byte offset if invalid
   */
  private Utf8ValidationResult validateUtf8Detailed(String text) {
    try {
      // Encode to UTF-8 bytes
      byte[] bytes = text.getBytes(StandardCharsets.UTF_8);

      // Try to decode back, checking for replacement characters
      String decoded = new String(bytes, StandardCharsets.UTF_8);

      // Check for replacement character (U+FFFD) which indicates encoding issues
      int replacementIndex = decoded.indexOf('\uFFFD');
      if (replacementIndex >= 0 && !text.contains("\uFFFD")) {
        // Find approximate byte offset
        int byteOffset = 0;
        for (int i = 0; i < Math.min(replacementIndex, text.length()); i++) {
          byteOffset += text.substring(i, i + 1).getBytes(StandardCharsets.UTF_8).length;
        }
        return new Utf8ValidationResult(false, byteOffset, replacementIndex);
      }

      return new Utf8ValidationResult(true, -1, -1);
    } catch (Exception e) {
      return new Utf8ValidationResult(false, -1, -1);
    }
  }

  /**
   * Validates UTF-8 encoding of a string (legacy method).
   *
   * @param text the text to validate
   * @return true if valid UTF-8, false otherwise
   */
  private boolean isValidUtf8(String text) {
    return validateUtf8Detailed(text).isValid();
  }

  /**
   * Checks if text contains control characters.
   *
   * @param text the text to check
   * @return true if control characters are present
   */
  public boolean hasControlCharacters(String text) {
    return CONTROL_CHARS.matcher(text).find();
  }

  /**
   * Gets the maximum allowed input tokens.
   *
   * @return max input tokens
   */
  public int getMaxInputTokens() {
    return maxInputTokens;
  }

  /**
   * Gets the maximum allowed characters.
   *
   * @return max characters
   */
  public int getMaxCharacters() {
    return maxCharacters;
  }
}
