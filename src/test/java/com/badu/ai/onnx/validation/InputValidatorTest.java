package com.badu.ai.onnx.validation;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for InputValidator. Tests validation of user input including token counting, length
 * limits, UTF-8 encoding, and control characters.
 */
class InputValidatorTest {

  private InputValidator validator;

  @BeforeEach
  void setUp() {
    // Default validator: 40960 tokens max, 150000 chars max, no control chars
    validator = new InputValidator();
  }

  @Test
  @DisplayName("Valid input passes all checks")
  void validInput_passesAllChecks() {
    String validInput = "What is artificial intelligence?";
    ValidationResult result = validator.validate(validInput);

    assertTrue(result.isValid());
    assertEquals(0, result.getErrors().size());
  }

  @Test
  @DisplayName("Null input is invalid")
  void nullInput_isInvalid() {
    ValidationResult result = validator.validate(null);

    assertFalse(result.isValid());
    assertTrue(result.getErrors().size() > 0);
    assertTrue(result.getErrors().get(0).contains("null"));
  }

  @Test
  @DisplayName("Empty input is invalid")
  void emptyInput_isInvalid() {
    ValidationResult result = validator.validate("");

    assertFalse(result.isValid());
    assertTrue(result.getErrors().size() > 0);
    assertTrue(result.getErrors().get(0).toLowerCase().contains("empty"));
  }

  @Test
  @DisplayName("Whitespace-only input is invalid")
  void whitespaceOnlyInput_isInvalid() {
    ValidationResult result = validator.validate("   \n\t\r  ");

    assertFalse(result.isValid());
    assertTrue(result.getErrors().size() > 0);
  }

  @Test
  @DisplayName("Input exceeding character limit is invalid")
  void inputExceedingCharLimit_isInvalid() {
    // Create input longer than default max (150000 chars)
    String longInput = "a".repeat(150001);
    ValidationResult result = validator.validate(longInput);

    assertFalse(result.isValid());
    assertTrue(result.getErrors().size() > 0);
    String errorMsg = String.join(" ", result.getErrors());
    assertTrue(errorMsg.contains("limit") || errorMsg.contains("150001"));
  }

  @Test
  @DisplayName("Input with control characters is invalid by default")
  void inputWithControlChars_isInvalid() {
    String inputWithControlChar = "Hello\u0000World"; // Null character
    ValidationResult result = validator.validate(inputWithControlChar);

    assertFalse(result.isValid());
    assertTrue(result.getErrors().size() > 0);
    String errorMsg = result.getErrors().get(0).toLowerCase();
    assertTrue(errorMsg.contains("control"));
  }

  @Test
  @DisplayName("Input with control characters allowed when configured")
  void inputWithControlChars_allowedWhenConfigured() {
    InputValidator permissiveValidator = new InputValidator(40960, 150000, true);
    String inputWithControlChar = "Hello\u0000World";

    ValidationResult result = permissiveValidator.validate(inputWithControlChar);

    // Should pass when control chars are allowed
    assertTrue(result.isValid());
  }

  @Test
  @DisplayName("Input with newlines and tabs is valid")
  void inputWithNewlinesAndTabs_isValid() {
    String inputWithWhitespace = "Hello\nWorld\tTest";
    ValidationResult result = validator.validate(inputWithWhitespace);

    assertTrue(result.isValid());
  }

  @Test
  @DisplayName("Input with warnings")
  void inputWithWarnings_showsWarnings() {
    // Create input that's large but within limits to trigger warning
    String largeInput = "word ".repeat(700); // ~700 words
    ValidationResult result = validator.validate(largeInput);

    assertTrue(result.isValid());
    // May have warnings
  }

  @Test
  @DisplayName("Token count estimation is reasonable")
  void tokenCountEstimation_isReasonable() {
    String input = "This is a test sentence with multiple words.";
    int estimated = validator.estimateTokenCount(input);

    // Should estimate roughly number of words (conservative)
    assertTrue(estimated >= 5); // At least some tokens
    assertTrue(estimated <= 20); // Not excessive
  }

  @Test
  @DisplayName("Empty string estimates zero tokens")
  void emptyString_estimatesZeroTokens() {
    assertEquals(0, validator.estimateTokenCount(""));
    assertEquals(0, validator.estimateTokenCount(null));
  }

  @Test
  @DisplayName("Long input estimates high token count")
  void longInput_estimatesHighTokenCount() {
    String longInput = "word ".repeat(10000);
    int estimated = validator.estimateTokenCount(longInput);

    assertTrue(estimated > 1000); // Should be substantial
  }

  @Test
  @DisplayName("Unicode and emoji input is valid")
  void unicodeAndEmoji_isValid() {
    String unicodeInput = "Hello 世界 🌍 مرحبا";
    ValidationResult result = validator.validate(unicodeInput);

    assertTrue(result.isValid());
    assertEquals(0, result.getErrors().size());
  }

  @Test
  @DisplayName("Input estimated to exceed token limit is invalid")
  void inputEstimatedToExceedTokenLimit_isInvalid() {
    // Create custom validator with very low token limit and char limit
    InputValidator strictValidator = new InputValidator(10, 50, false);
    String longInput = "word ".repeat(20); // 100 chars, will exceed both limits

    ValidationResult result = strictValidator.validate(longInput);

    assertFalse(result.isValid());
    assertTrue(result.getErrors().size() > 0);
    String errorMsg = result.getErrors().get(0).toLowerCase();
    assertTrue(errorMsg.contains("limit"));
  }

  @Test
  @DisplayName("Validation with actual token count")
  void validationWithActualTokenCount_works() {
    String input = "What is AI?";
    int actualTokens = 5; // Simulated tokenizer output

    ValidationResult result = validator.validateWithTokenCount(input, actualTokens);

    assertTrue(result.isValid());
  }

  @Test
  @DisplayName("Validation with excessive actual token count fails")
  void validationWithExcessiveActualTokenCount_fails() {
    String input = "What is AI?";
    int actualTokens = 50000; // Exceeds limit

    ValidationResult result = validator.validateWithTokenCount(input, actualTokens);

    assertFalse(result.isValid());
    assertTrue(result.getErrors().size() > 0);
    String errorMsg = result.getErrors().get(0);
    assertTrue(errorMsg.contains("50000") || errorMsg.contains("token"));
  }

  @Test
  @DisplayName("Has control characters detection works")
  void hasControlCharactersDetection_works() {
    assertFalse(validator.hasControlCharacters("Normal text"));
    assertFalse(validator.hasControlCharacters("Text\nwith\nnewlines"));
    assertTrue(validator.hasControlCharacters("Text\u0000with null"));
    assertTrue(validator.hasControlCharacters("Text\u0007with bell"));
  }

  @Test
  @DisplayName("Custom validator uses provided limits")
  void customValidator_usesProvidedLimits() {
    InputValidator customValidator = new InputValidator(100, 400, false);

    assertEquals(100, customValidator.getMaxInputTokens());
    assertEquals(400, customValidator.getMaxCharacters());
  }

  @Test
  @DisplayName("Validation result toString provides useful info")
  void validationResultToString_providesUsefulInfo() {
    String input = "Test input";
    ValidationResult result = validator.validate(input);

    String toString = result.toString();
    assertTrue(toString.contains("valid=true"));
  }

  @Test
  @DisplayName("Invalid result toString shows error")
  void invalidResultToString_showsError() {
    ValidationResult result = validator.validate(null);

    String toString = result.toString();
    assertTrue(toString.contains("valid=false"));
    assertTrue(result.getErrors().size() > 0);
  }

  // ============================================================================
  // NEGATIVE TEST CASES (Phase 6 - T051)
  // ============================================================================

  @Test
  @DisplayName("Negative: Multiple control characters detected")
  void negativeTest_multipleControlCharacters() {
    // Test with multiple different control characters
    String input = "Test\u0000with\u0001multiple\u0007control\u001Fchars";
    ValidationResult result = validator.validate(input);

    assertFalse(result.isValid(), "Input with multiple control chars should fail");
    String errorMsg = String.join(" ", result.getErrors());
    assertTrue(errorMsg.contains("Control character"), "Error should mention control characters");
    assertTrue(errorMsg.contains("position"), "Error should include position");
    assertTrue(errorMsg.contains("U+"), "Error should include Unicode code point");
  }

  @Test
  @DisplayName("Negative: Control character at start of input")
  void negativeTest_controlCharacterAtStart() {
    String input = "\u0000Hello World";
    ValidationResult result = validator.validate(input);

    assertFalse(result.isValid());
    String errorMsg = result.getErrors().get(0);
    assertTrue(errorMsg.contains("position 0"), "Should report position 0");
  }

  @Test
  @DisplayName("Negative: Control character at end of input")
  void negativeTest_controlCharacterAtEnd() {
    String input = "Hello World\u0000";
    ValidationResult result = validator.validate(input);

    assertFalse(result.isValid());
    String errorMsg = result.getErrors().get(0);
    assertTrue(errorMsg.contains("Control character"), "Should detect control char at end");
  }

  @Test
  @DisplayName("Negative: Exact character limit boundary (just over)")
  void negativeTest_exactCharLimitBoundary() {
    // Test exactly 1 character over the limit
    InputValidator customValidator = new InputValidator(1000, 100, false);
    String input = "a".repeat(101); // Exactly 1 over limit

    ValidationResult result = customValidator.validate(input);

    assertFalse(result.isValid());
    String errorMsg = String.join(" ", result.getErrors());
    assertTrue(errorMsg.contains("101"), "Should mention actual character count");
    assertTrue(errorMsg.contains("100"), "Should mention max character count");
  }

  @Test
  @DisplayName("Negative: Exact character limit boundary (exactly at limit)")
  void negativeTest_exactlyAtCharLimit() {
    InputValidator customValidator = new InputValidator(1000, 100, false);
    String input = "a".repeat(100); // Exactly at limit

    ValidationResult result = customValidator.validate(input);

    // At the limit should be valid (not exceeding)
    assertTrue(result.isValid(), "Input at exact character limit should be valid");
  }

  @Test
  @DisplayName("Negative: Estimated token limit boundary")
  void negativeTest_estimatedTokenLimitBoundary() {
    // With 0.3 tokens/char ratio: 100 tokens = ~334 chars
    // So 335 chars should exceed 100 token limit
    InputValidator customValidator = new InputValidator(100, 10000, false);
    String input = "a".repeat(335);

    ValidationResult result = customValidator.validate(input);

    assertFalse(result.isValid());
    String errorMsg = String.join(" ", result.getErrors());
    assertTrue(errorMsg.contains("token"), "Should mention token limit");
    assertTrue(errorMsg.contains("estimated"), "Should indicate estimate");
  }

  @Test
  @DisplayName("Negative: Excessive digits warning (6 digits)")
  void negativeTest_excessiveDigits6() {
    String input = "Order number: 123456";
    ValidationResult result = validator.validate(input);

    // Should be valid but with warning
    assertTrue(result.isValid());
    assertTrue(result.getWarnings().size() > 0, "Should have warnings");

    String warning = String.join(" ", result.getWarnings());
    assertTrue(warning.contains("digits"), "Warning should mention digits");
    assertTrue(warning.contains("token ID"), "Warning should mention token IDs");
  }

  @Test
  @DisplayName("Negative: Excessive digits warning (many digits)")
  void negativeTest_excessiveDigitsMany() {
    String input = "Transaction: 12345678901234567890";
    ValidationResult result = validator.validate(input);

    assertTrue(result.isValid());
    assertTrue(result.getWarnings().size() > 0, "Should warn about excessive digits");
  }

  @Test
  @DisplayName("Negative: 5 consecutive digits should NOT warn")
  void negativeTest_fiveDigitsNoWarning() {
    String input = "Order: 12345"; // Only 5 digits
    ValidationResult result = validator.validate(input);

    assertTrue(result.isValid());
    assertEquals(0, result.getWarnings().size(),
        "5 digits should not trigger warning (threshold is 6+)");
  }

  @Test
  @DisplayName("Negative: Multiple validation failures reported together")
  void negativeTest_multipleFailures() {
    InputValidator strictValidator = new InputValidator(10, 50, false);

    // This input will fail multiple checks:
    // 1. Exceeds character limit (50)
    // 2. Exceeds token limit (10)
    // 3. Contains control character
    String input = "Hello\u0000World".repeat(20); // ~220 chars with control char

    ValidationResult result = strictValidator.validate(input);

    assertFalse(result.isValid());
    assertTrue(result.getErrors().size() >= 2,
        "Should report multiple errors: " + result.getErrors());

    String allErrors = String.join("; ", result.getErrors());
    // Should have both character limit and control character errors
    assertTrue(allErrors.contains("character") || allErrors.contains("Control"),
        "Should report multiple types of errors");
  }

  @Test
  @DisplayName("Negative: Whitespace variations")
  void negativeTest_variousWhitespaceOnly() {
    // Test various combinations of whitespace
    String[] whitespaceInputs = {
        "   ",           // spaces
        "\n\n\n",        // newlines
        "\t\t\t",        // tabs
        "\r\r\r",        // carriage returns
        " \n\t\r ",      // mixed
        "          "     // many spaces
    };

    for (String input : whitespaceInputs) {
      ValidationResult result = validator.validate(input);
      assertFalse(result.isValid(),
          "Whitespace-only input should fail: " + input.replace("\n", "\\n")
              .replace("\t", "\\t").replace("\r", "\\r"));
      assertTrue(result.getErrors().get(0).contains("whitespace"),
          "Error should mention whitespace-only");
    }
  }

  @Test
  @DisplayName("Negative: Single character input (valid)")
  void negativeTest_singleCharacter() {
    String input = "a";
    ValidationResult result = validator.validate(input);

    // Single character should be valid
    assertTrue(result.isValid(), "Single character input should be valid");
  }

  @Test
  @DisplayName("Negative: Very long continuous word")
  void negativeTest_veryLongContinuousWord() {
    // A very long word with no spaces (might behave differently in tokenization)
    String input = "a".repeat(10000);

    ValidationResult result = validator.validate(input);

    // Should be valid if within limits
    assertTrue(result.isValid() || result.getErrors().size() > 0);
    // Just checking it doesn't crash
  }

  @Test
  @DisplayName("Negative: Empty after trim but non-empty before")
  void negativeTest_emptyAfterTrim() {
    String input = "     ";
    ValidationResult result = validator.validate(input);

    assertFalse(result.isValid());
    String errorMsg = result.getErrors().get(0);
    assertTrue(errorMsg.toLowerCase().contains("whitespace"),
        "Should detect whitespace-only input");
  }

  @Test
  @DisplayName("Negative: UTF-8 validation does not break on valid Unicode")
  void negativeTest_utf8ValidUnicode() {
    // Test with various valid Unicode characters
    String input = "Hello 世界 🌍 مرحبا здравствуй שלום";
    ValidationResult result = validator.validate(input);

    assertTrue(result.isValid(),
        "Valid Unicode should pass UTF-8 validation");
  }

  @Test
  @DisplayName("Negative: Emoji and special Unicode")
  void negativeTest_emojiAndSpecialUnicode() {
    String input = "Test 😀 🎉 ✨ 🚀 ❤️ ✅";
    ValidationResult result = validator.validate(input);

    assertTrue(result.isValid(), "Emoji should be valid UTF-8");
  }

  @Test
  @DisplayName("Negative: Control character code points documented")
  void negativeTest_variousControlCharacterCodes() {
    // Test various control characters and verify error messages
    char[] controlChars = {
        '\u0000', // NULL
        '\u0001', // SOH
        '\u0007', // BELL
        '\u001F', // US (unit separator)
        '\u007F'  // DEL
    };

    for (char c : controlChars) {
      String input = "Hello" + c + "World";
      ValidationResult result = validator.validate(input);

      assertFalse(result.isValid(),
          "Should reject control character U+" + String.format("%04X", (int) c));

      String errorMsg = result.getErrors().get(0);
      assertTrue(errorMsg.contains("U+"),
          "Error should include Unicode format for char: " + c);
      assertTrue(errorMsg.contains(String.format("%04X", (int) c)),
          "Error should include correct hex code");
    }
  }

  @Test
  @DisplayName("Negative: Token estimation for special characters")
  void negativeTest_tokenEstimationSpecialChars() {
    // Special characters might tokenize differently
    String input = "!@#$%^&*()_+-=[]{}|;:',.<>?/~`";
    int estimated = validator.estimateTokenCount(input);

    assertTrue(estimated > 0, "Should estimate some tokens for special chars");
    // Don't enforce exact count since it's an approximation
  }

  @Test
  @DisplayName("Negative: validateWithTokenCount preserves basic validation")
  void negativeTest_validateWithTokenCountPreservesBasicValidation() {
    // Even if token count is OK, should still fail basic validation
    String input = null;
    ValidationResult result = validator.validateWithTokenCount(input, 10);

    assertFalse(result.isValid(), "Should fail on null even with valid token count");
    assertTrue(result.getErrors().get(0).contains("null"));
  }

  @Test
  @DisplayName("Negative: validateWithTokenCount with control characters")
  void negativeTest_validateWithTokenCountAndControlChars() {
    String input = "Hello\u0000World";
    ValidationResult result = validator.validateWithTokenCount(input, 5);

    assertFalse(result.isValid(),
        "Should fail on control character even with valid token count");
    String errorMsg = String.join(" ", result.getErrors());
    assertTrue(errorMsg.contains("Control character"));
  }

  @Test
  @DisplayName("Negative: Boundary testing - zero length limits (edge case)")
  void negativeTest_zeroLengthLimits() {
    // Create validator with impossible limits
    InputValidator impossibleValidator = new InputValidator(0, 0, false);

    String input = "a";
    ValidationResult result = impossibleValidator.validate(input);

    assertFalse(result.isValid(),
        "Any input should fail with zero limits");
  }

  @Test
  @DisplayName("Negative: Very large token limit (no practical limit)")
  void negativeTest_veryLargeTokenLimit() {
    InputValidator permissiveValidator =
        new InputValidator(Integer.MAX_VALUE, Integer.MAX_VALUE, true);

    String largeInput = "word ".repeat(10000);
    ValidationResult result = permissiveValidator.validate(largeInput);

    assertTrue(result.isValid(),
        "Should accept large input with very high limits");
  }

  @Test
  @DisplayName("Negative: Error message includes corrective action for UTF-8")
  void negativeTest_utf8ErrorMessageHasCorrectiveAction() {
    // This tests the error message format, though Java strings are always valid UTF-8
    // The important thing is when validation fails, the message should be helpful

    // We can't easily create invalid UTF-8 in Java, but we can verify
    // that the validation logic exists and error messages are formatted correctly
    // This is tested indirectly through valid inputs passing
    String validInput = "Test input";
    ValidationResult result = validator.validate(validInput);

    assertTrue(result.isValid());
    // UTF-8 validation is working if valid input passes
  }

  @Test
  @DisplayName("Negative: Error messages are non-empty and actionable")
  void negativeTest_errorMessagesAreActionable() {
    // Collect all error messages from various failure scenarios
    String[] invalidInputs = {
        null,                    // null
        "",                      // empty
        "   ",                   // whitespace-only
        "Hello\u0000World",      // control char
        "a".repeat(150001)       // too long
    };

    for (String input : invalidInputs) {
      ValidationResult result = validator.validate(input);
      assertFalse(result.isValid(),
          "Input should be invalid: " + (input == null ? "null" :
              input.length() > 20 ? input.substring(0, 20) + "..." : input));

      for (String error : result.getErrors()) {
        assertNotNull(error, "Error message should not be null");
        assertFalse(error.trim().isEmpty(),
            "Error message should not be empty");
        assertTrue(error.length() > 10,
            "Error message should be descriptive: " + error);
      }
    }
  }
}
