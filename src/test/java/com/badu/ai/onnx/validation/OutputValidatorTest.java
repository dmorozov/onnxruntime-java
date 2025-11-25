package com.badu.ai.onnx.validation;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for OutputValidator.
 * Tests validation of generated output including token ID leakage detection,
 * UTF-8 encoding, control characters, and human readability checks.
 */
class OutputValidatorTest {

    private OutputValidator validator;

    @BeforeEach
    void setUp() {
        // Default validator: 1 char min, 100000 max, non-strict mode
        validator = new OutputValidator();
    }

    @Test
    @DisplayName("Valid output passes all checks")
    void validOutput_passesAllChecks() {
        String validOutput = "Artificial intelligence is the simulation of human intelligence.";
        ValidationResult result = validator.validate(validOutput);

        assertTrue(result.isValid());
        assertTrue(result.getWarnings().isEmpty());
        assertTrue(result.getErrors().isEmpty());
    }

    @Test
    @DisplayName("Null output is invalid")
    void nullOutput_isInvalid() {
        ValidationResult result = validator.validate(null);

        assertFalse(result.isValid());
        assertFalse(result.getErrors().isEmpty());
        assertTrue(result.getErrors().get(0).toLowerCase().contains("null"));
    }

    @Test
    @DisplayName("Empty output is invalid")
    void emptyOutput_isInvalid() {
        ValidationResult result = validator.validate("");

        assertFalse(result.isValid());
        assertFalse(result.getErrors().isEmpty());
        assertTrue(result.getErrors().get(0).toLowerCase().contains("empty"));
    }

    @Test
    @DisplayName("Whitespace-only output is invalid")
    void whitespaceOnlyOutput_isInvalid() {
        ValidationResult result = validator.validate("   \n\t\r  ");

        assertFalse(result.isValid());
        assertFalse(result.getErrors().isEmpty());
        assertTrue(result.getErrors().get(0).toLowerCase().contains("whitespace") ||
                   result.getErrors().get(0).toLowerCase().contains("empty"));
    }

    @Test
    @DisplayName("Output below minimum length is invalid")
    void outputBelowMinLength_isInvalid() {
        OutputValidator strictValidator = new OutputValidator(10, 100000, false);
        String shortOutput = "Hi";  // Only 2 chars, below min of 10

        ValidationResult result = strictValidator.validate(shortOutput);

        assertFalse(result.isValid());
        assertFalse(result.getErrors().isEmpty());
        assertTrue(result.getErrors().get(0).contains("too short") ||
                   result.getErrors().get(0).contains("short"));
    }

    @Test
    @DisplayName("Output exceeding maximum length is invalid")
    void outputExceedingMaxLength_isInvalid() {
        OutputValidator strictValidator = new OutputValidator(1, 100, false);
        String longOutput = "a".repeat(101);

        ValidationResult result = strictValidator.validate(longOutput);

        assertFalse(result.isValid());
        assertFalse(result.getErrors().isEmpty());
        assertTrue(result.getErrors().get(0).contains("too long") ||
                   result.getErrors().get(0).contains("long"));
    }

    @Test
    @DisplayName("Output with potential token ID leakage generates warning")
    void outputWithTokenIds_generatesWarning() {
        String outputWithIds = "The result is 12345 tokens were generated 67890.";
        ValidationResult result = validator.validate(outputWithIds);

        assertTrue(result.isValid());
        assertFalse(result.getWarnings().isEmpty());
        assertTrue(result.getWarnings().get(0).toLowerCase().contains("digit") ||
                   result.getWarnings().get(0).toLowerCase().contains("token"));
    }

    @Test
    @DisplayName("Output with token IDs fails in strict mode")
    void outputWithTokenIds_failsInStrictMode() {
        OutputValidator strictValidator = new OutputValidator(1, 100000, true);
        String outputWithIds = "The result is 12345 and 67890.";

        ValidationResult result = strictValidator.validate(outputWithIds);

        assertFalse(result.isValid());
        assertFalse(result.getErrors().isEmpty());
        assertTrue(result.getErrors().get(0).toLowerCase().contains("digit") ||
                   result.getErrors().get(0).toLowerCase().contains("token"));
    }

    @Test
    @DisplayName("Output with normal numbers is valid")
    void outputWithNormalNumbers_isValid() {
        String output = "The answer is 42 and the year is 2024.";
        ValidationResult result = validator.validate(output);

        assertTrue(result.isValid());
        assertTrue(result.getWarnings().isEmpty());  // 4 digits or less shouldn't trigger warning
    }

    @Test
    @DisplayName("Output with control characters generates warning")
    void outputWithControlChars_generatesWarning() {
        String outputWithControlChar = "Hello\u0000World";
        ValidationResult result = validator.validate(outputWithControlChar);

        assertTrue(result.isValid());
        assertFalse(result.getWarnings().isEmpty());
        assertTrue(result.getWarnings().get(0).toLowerCase().contains("control"));
    }

    @Test
    @DisplayName("Output with control characters fails in strict mode")
    void outputWithControlChars_failsInStrictMode() {
        OutputValidator strictValidator = new OutputValidator(1, 100000, true);
        String outputWithControlChar = "Hello\u0007World";

        ValidationResult result = strictValidator.validate(outputWithControlChar);

        assertFalse(result.isValid());
        assertFalse(result.getErrors().isEmpty());
        assertTrue(result.getErrors().get(0).toLowerCase().contains("control"));
    }

    @Test
    @DisplayName("Output with newlines and tabs is valid")
    void outputWithNewlinesAndTabs_isValid() {
        String output = "First line\nSecond line\tWith tab";
        ValidationResult result = validator.validate(output);

        assertTrue(result.isValid());
        assertTrue(result.getWarnings().isEmpty());
    }

    @Test
    @DisplayName("Output with excessive whitespace generates warning")
    void outputWithExcessiveWhitespace_generatesWarning() {
        String output = "Hello          World";  // 10+ spaces
        ValidationResult result = validator.validate(output);

        assertTrue(result.isValid());
        assertFalse(result.getWarnings().isEmpty());
        assertTrue(result.getWarnings().get(0).toLowerCase().contains("whitespace"));
    }

    @Test
    @DisplayName("Unicode and emoji output is valid")
    void unicodeAndEmojiOutput_isValid() {
        String unicodeOutput = "Hello 世界 🌍 مرحبا!";
        ValidationResult result = validator.validate(unicodeOutput);

        assertTrue(result.isValid());
    }

    @Test
    @DisplayName("Human-readable check for normal text returns true")
    void humanReadableCheck_normalText_returnsTrue() {
        String normalText = "This is a normal sentence with words.";
        assertTrue(validator.isHumanReadable(normalText));
    }

    @Test
    @DisplayName("Human-readable check for mostly digits returns false")
    void humanReadableCheck_mostlyDigits_returnsFalse() {
        String digitString = "123456789012345678901234567890";
        assertFalse(validator.isHumanReadable(digitString));
    }

    @Test
    @DisplayName("Human-readable check for mixed content returns true")
    void humanReadableCheck_mixedContent_returnsTrue() {
        String mixedContent = "The answer is 42 and the year is 2024.";
        assertTrue(validator.isHumanReadable(mixedContent));
    }

    @Test
    @DisplayName("Human-readable check for empty returns false")
    void humanReadableCheck_empty_returnsFalse() {
        assertFalse(validator.isHumanReadable(""));
        assertFalse(validator.isHumanReadable(null));
    }

    @Test
    @DisplayName("Human-readable check for pure symbols returns false")
    void humanReadableCheck_pureSymbols_returnsFalse() {
        String symbols = "!@#$%^&*()_+-=[]{}|;:,.<>?";
        assertFalse(validator.isHumanReadable(symbols));
    }

    @Test
    @DisplayName("Validation result toString provides useful info")
    void validationResultToString_providesUsefulInfo() {
        String output = "Test output";
        ValidationResult result = validator.validate(output);

        String toString = result.toString();
        assertTrue(toString.contains("valid=true"));
    }

    @Test
    @DisplayName("Invalid result toString shows error")
    void invalidResultToString_showsError() {
        ValidationResult result = validator.validate(null);

        String toString = result.toString();
        assertTrue(toString.contains("valid=false"));
        assertTrue(toString.contains("errors="));
    }

    @Test
    @DisplayName("Warning result toString includes warning message")
    void warningResultToString_includesWarningMessage() {
        String outputWithWarning = "Output with 12345 digits";
        ValidationResult result = validator.validate(outputWithWarning);

        String toString = result.toString();
        assertTrue(toString.contains("warnings="));
    }

    @Test
    @DisplayName("Output with mixed good and bad patterns gets warning")
    void outputWithMixedPatterns_getsWarning() {
        // Contains token ID pattern but also readable text
        String output = "The model generated 123456 tokens successfully.";
        ValidationResult result = validator.validate(output);

        assertTrue(result.isValid());
        assertFalse(result.getWarnings().isEmpty());
    }

    @Test
    @DisplayName("Clean technical output without issues passes")
    void cleanTechnicalOutput_passes() {
        String technicalOutput = "The computation completed with result: 3.14 meters";
        ValidationResult result = validator.validate(technicalOutput);

        assertTrue(result.isValid());
        assertTrue(result.getWarnings().isEmpty());
    }

    @Test
    @DisplayName("Output with exactly 5 digits triggers warning")
    void outputWithExactly5Digits_triggersWarning() {
        String output = "Code: 12345";
        ValidationResult result = validator.validate(output);

        assertTrue(result.isValid());
        assertFalse(result.getWarnings().isEmpty());
    }

    @Test
    @DisplayName("Output with 4 digits does not trigger warning")
    void outputWith4Digits_doesNotTriggerWarning() {
        String output = "Year: 2024";
        ValidationResult result = validator.validate(output);

        assertTrue(result.isValid());
        assertTrue(result.getWarnings().isEmpty());
    }
}
