package com.badu.ai.onnx.validation;

import lombok.Builder;
import lombok.Value;

import java.util.List;

/**
 * Immutable data class capturing validation findings from input/output validation.
 *
 * <p>Validation results include:
 * <ul>
 *   <li><b>valid</b>: Whether validation passed (no blocking errors)</li>
 *   <li><b>errors</b>: Blocking errors that prevent inference (empty if valid=true)</li>
 *   <li><b>warnings</b>: Non-blocking warnings that don't prevent inference</li>
 * </ul>
 *
 * <p>This class is immutable and thread-safe.
 *
 * @see com.badu.ai.onnx.validation.InputValidator
 * @see com.badu.ai.onnx.validation.OutputValidator
 */
@Value
@Builder
public class ValidationResult {
    /**
     * Whether validation passed (no blocking errors).
     * Can be true even with warnings present.
     */
    boolean valid;

    /**
     * List of blocking errors that prevent inference.
     * Empty if valid=true.
     *
     * <p>Example errors:
     * <ul>
     *   <li>"Invalid UTF-8 encoding at byte offset 1234"</li>
     *   <li>"Input exceeds maximum length: 45000 tokens > 40960 max"</li>
     *   <li>"Control character detected at position 567 (U+0000)"</li>
     * </ul>
     */
    List<String> errors;

    /**
     * List of non-blocking warnings that don't prevent inference.
     * Can be present even when valid=true.
     *
     * <p>Example warnings:
     * <ul>
     *   <li>"Potential token ID leakage: 5+ consecutive digits detected"</li>
     *   <li>"Output readability low: only 15% letters"</li>
     *   <li>"Excessive whitespace detected (10+ consecutive spaces)"</li>
     * </ul>
     */
    List<String> warnings;

    /**
     * Creates a valid validation result with no errors or warnings.
     *
     * @return A valid ValidationResult
     */
    public static ValidationResult valid() {
        return ValidationResult.builder()
                .valid(true)
                .errors(List.of())
                .warnings(List.of())
                .build();
    }

    /**
     * Creates a valid validation result with warnings.
     *
     * @param warnings Non-blocking warnings
     * @return A valid ValidationResult with warnings
     */
    public static ValidationResult validWithWarnings(List<String> warnings) {
        return ValidationResult.builder()
                .valid(true)
                .errors(List.of())
                .warnings(warnings)
                .build();
    }

    /**
     * Creates an invalid validation result with errors.
     *
     * @param errors Blocking errors
     * @return An invalid ValidationResult
     */
    public static ValidationResult invalid(List<String> errors) {
        return ValidationResult.builder()
                .valid(false)
                .errors(errors)
                .warnings(List.of())
                .build();
    }

    /**
     * Creates an invalid validation result with both errors and warnings.
     *
     * @param errors Blocking errors
     * @param warnings Non-blocking warnings
     * @return An invalid ValidationResult
     */
    public static ValidationResult invalid(List<String> errors, List<String> warnings) {
        return ValidationResult.builder()
                .valid(false)
                .errors(errors)
                .warnings(warnings)
                .build();
    }
}
