package com.badu.ai.onnx.validation;

/**
 * Exception thrown when validation fails.
 *
 * <p>Contains the validation result with detailed error messages.
 * This exception is thrown when input or output validation encounters
 * blocking errors that prevent inference from proceeding.
 *
 * @see ValidationResult
 * @see InputValidator
 * @see OutputValidator
 */
public class ValidationException extends Exception {

  private final ValidationResult validationResult;

  /**
   * Constructs a ValidationException with the given validation result.
   *
   * @param validationResult the validation result containing errors
   */
  public ValidationException(ValidationResult validationResult) {
    super(buildMessage(validationResult));
    this.validationResult = validationResult;
  }

  /**
   * Constructs a ValidationException with a simple message.
   *
   * @param message the error message
   */
  public ValidationException(String message) {
    super(message);
    this.validationResult = ValidationResult.invalid(java.util.List.of(message));
  }

  /**
   * Gets the validation result containing detailed error information.
   *
   * @return the validation result
   */
  public ValidationResult getValidationResult() {
    return validationResult;
  }

  /**
   * Builds a detailed error message from the validation result.
   *
   * @param result the validation result
   * @return formatted error message
   */
  private static String buildMessage(ValidationResult result) {
    if (result == null) {
      return "Validation failed";
    }

    StringBuilder message = new StringBuilder("Validation failed");

    if (!result.getErrors().isEmpty()) {
      message.append(" with ").append(result.getErrors().size()).append(" error(s):\n");
      for (int i = 0; i < result.getErrors().size(); i++) {
        message.append("  ").append(i + 1).append(". ").append(result.getErrors().get(i));
        if (i < result.getErrors().size() - 1) {
          message.append("\n");
        }
      }
    }

    if (!result.getWarnings().isEmpty()) {
      if (!result.getErrors().isEmpty()) {
        message.append("\n");
      }
      message.append("Warnings:\n");
      for (int i = 0; i < result.getWarnings().size(); i++) {
        message.append("  - ").append(result.getWarnings().get(i));
        if (i < result.getWarnings().size() - 1) {
          message.append("\n");
        }
      }
    }

    return message.toString();
  }
}
