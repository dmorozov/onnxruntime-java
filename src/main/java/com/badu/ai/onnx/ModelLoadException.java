package com.badu.ai.onnx;

import com.badu.ai.onnx.config.ModelVariant;

import java.nio.file.Path;

/**
 * Exception thrown when model loading fails with actionable error messages.
 *
 * <p>This exception provides detailed context about model loading failures including:
 * <ul>
 *   <li>File path that failed to load</li>
 *   <li>Expected file location</li>
 *   <li>Model variant being loaded</li>
 *   <li>Corrective actions to fix the issue</li>
 * </ul>
 *
 * <p><b>Common Scenarios:</b>
 * <ul>
 *   <li>Missing model files</li>
 *   <li>Missing tokenizer.json</li>
 *   <li>Invalid model format</li>
 *   <li>Insufficient permissions</li>
 *   <li>Corrupted model files</li>
 * </ul>
 *
 * <p>Usage example:
 * <pre>{@code
 * try {
 *   OnnxInference inference = OnnxInference.create(config, genConfig);
 * } catch (ModelLoadException e) {
 *   System.err.println("Failed to load model: " + e.getMessage());
 *   System.err.println("File path: " + e.getFilePath());
 *   System.err.println("Corrective action: " + e.getCorrectiveAction());
 * }
 * }</pre>
 *
 * @see com.badu.ai.onnx.config.ModelConfig
 * @see com.badu.ai.onnx.genai.internal.ModelLoader
 */
public class ModelLoadException extends RuntimeException {

  private final Path filePath;
  private final Path expectedLocation;
  private final ModelVariant modelVariant;
  private final String correctiveAction;
  private final FileType fileType;

  /**
   * Types of files that can fail to load.
   */
  public enum FileType {
    /** ONNX encoder model file */
    ENCODER_MODEL,

    /** ONNX decoder model file */
    DECODER_MODEL,

    /** Tokenizer configuration file (tokenizer.json) */
    TOKENIZER,

    /** Model data file (model.onnx_data for FULL variant) */
    MODEL_DATA,

    /** Unknown file type */
    UNKNOWN
  }

  /**
   * Creates a ModelLoadException with full context.
   *
   * @param message Error message
   * @param fileType Type of file that failed
   * @param filePath Path to the file that failed
   * @param expectedLocation Expected location of the file
   * @param modelVariant Model variant being loaded
   * @param correctiveAction Suggested fix
   * @param cause Root cause exception
   */
  public ModelLoadException(String message, FileType fileType, Path filePath,
                             Path expectedLocation, ModelVariant modelVariant,
                             String correctiveAction, Throwable cause) {
    super(buildFullMessage(message, fileType, filePath, expectedLocation, modelVariant,
        correctiveAction), cause);
    this.fileType = fileType;
    this.filePath = filePath;
    this.expectedLocation = expectedLocation;
    this.modelVariant = modelVariant;
    this.correctiveAction = correctiveAction;
  }

  /**
   * Creates a ModelLoadException for missing file.
   *
   * @param fileType Type of file missing
   * @param filePath Path where file was expected
   * @param modelVariant Model variant
   * @return ModelLoadException with helpful message
   */
  public static ModelLoadException missingFile(FileType fileType, Path filePath,
                                                 ModelVariant modelVariant) {
    String fileName = filePath.getFileName().toString();
    String correctiveAction;

    switch (fileType) {
      case ENCODER_MODEL:
        correctiveAction = String.format(
            "Ensure encoder model file '%s' exists in model directory. " +
            "For %s variant, download or export the encoder model with appropriate quantization.",
            fileName, modelVariant);
        break;

      case DECODER_MODEL:
        correctiveAction = String.format(
            "Ensure decoder model file '%s' exists in model directory. " +
            "For %s variant, download or export the decoder model with appropriate quantization.",
            fileName, modelVariant);
        break;

      case TOKENIZER:
        correctiveAction =
            "Ensure tokenizer.json file exists in model directory. " +
            "Download from Hugging Face or use ModelConfig.tokenizerPath() to specify custom location. " +
            "Example: ModelConfig.builder().tokenizerPath(\"/path/to/tokenizer.json\").build()";
        break;

      case MODEL_DATA:
        correctiveAction =
            "FULL variant requires external data file (model.onnx_data). " +
            "Ensure both model.onnx and model.onnx_data are in the model directory. " +
            "These files must be from the same export.";
        break;

      default:
        correctiveAction = "Verify file exists and has correct permissions.";
    }

    return new ModelLoadException(
        fileType + " file not found: " + fileName,
        fileType,
        filePath,
        filePath.getParent(),
        modelVariant,
        correctiveAction,
        null
    );
  }

  /**
   * Creates a ModelLoadException for invalid file format.
   *
   * @param fileType Type of file with invalid format
   * @param filePath Path to the invalid file
   * @param modelVariant Model variant
   * @param cause Root cause exception
   * @return ModelLoadException with helpful message
   */
  public static ModelLoadException invalidFormat(FileType fileType, Path filePath,
                                                   ModelVariant modelVariant, Throwable cause) {
    String correctiveAction;

    if (fileType == FileType.TOKENIZER) {
      correctiveAction =
          "Tokenizer file must be in HuggingFace tokenizers JSON format. " +
          "Download tokenizer.json from the model repository on Hugging Face. " +
          "Ensure the file is not corrupted.";
    } else {
      correctiveAction =
          "Model file must be valid ONNX format. " +
          "Re-export the model using optimum or onnxruntime tools. " +
          "Verify the file is not corrupted (check file size and hash).";
    }

    return new ModelLoadException(
        fileType + " file has invalid format: " + filePath.getFileName(),
        fileType,
        filePath,
        null,
        modelVariant,
        correctiveAction,
        cause
    );
  }

  /**
   * Creates a ModelLoadException for ONNX Runtime errors.
   *
   * @param fileType Type of model file
   * @param filePath Path to the model
   * @param modelVariant Model variant
   * @param ortException ONNX Runtime exception
   * @return ModelLoadException with helpful message
   */
  public static ModelLoadException onnxRuntimeError(FileType fileType, Path filePath,
                                                      ModelVariant modelVariant,
                                                      Throwable ortException) {
    String correctiveAction =
        "ONNX Runtime failed to load model. Common causes:\n" +
        "  1. Model opset version incompatible with ONNX Runtime 1.23.2\n" +
        "  2. Model file corrupted (verify file integrity)\n" +
        "  3. Missing external data file (for FULL variant)\n" +
        "  4. Unsupported operators in the model\n" +
        "Re-export model with compatible opset version: optimum-cli export onnx --opset 17";

    return new ModelLoadException(
        "ONNX Runtime error loading " + fileType + ": " + ortException.getMessage(),
        fileType,
        filePath,
        null,
        modelVariant,
        correctiveAction,
        ortException
    );
  }

  /**
   * Builds the full error message with all context.
   */
  private static String buildFullMessage(String message, FileType fileType, Path filePath,
                                          Path expectedLocation, ModelVariant modelVariant,
                                          String correctiveAction) {
    StringBuilder sb = new StringBuilder();
    sb.append("[MODEL_LOAD_ERROR] ").append(message);

    if (fileType != null) {
      sb.append("\n  File type: ").append(fileType);
    }

    if (filePath != null) {
      sb.append("\n  File path: ").append(filePath);
    }

    if (expectedLocation != null) {
      sb.append("\n  Expected location: ").append(expectedLocation);
    }

    if (modelVariant != null) {
      sb.append("\n  Model variant: ").append(modelVariant);
    }

    if (correctiveAction != null) {
      sb.append("\n  Corrective action: ").append(correctiveAction);
    }

    return sb.toString();
  }

  /**
   * Gets the file type that failed.
   *
   * @return file type
   */
  public FileType getFileType() {
    return fileType;
  }

  /**
   * Gets the file path that failed.
   *
   * @return file path (may be null)
   */
  public Path getFilePath() {
    return filePath;
  }

  /**
   * Gets the expected file location.
   *
   * @return expected location (may be null)
   */
  public Path getExpectedLocation() {
    return expectedLocation;
  }

  /**
   * Gets the model variant being loaded.
   *
   * @return model variant (may be null)
   */
  public ModelVariant getModelVariant() {
    return modelVariant;
  }

  /**
   * Gets the suggested corrective action.
   *
   * @return corrective action (may be null)
   */
  public String getCorrectiveAction() {
    return correctiveAction;
  }
}
