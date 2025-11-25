package com.badu.ai.onnx.genai;

import java.nio.file.Path;
import java.nio.file.Paths;
import com.badu.ai.onnx.config.ModelConfig;
import com.badu.ai.onnx.genai.internal.GeneratorParams;
import com.badu.ai.onnx.genai.internal.SimpleGenAI;

/**
 * Wrapper for ONNX Runtime GenAI SimpleGenAI with lifecycle management. SimpleGenAI is a high-level
 * wrapper that manages Model, Tokenizer, and Generator internally.
 * <p>
 * Usage:
 * 
 * <pre>{@code
 * try (ModelWrapper model = ModelWrapper.create(config)) {
 *   // Use model
 * }
 * }</pre>
 */
public class ModelWrapper implements AutoCloseable {

  private final SimpleGenAI simpleGenAI;
  private final String modelDir;

  private ModelWrapper(SimpleGenAI simpleGenAI, String modelDir) {
    this.simpleGenAI = simpleGenAI;
    this.modelDir = modelDir;
  }

  /**
   * Creates and loads GenAI SimpleGenAI from configuration.
   *
   * @param config model configuration
   * @return initialized ModelWrapper
   * @throws RuntimeException if model loading fails
   */
  public static ModelWrapper create(ModelConfig config) {
    try {
      // Construct full path to model directory
      Path modelPath = Paths.get(config.getModelPath());

      // Create SimpleGenAI - it will load model, tokenizer, and all necessary components
      // Note: SimpleGenAI auto-detects the model file (model.onnx, model_q4.onnx, etc.)
      SimpleGenAI genAI = new SimpleGenAI(modelPath.toString());

      return new ModelWrapper(genAI, config.getModelPath());
    } catch (Exception e) {
      throw new RuntimeException("Failed to load model from: " + config.getModelPath(), e);
    }
  }

  /**
   * Returns the underlying SimpleGenAI instance.
   *
   * @return SimpleGenAI (non-null if wrapper is open)
   */
  public SimpleGenAI getSimpleGenAI() {
    return simpleGenAI;
  }

  /**
   * Creates GeneratorParams from SimpleGenAI.
   *
   * @return new GeneratorParams instance
   * @throws RuntimeException if creation fails
   */
  public GeneratorParams createGeneratorParams() {
    try {
      return simpleGenAI.createGeneratorParams();
    } catch (Exception e) {
      throw new RuntimeException("Failed to create GeneratorParams", e);
    }
  }

  /**
   * Returns the model directory path.
   *
   * @return model directory
   */
  public String getModelDir() {
    return modelDir;
  }

  /**
   * Releases SimpleGenAI resources. Safe to call multiple times (idempotent).
   */
  @Override
  public void close() {
    if (simpleGenAI != null) {
      try {
        simpleGenAI.close();
      } catch (Exception e) {
        // Log but don't throw on cleanup
        System.err.println("Warning: Error closing SimpleGenAI: " + e.getMessage());
      }
    }
  }
}
