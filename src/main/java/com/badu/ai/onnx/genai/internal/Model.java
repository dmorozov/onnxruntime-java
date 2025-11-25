package com.badu.ai.onnx.genai.internal;

/**
 * Lightweight wrapper around model directory path.
 * Represents a GenAI model that can be used for inference.
 */
public class Model implements AutoCloseable {

  private final String modelDir;

  /**
   * Creates a Model from the given model directory.
   *
   * @param modelDir Path to the directory containing model files
   * @throws GenAIException If model directory is invalid
   */
  public Model(String modelDir) throws GenAIException {
    if (modelDir == null || modelDir.trim().isEmpty()) {
      throw new GenAIException("Model directory path cannot be null or empty");
    }
    this.modelDir = modelDir;
  }

  /**
   * Gets the model directory path.
   *
   * @return The model directory path
   */
  public String getModelDir() {
    return modelDir;
  }

  /**
   * Closes the model and releases any associated resources.
   * Currently a no-op as Model is just a lightweight wrapper.
   */
  @Override
  public void close() {
    // No resources to release at this level
  }
}
