package com.badu.ai.onnx.genai.internal;

import java.util.HashMap;
import java.util.Map;

/**
 * Represents the parameters used for generating sequences with a model. Set the prompt using
 * setInputs, and any other search options using setSearchOption.
 */
public class GeneratorParams {

  private final Model model;
  private final Map<String, Object> searchOptions;
  private Sequences inputSequences;

  /**
   * Creates a GeneratorParams from the given model.
   *
   * @param model The model to use.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public GeneratorParams(Model model) throws GenAIException {
    if (model == null) {
      throw new GenAIException("Model cannot be null");
    }
    this.model = model;
    this.searchOptions = new HashMap<>();

    // Set default values for common options
    setSearchOption("max_length", 512.0);
    setSearchOption("min_length", 0.0);
    setSearchOption("do_sample", false);
    setSearchOption("temperature", 1.0);
    setSearchOption("top_k", 50.0);
    setSearchOption("top_p", 1.0);
    setSearchOption("repetition_penalty", 1.0);
  }

  /**
   * Set search option with boolean value.
   *
   * @param optionName The option name.
   * @param value The option value.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public void setSearchOption(String optionName, boolean value) throws GenAIException {
    if (optionName == null || optionName.trim().isEmpty()) {
      throw new GenAIException("Option name cannot be null or empty");
    }
    searchOptions.put(optionName, value);
  }

  /**
   * Set search option with double value.
   *
   * @param optionName The option name.
   * @param value The option value.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public void setSearchOption(String optionName, double value) throws GenAIException {
    if (optionName == null || optionName.trim().isEmpty()) {
      throw new GenAIException("Option name cannot be null or empty");
    }
    searchOptions.put(optionName, value);
  }

  /**
   * Set search option with integer value.
   *
   * @param optionName The option name.
   * @param value The option value.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public void setSearchOption(String optionName, int value) throws GenAIException {
    if (optionName == null || optionName.trim().isEmpty()) {
      throw new GenAIException("Option name cannot be null or empty");
    }
    searchOptions.put(optionName, value);
  }

  /**
   * Gets a search option value.
   *
   * @param optionName The option name.
   * @return The option value, or null if not set.
   */
  public Object getSearchOption(String optionName) {
    return searchOptions.get(optionName);
  }

  /**
   * Gets all search options.
   *
   * @return Map of all search options.
   */
  public Map<String, Object> getAllSearchOptions() {
    return new HashMap<>(searchOptions);
  }

  /**
   * Sets the input sequences for generation.
   *
   * @param sequences The input token sequences.
   */
  public void setInputSequences(Sequences sequences) {
    this.inputSequences = sequences;
  }

  /**
   * Gets the input sequences.
   *
   * @return The input token sequences.
   */
  public Sequences getInputSequences() {
    return inputSequences;
  }

  /**
   * Gets the model associated with these parameters.
   *
   * @return The model.
   */
  public Model getModel() {
    return model;
  }

  /**
   * Gets the maximum length option as an integer.
   *
   * @return Maximum generation length.
   */
  public int getMaxLength() {
    Object value = searchOptions.get("max_length");
    if (value instanceof Number) {
      return ((Number) value).intValue();
    }
    return 512; // Default
  }

  /**
   * Gets the temperature option as a double.
   *
   * @return Temperature value for sampling.
   */
  public double getTemperature() {
    Object value = searchOptions.get("temperature");
    if (value instanceof Number) {
      return ((Number) value).doubleValue();
    }
    return 1.0; // Default
  }

  /**
   * Gets whether sampling is enabled.
   *
   * @return True if sampling is enabled.
   */
  public boolean isDoSample() {
    Object value = searchOptions.get("do_sample");
    if (value instanceof Boolean) {
      return (Boolean) value;
    }
    return false; // Default
  }
}
