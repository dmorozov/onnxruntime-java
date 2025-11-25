package com.badu.ai.onnx.genai;

import com.badu.ai.onnx.genai.internal.Model;
import com.badu.ai.onnx.genai.internal.Tokenizer;

/**
 * Wrapper for ONNX Runtime GenAI Tokenizer with type-safe interface. Provides encoding (text →
 * token IDs) and decoding (token IDs → text).
 * <p>
 * Note: SimpleGenAI handles tokenization internally for generation. This wrapper is useful for
 * token counting, validation, and testing.
 * <p>
 * The GenAI Tokenizer automatically loads configuration from the model directory (tokenizer.json,
 * vocab.json).
 */
public class TokenizerWrapper implements AutoCloseable {

  private final Tokenizer tokenizer;
  private final Model model; // Keep model reference for resource cleanup

  private TokenizerWrapper(Tokenizer tokenizer, Model model) {
    this.tokenizer = tokenizer;
    this.model = model;
  }

  /**
   * Returns the underlying GenAI Tokenizer instance.
   *
   * @return GenAI Tokenizer
   */
  public Tokenizer getTokenizer() {
    return tokenizer;
  }

  /**
   * Creates GenAI Tokenizer from model directory path. This creates a separate Model instance for
   * tokenization only.
   *
   * @param modelDir path to model directory
   * @return initialized TokenizerWrapper
   * @throws RuntimeException if tokenizer creation fails
   */
  public static TokenizerWrapper create(String modelDir) {
    try {
      // Create Model instance for tokenization
      Model model = new Model(modelDir);

      // Create Tokenizer from Model - GenAI handles loading config files
      Tokenizer genAiTokenizer = new Tokenizer(model);

      return new TokenizerWrapper(genAiTokenizer, model);
    } catch (Exception e) {
      throw new RuntimeException("Failed to create tokenizer from: " + modelDir, e);
    }
  }

  /**
   * Encodes text to token IDs using GenAI Tokenizer.
   *
   * @param text input text (non-null)
   * @return token ID array
   * @throws RuntimeException if encoding fails
   */
  public int[] encode(String text) {
    if (text == null) {
      throw new IllegalArgumentException("text cannot be null");
    }

    try {
      // GenAI Tokenizer.encode() returns Sequences object
      var sequences = tokenizer.encode(text);

      // Extract token IDs as int[]
      return sequences.getSequence(0);
    } catch (Exception e) {
      throw new RuntimeException("Tokenization failed for text: "
          + text.substring(0, Math.min(text.length(), 100)) + "...", e);
    }
  }

  /**
   * Decodes token IDs to text using GenAI Tokenizer.
   *
   * @param tokenIds token ID array (non-null)
   * @return decoded text
   * @throws RuntimeException if decoding fails
   */
  public String decode(int[] tokenIds) {
    if (tokenIds == null) {
      throw new IllegalArgumentException("tokenIds cannot be null");
    }

    try {
      // GenAI Tokenizer.decode() converts token IDs back to text
      return tokenizer.decode(tokenIds);
    } catch (Exception e) {
      throw new RuntimeException("Decoding failed for " + tokenIds.length + " tokens", e);
    }
  }

  /**
   * Releases tokenizer and model resources. Must close tokenizer before model.
   */
  @Override
  public void close() {
    // Close tokenizer first
    if (tokenizer != null) {
      try {
        tokenizer.close();
      } catch (Exception e) {
        System.err.println("Warning: Error closing GenAI Tokenizer: " + e.getMessage());
      }
    }

    // Then close model
    if (model != null) {
      try {
        model.close();
      } catch (Exception e) {
        System.err.println("Warning: Error closing GenAI Model: " + e.getMessage());
      }
    }
  }
}
