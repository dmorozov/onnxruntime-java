package com.badu.ai.onnx.tokenization;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Path;

/**
 * T5 tokenizer wrapper using DJL HuggingFace Tokenizers.
 *
 * <p>Handles tokenization and decoding for T5 models (Flan-T5, T5-Base, etc.).
 * Uses the HuggingFace tokenizer.json format for compatibility.
 *
 * <p>Key features:
 * <ul>
 *   <li>Encodes text to token IDs (long[])</li>
 *   <li>Decodes token IDs back to text</li>
 *   <li>Handles special tokens (BOS/EOS) automatically</li>
 *   <li>Thread-safe for concurrent use</li>
 * </ul>
 *
 * <p>Usage:
 * <pre>{@code
 * try (T5Tokenizer tokenizer = new T5Tokenizer(Paths.get("models/flan-t5-small-ONNX/tokenizer.json"))) {
 *     long[] tokenIds = tokenizer.encode("Summarize: This is a test.");
 *     String decoded = tokenizer.decode(tokenIds);
 * }
 * }</pre>
 *
 * @see HuggingFaceTokenizer
 */
public class T5Tokenizer implements AutoCloseable {

  private static final Logger logger = LoggerFactory.getLogger(T5Tokenizer.class);

  private final HuggingFaceTokenizer tokenizer;
  private final Path tokenizerPath;

  /**
   * Creates a T5Tokenizer from a tokenizer.json file.
   *
   * @param tokenizerPath Path to tokenizer.json file
   * @throws IOException if tokenizer file cannot be loaded
   */
  public T5Tokenizer(Path tokenizerPath) throws IOException {
    this.tokenizerPath = tokenizerPath;

    logger.debug("Loading tokenizer from: {}", tokenizerPath);

    try {
      this.tokenizer = HuggingFaceTokenizer.newInstance(tokenizerPath);
      logger.debug("T5Tokenizer loaded successfully from: {}", tokenizerPath);
    } catch (IOException e) {
      logger.error("Failed to load tokenizer from: {}", tokenizerPath, e);
      throw new IOException("Failed to load tokenizer from: " + tokenizerPath, e);
    }
  }

  /**
   * Encodes text to token IDs.
   *
   * <p>The tokenizer automatically handles:
   * <ul>
   *   <li>Tokenization according to T5 vocabulary</li>
   *   <li>Special tokens (BOS/EOS) if configured in tokenizer.json</li>
   *   <li>Padding and truncation (if configured)</li>
   * </ul>
   *
   * @param text Input text to encode
   * @return Array of token IDs
   * @throws IllegalArgumentException if text is null or empty
   */
  public long[] encode(String text) {
    if (text == null || text.isEmpty()) {
      throw new IllegalArgumentException("Text cannot be null or empty");
    }

    logger.trace("Encoding text: {}", text.substring(0, Math.min(100, text.length())));

    Encoding encoding = tokenizer.encode(text);
    long[] tokenIds = encoding.getIds();

    logger.trace("Encoded {} characters to {} tokens", text.length(), tokenIds.length);

    return tokenIds;
  }

  /**
   * Encodes text with optional truncation.
   *
   * @param text Input text to encode
   * @param maxLength Maximum number of tokens (truncates if exceeded)
   * @return Array of token IDs
   */
  public long[] encode(String text, int maxLength) {
    long[] tokenIds = encode(text);

    if (tokenIds.length > maxLength) {
      logger.warn("Truncating tokens from {} to {} (max length)", tokenIds.length, maxLength);
      long[] truncated = new long[maxLength];
      System.arraycopy(tokenIds, 0, truncated, 0, maxLength);
      return truncated;
    }

    return tokenIds;
  }

  /**
   * Decodes token IDs back to text.
   *
   * <p>The tokenizer automatically:
   * <ul>
   *   <li>Skips special tokens (BOS/EOS/PAD)</li>
   *   <li>Reconstructs text with proper spacing</li>
   *   <li>Handles subword tokens correctly</li>
   * </ul>
   *
   * @param tokenIds Array of token IDs to decode
   * @return Decoded text
   * @throws IllegalArgumentException if tokenIds is null or empty
   */
  public String decode(long[] tokenIds) {
    if (tokenIds == null || tokenIds.length == 0) {
      throw new IllegalArgumentException("Token IDs cannot be null or empty");
    }

    logger.trace("Decoding {} tokens", tokenIds.length);

    // DJL tokenizer handles special token skipping automatically
    String decoded = tokenizer.decode(tokenIds);

    logger.trace("Decoded {} tokens to {} characters", tokenIds.length, decoded.length());

    return decoded;
  }

  /**
   * Decodes token IDs with special token skipping control.
   *
   * @param tokenIds Array of token IDs to decode
   * @param skipSpecialTokens Whether to skip special tokens in output
   * @return Decoded text
   */
  public String decode(long[] tokenIds, boolean skipSpecialTokens) {
    if (tokenIds == null || tokenIds.length == 0) {
      throw new IllegalArgumentException("Token IDs cannot be null or empty");
    }

    logger.trace("Decoding {} tokens (skipSpecialTokens={})", tokenIds.length, skipSpecialTokens);

    String decoded = tokenizer.decode(tokenIds, skipSpecialTokens);

    logger.trace("Decoded {} tokens to {} characters", tokenIds.length, decoded.length());

    return decoded;
  }

  /**
   * Gets the tokenizer instance (for advanced use).
   *
   * @return HuggingFaceTokenizer instance
   */
  public HuggingFaceTokenizer getTokenizer() {
    return tokenizer;
  }

  /**
   * Gets the tokenizer path.
   *
   * @return Path to tokenizer.json file
   */
  public Path getTokenizerPath() {
    return tokenizerPath;
  }

  /**
   * Closes the tokenizer and releases resources.
   */
  @Override
  public void close() {
    if (tokenizer != null) {
      tokenizer.close();
      logger.debug("T5Tokenizer closed");
    }
  }
}
