package com.badu.ai.onnx.genai.internal;

/**
 * A TokenizerStream is used to convert individual tokens when using Generator.generateNextToken.
 * Provides streaming decode functionality for token-by-token generation.
 */
public class TokenizerStream implements AutoCloseable {

  private final Tokenizer tokenizer;

  /**
   * Creates a TokenizerStream from the given tokenizer.
   *
   * @param tokenizer The tokenizer to use for decoding.
   */
  public TokenizerStream(Tokenizer tokenizer) {
    this.tokenizer = tokenizer;
  }

  /**
   * Decode one token.
   *
   * @param token The token ID.
   * @return The decoded result (text representation of the token).
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public String decode(int token) throws GenAIException {
    try {
      return tokenizer.decode(new int[]{token});
    } catch (Exception e) {
      throw new GenAIException("Failed to decode token: " + token, e);
    }
  }

  @Override
  public void close() {
    // TokenizerStream doesn't own the tokenizer, so don't close it
    // The parent Tokenizer will handle cleanup
  }
}
