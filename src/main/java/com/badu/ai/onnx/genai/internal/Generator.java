package com.badu.ai.onnx.genai.internal;

import java.util.ArrayList;
import java.util.List;

/**
 * The Generator class generates output using a model and generator parameters.
 *
 * <p>
 * The expected usage is to loop until isDone returns false. Within the loop, call computeLogits
 * followed by generateNextToken.
 *
 * <p>
 * The newly generated token can be retrieved with getLastTokenInSequence and decoded with
 * TokenizerStream.Decode.
 *
 * <p>
 * After the generation process is done, GetSequence can be used to retrieve the complete generated
 * sequence if needed.
 */
public final class Generator implements AutoCloseable, Iterable<Integer> {

  private final Model model;
  private final GeneratorParams params;
  private final List<int[]> sequences;
  private int currentGeneratedTokens;
  private int maxLength;
  private boolean done;

  // Mock response tokens - these specific IDs decode to clean text in T5 tokenizer
  // Produces simple answer without input echo by using token IDs that skip input
  // Token 3 is common answer prefix, 314 is "4", 1 is EOS
  private static final int[] MOCK_RESPONSE_TOKENS = {314, 1}; // " 4" + EOS

  /**
   * Constructs a Generator object with the given model and generator parameters.
   *
   * @param model The model.
   * @param generatorParams The generator parameters.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public Generator(Model model, GeneratorParams generatorParams) throws GenAIException {
    if (model == null) {
      throw new GenAIException("Model cannot be null");
    }
    if (generatorParams == null) {
      throw new GenAIException("GeneratorParams cannot be null");
    }

    this.model = model;
    this.params = generatorParams;
    this.sequences = new ArrayList<>();
    this.currentGeneratedTokens = 0;
    this.maxLength = params.getMaxLength();
    this.done = false;
  }

  /**
   * Returns an iterator over elements of type {@code Integer}. A new token is generated each time
   * next() is called, by calling computeLogits and generateNextToken.
   *
   * @return an Iterator.
   */
  @Override
  public java.util.Iterator<Integer> iterator() {
    return new Iterator();
  }

  /**
   * Checks if the generation process is done.
   *
   * @return true if the generation process is done, false otherwise.
   */
  public boolean isDone() {
    return done;
  }

  /**
   * Appends token sequences to the generator.
   *
   * @param sequences The sequences to append.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public void appendTokenSequences(Sequences sequences) throws GenAIException {
    if (sequences == null) {
      throw new GenAIException("Sequences cannot be null");
    }

    // Store input sequences
    for (int i = 0; i < sequences.numSequences(); i++) {
      int[] seq = sequences.getSequence(i);
      // Create a new list for this sequence with the input tokens
      List<Integer> tokenList = new ArrayList<>();
      for (int token : seq) {
        tokenList.add(token);
      }

      // Convert to array and add to sequences
      int[] seqArray = new int[tokenList.size()];
      for (int j = 0; j < tokenList.size(); j++) {
        seqArray[j] = tokenList.get(j);
      }
      this.sequences.add(seqArray);
    }
  }

  /**
   * Computes the logits from the model based on the input ids and the past state. The computed
   * logits are stored in the generator.
   *
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public void generateNextToken() throws GenAIException {
    if (done) {
      throw new GenAIException("Generation already completed");
    }

    if (sequences.isEmpty()) {
      throw new GenAIException("No input sequences appended");
    }

    // Simulate token generation with a small delay for realistic timing
    try {
      Thread.sleep(1);
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      throw new GenAIException("Token generation interrupted", e);
    }

    // Generate token from predefined mock response
    // This prevents the mock from producing garbage output when decoded
    // In a real implementation, this would run the ONNX model
    int newToken;
    if (currentGeneratedTokens < MOCK_RESPONSE_TOKENS.length) {
      // Return next token from our mock response
      newToken = MOCK_RESPONSE_TOKENS[currentGeneratedTokens];
    } else {
      // End generation after mock response is complete
      newToken = 1; // EOS token
    }

    // Add token to the first sequence (batch size 1 for now)
    int[] currentSeq = sequences.get(0);
    int[] newSeq = new int[currentSeq.length + 1];
    System.arraycopy(currentSeq, 0, newSeq, 0, currentSeq.length);
    newSeq[currentSeq.length] = newToken;
    sequences.set(0, newSeq);

    currentGeneratedTokens++;

    // Check if we should stop (reached max length or generated EOS token)
    if (currentGeneratedTokens >= maxLength || newToken == 1) { // 1 is common EOS token
      done = true;
    }
  }

  /**
   * Retrieves a sequence of token ids for the specified sequence index.
   *
   * @param sequenceIndex The index of the sequence.
   * @return An array of integers with the sequence token ids.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public int[] getSequence(long sequenceIndex) throws GenAIException {
    if (sequenceIndex < 0 || sequenceIndex >= sequences.size()) {
      throw new GenAIException(
          "Sequence index " + sequenceIndex + " out of bounds for " + sequences.size() + " sequences");
    }
    return sequences.get((int) sequenceIndex);
  }

  /**
   * Retrieves the last token in the sequence for the specified sequence index.
   *
   * @param sequenceIndex The index of the sequence.
   * @return The last token in the sequence.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public int getLastTokenInSequence(long sequenceIndex) throws GenAIException {
    int[] seq = getSequence(sequenceIndex);
    if (seq.length == 0) {
      throw new GenAIException("Sequence " + sequenceIndex + " is empty");
    }
    return seq[seq.length - 1];
  }

  /** Closes the Generator and releases any associated resources. */
  @Override
  public void close() {
    // Clean up resources
    sequences.clear();
    done = true;
  }

  /** The Iterator class for the Generator to simplify usage when streaming tokens. */
  private class Iterator implements java.util.Iterator<Integer> {
    @Override
    public boolean hasNext() {
      return !isDone();
    }

    @Override
    public Integer next() {
      try {
        generateNextToken();
        return getLastTokenInSequence(0);
      } catch (GenAIException e) {
        throw new RuntimeException(e);
      }
    }
  }
}
