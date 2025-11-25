package com.badu.ai.onnx.genai.internal;

import java.util.ArrayList;
import java.util.List;

/**
 * Represents a collection of encoded prompts/responses.
 * Each sequence is an array of token IDs.
 */
public final class Sequences implements AutoCloseable {

  private final List<int[]> sequences;

  /**
   * Creates an empty Sequences collection.
   */
  public Sequences() {
    this.sequences = new ArrayList<>();
  }

  /**
   * Creates a Sequences collection from a list of token arrays.
   *
   * @param sequences List of token ID arrays
   */
  public Sequences(List<int[]> sequences) {
    this.sequences = new ArrayList<>(sequences);
  }

  /**
   * Creates a Sequences collection with a single sequence.
   *
   * @param sequence Array of token IDs
   */
  public Sequences(int[] sequence) {
    this.sequences = new ArrayList<>();
    this.sequences.add(sequence);
  }

  /**
   * Gets the number of sequences in the collection. This is equivalent to the batch size.
   *
   * @return The number of sequences.
   */
  public long numSequences() {
    return sequences.size();
  }

  /**
   * Gets the sequence at the specified index.
   *
   * @param sequenceIndex The index of the sequence.
   * @return The sequence as an array of integers.
   * @throws IndexOutOfBoundsException if the index is out of range
   */
  public int[] getSequence(long sequenceIndex) {
    if (sequenceIndex < 0 || sequenceIndex >= sequences.size()) {
      throw new IndexOutOfBoundsException(
          "Sequence index " + sequenceIndex + " out of bounds for size " + sequences.size());
    }
    return sequences.get((int) sequenceIndex);
  }

  /**
   * Adds a sequence to the collection.
   *
   * @param sequence Array of token IDs to add
   */
  public void addSequence(int[] sequence) {
    this.sequences.add(sequence);
  }

  /**
   * Gets all sequences as a list.
   *
   * @return List of all token ID arrays
   */
  public List<int[]> getAllSequences() {
    return new ArrayList<>(sequences);
  }

  /**
   * Closes the Sequences and releases any associated resources.
   * Currently a no-op as Sequences uses standard Java collections.
   */
  @Override
  public void close() {
    // No resources to release
  }
}
