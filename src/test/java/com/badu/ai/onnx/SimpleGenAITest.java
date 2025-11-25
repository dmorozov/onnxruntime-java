package com.badu.ai.onnx;

import com.badu.ai.onnx.genai.internal.GeneratorParams;
import com.badu.ai.onnx.genai.internal.SimpleGenAI;
import org.junit.jupiter.api.DisplayName;

import java.nio.file.Files;
import java.nio.file.Paths;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Direct SimpleGenAI API test to debug tokenizer issues.
 * Requires models from "models/" directory.
 */
class SimpleGenAITest {

  @DisplayName("Direct SimpleGenAI test")
  void directSimpleGenAITest() throws Exception {
    String modelDir = "models/flan-t5-base-ONNX";
    assumeTrue(Files.exists(Paths.get(modelDir)), "Model directory not found");

    System.out.println("Creating SimpleGenAI...");
    try (SimpleGenAI genAI = new SimpleGenAI(modelDir)) {
      System.out.println("SimpleGenAI created successfully");

      System.out.println("Creating GeneratorParams...");
      GeneratorParams params = genAI.createGeneratorParams();
      System.out.println("GeneratorParams created successfully");

      params.setSearchOption("max_length", 50);
      System.out.println("Search options set successfully");

      String prompt = "What is 2+2?";
      System.out.println("Attempting generation with prompt: " + prompt);

      String result = genAI.generate(params, prompt, null);
      System.out.println("Generation successful!");
      System.out.println("Result: " + result);
    }
  }
}
