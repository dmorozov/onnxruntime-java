package com.badu.ai.onnx.config;

import org.junit.jupiter.api.Test;

import com.badu.ai.onnx.config.templates.LLama3Template;
import com.badu.ai.onnx.config.templates.Phi3Template;
import com.badu.ai.onnx.config.templates.Qwen3Template;

import org.junit.jupiter.api.DisplayName;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for ChatTemplate implementations. Tests all three template types: Qwen3, LLama3, Phi3.
 */
class ChatTemplateTest {

  @Test
  @DisplayName("Qwen3Template: formats prompt with custom system message")
  void qwen3Template_formatPrompt_customSystem() {
    ChatTemplate template = new Qwen3Template();

    String formatted =
        template.formatPrompt("You are a coding assistant.", "Write a hello world in Java");

    String expected =
        "<|im_start|>system\n" + "You are a coding assistant.<|im_end|>\n" + "<|im_start|>user\n"
            + "Write a hello world in Java<|im_end|>\n" + "<|im_start|>assistant\n";

    assertEquals(expected, formatted);
  }

  @Test
  @DisplayName("Qwen3Template: formats prompt with default system message when null")
  void qwen3Template_formatPrompt_nullSystem() {
    ChatTemplate template = new Qwen3Template();

    String formatted = template.formatPrompt(null, "What is 2+2?");

    String expected = "<|im_start|>system\n" + "You are a helpful assistant.<|im_end|>\n"
        + "<|im_start|>user\n" + "What is 2+2?<|im_end|>\n" + "<|im_start|>assistant\n";

    assertEquals(expected, formatted);
  }

  @Test
  @DisplayName("Qwen3Template: formats prompt with default system message when empty")
  void qwen3Template_formatPrompt_emptySystem() {
    ChatTemplate template = new Qwen3Template();

    String formatted = template.formatPrompt("   ", "Hello!");

    String expected = "<|im_start|>system\n" + "You are a helpful assistant.<|im_end|>\n"
        + "<|im_start|>user\n" + "Hello!<|im_end|>\n" + "<|im_start|>assistant\n";

    assertEquals(expected, formatted);
  }

  @Test
  @DisplayName("Qwen3Template: throws exception for null user prompt")
  void qwen3Template_formatPrompt_nullUser() {
    ChatTemplate template = new Qwen3Template();

    Exception exception = assertThrows(IllegalArgumentException.class, () -> {
      template.formatPrompt("System", null);
    });

    assertTrue(exception.getMessage().contains("userPrompt cannot be null"));
  }

  @Test
  @DisplayName("Qwen3Template: throws exception for empty user prompt")
  void qwen3Template_formatPrompt_emptyUser() {
    ChatTemplate template = new Qwen3Template();

    Exception exception = assertThrows(IllegalArgumentException.class, () -> {
      template.formatPrompt("System", "   ");
    });

    assertTrue(exception.getMessage().contains("userPrompt cannot be null or empty"));
  }

  @Test
  @DisplayName("Qwen3Template: returns correct EOS token ID")
  void qwen3Template_eosTokenId() {
    ChatTemplate template = new Qwen3Template();
    assertEquals(151645, template.getEosTokenId());
  }

  @Test
  @DisplayName("Qwen3Template: returns correct model type")
  void qwen3Template_modelType() {
    ChatTemplate template = new Qwen3Template();
    assertEquals("Qwen3", template.getModelType());
  }

  @Test
  @DisplayName("Qwen3Template: handles multi-line prompts correctly")
  void qwen3Template_formatPrompt_multiLine() {
    ChatTemplate template = new Qwen3Template();

    String userPrompt = "Line 1\nLine 2\nLine 3";
    String formatted = template.formatPrompt("System", userPrompt);

    assertTrue(formatted.contains("Line 1\nLine 2\nLine 3"));
    assertTrue(formatted.startsWith("<|im_start|>system\n"));
    assertTrue(formatted.endsWith("<|im_start|>assistant\n"));
  }

  @Test
  @DisplayName("LLama3Template: formats prompt with custom system message")
  void llama3Template_formatPrompt_customSystem() {
    ChatTemplate template = new LLama3Template();

    String formatted = template.formatPrompt("You are a helpful assistant.", "What is AI?");

    String expected = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        + "You are a helpful assistant.<|eot_id|>\n" + "<|start_header_id|>user<|end_header_id|>\n"
        + "What is AI?<|eot_id|>\n" + "<|start_header_id|>assistant<|end_header_id|>\n";

    assertEquals(expected, formatted);
  }

  @Test
  @DisplayName("LLama3Template: formats prompt with default system when null")
  void llama3Template_formatPrompt_nullSystem() {
    ChatTemplate template = new LLama3Template();

    String formatted = template.formatPrompt(null, "Hello");

    assertTrue(formatted.contains("You are a helpful assistant."));
    assertTrue(formatted.contains("<|begin_of_text|>"));
    assertTrue(formatted.contains("Hello<|eot_id|>"));
  }

  @Test
  @DisplayName("LLama3Template: throws exception for null user prompt")
  void llama3Template_formatPrompt_nullUser() {
    ChatTemplate template = new LLama3Template();

    assertThrows(IllegalArgumentException.class, () -> {
      template.formatPrompt("System", null);
    });
  }

  @Test
  @DisplayName("LLama3Template: returns correct EOS token ID")
  void llama3Template_eosTokenId() {
    ChatTemplate template = new LLama3Template();
    assertEquals(128009, template.getEosTokenId());
  }

  @Test
  @DisplayName("LLama3Template: returns correct model type")
  void llama3Template_modelType() {
    ChatTemplate template = new LLama3Template();
    assertEquals("LLama3.2", template.getModelType());
  }

  @Test
  @DisplayName("Phi3Template: formats prompt with custom system message")
  void phi3Template_formatPrompt_customSystem() {
    ChatTemplate template = new Phi3Template();

    String formatted = template.formatPrompt("You are an AI agent.", "Explain recursion");

    String expected = "<|system|>\n" + "You are an AI agent.<|end|>\n" + "<|user|>\n"
        + "Explain recursion<|end|>\n" + "<|assistant|>\n";

    assertEquals(expected, formatted);
  }

  @Test
  @DisplayName("Phi3Template: formats prompt with default system when null")
  void phi3Template_formatPrompt_nullSystem() {
    ChatTemplate template = new Phi3Template();

    String formatted = template.formatPrompt(null, "Test");

    assertTrue(formatted.contains("You are a helpful assistant."));
    assertTrue(formatted.contains("<|system|>"));
    assertTrue(formatted.contains("Test<|end|>"));
  }

  @Test
  @DisplayName("Phi3Template: throws exception for empty user prompt")
  void phi3Template_formatPrompt_emptyUser() {
    ChatTemplate template = new Phi3Template();

    assertThrows(IllegalArgumentException.class, () -> {
      template.formatPrompt("System", "");
    });
  }

  @Test
  @DisplayName("Phi3Template: returns correct EOS token ID")
  void phi3Template_eosTokenId() {
    ChatTemplate template = new Phi3Template();
    assertEquals(32007, template.getEosTokenId());
  }

  @Test
  @DisplayName("Phi3Template: returns correct model type")
  void phi3Template_modelType() {
    ChatTemplate template = new Phi3Template();
    assertEquals("Phi3.5", template.getModelType());
  }

  @Test
  @DisplayName("All templates: handle special characters in prompts")
  void allTemplates_formatPrompt_specialCharacters() {
    String specialChars = "Test with special chars: @#$%^&*()[]{}\"'<>";

    ChatTemplate qwen3 = new Qwen3Template();
    String qwen3Result = qwen3.formatPrompt("System", specialChars);
    assertTrue(qwen3Result.contains(specialChars));

    ChatTemplate llama3 = new LLama3Template();
    String llama3Result = llama3.formatPrompt("System", specialChars);
    assertTrue(llama3Result.contains(specialChars));

    ChatTemplate phi3 = new Phi3Template();
    String phi3Result = phi3.formatPrompt("System", specialChars);
    assertTrue(phi3Result.contains(specialChars));
  }

  @Test
  @DisplayName("All templates: handle very long prompts")
  void allTemplates_formatPrompt_longPrompt() {
    StringBuilder longPrompt = new StringBuilder();
    for (int i = 0; i < 1000; i++) {
      longPrompt.append("Word").append(i).append(" ");
    }
    String prompt = longPrompt.toString();

    ChatTemplate qwen3 = new Qwen3Template();
    assertDoesNotThrow(() -> qwen3.formatPrompt("System", prompt));

    ChatTemplate llama3 = new LLama3Template();
    assertDoesNotThrow(() -> llama3.formatPrompt("System", prompt));

    ChatTemplate phi3 = new Phi3Template();
    assertDoesNotThrow(() -> phi3.formatPrompt("System", prompt));
  }
}
