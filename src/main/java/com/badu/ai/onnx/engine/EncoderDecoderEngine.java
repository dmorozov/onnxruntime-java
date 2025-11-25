package com.badu.ai.onnx.engine;

import com.badu.ai.onnx.InferenceException;
import com.badu.ai.onnx.InferenceResponse;
import com.badu.ai.onnx.TokenCallback;
import com.badu.ai.onnx.config.GenerationConfig;
import com.badu.ai.onnx.config.ModelConfig;
import com.badu.ai.onnx.metrics.PerformanceMetrics;

// TODO: Implement Encoder-Decoder engine
public class EncoderDecoderEngine implements InferenceEngine{

    @Override
    public void initialize(ModelConfig modelConfig, GenerationConfig generationConfig) throws InferenceException {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'initialize'");
    }

    @Override
    public InferenceResponse generate(String prompt) throws InferenceException {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'generate'");
    }

    @Override
    public InferenceResponse generateStreaming(String prompt, TokenCallback callback) throws InferenceException {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'generateStreaming'");
    }

    @Override
    public PerformanceMetrics getMetrics() {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'getMetrics'");
    }

    @Override
    public boolean isInitialized() {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'isInitialized'");
    }

    @Override
    public void close() {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'close'");
    }
}
