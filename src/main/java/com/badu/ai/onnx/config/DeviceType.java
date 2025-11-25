package com.badu.ai.onnx.config;

/**
 * Execution device type for ONNX Runtime inference.
 *
 * <p>ONNX Runtime supports multiple execution providers:
 * <ul>
 *   <li><b>CPU</b>: Default provider, runs on any machine (recommended for most use cases)</li>
 *   <li><b>GPU</b>: CUDA provider for NVIDIA GPUs (requires CUDA runtime and GPU build)</li>
 * </ul>
 *
 * <p>Device selection affects:
 * <ul>
 *   <li>Inference latency (GPU typically 2-5x faster for large models)</li>
 *   <li>Memory location (system RAM vs GPU VRAM)</li>
 *   <li>Dependency requirements (CPU-only vs CUDA libraries)</li>
 * </ul>
 *
 * @see com.badu.ai.onnx.config.ModelConfig
 */
public enum DeviceType {
    /**
     * CPU execution provider.
     *
     * <p>Advantages:
     * <ul>
     *   <li>No special dependencies (works everywhere)</li>
     *   <li>Lower memory overhead</li>
     *   <li>Suitable for small models (FLAN-T5-Small, DistilBART)</li>
     * </ul>
     *
     * <p>Default for most deployments.
     */
    CPU,

    /**
     * GPU execution provider (NVIDIA CUDA).
     *
     * <p>Advantages:
     * <ul>
     *   <li>2-5x faster inference for large models</li>
     *   <li>Better batch processing throughput</li>
     *   <li>Recommended for Phi-3 Mini and larger models</li>
     * </ul>
     *
     * <p>Requirements:
     * <ul>
     *   <li>NVIDIA GPU with CUDA support</li>
     *   <li>CUDA runtime libraries installed</li>
     *   <li>GPU-enabled ONNX Runtime build (mvn install -Dgpu)</li>
     * </ul>
     *
     * <p>Note: If GPU is selected but unavailable, ONNX Runtime will fall back to CPU.
     */
    GPU;

    /**
     * Gets the ONNX Runtime provider name for this device type.
     *
     * @return Provider name ("CPUExecutionProvider" or "CUDAExecutionProvider")
     */
    public String getProviderName() {
        return switch (this) {
            case CPU -> "CPUExecutionProvider";
            case GPU -> "CUDAExecutionProvider";
        };
    }

    /**
     * Checks if this device type is CPU.
     *
     * @return true if CPU, false otherwise
     */
    public boolean isCpu() {
        return this == CPU;
    }

    /**
     * Checks if this device type is GPU.
     *
     * @return true if GPU, false otherwise
     */
    public boolean isGpu() {
        return this == GPU;
    }
}
