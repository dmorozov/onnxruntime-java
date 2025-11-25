package com.badu.ai.onnx.gpu;

import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtProvider;
import ai.onnxruntime.OrtSession;

/**
 * Abstract GPU adapter interface for different GPU backends.
 *
 * <p>Provides a unified interface for configuring ONNX Runtime execution providers
 * across different GPU architectures (CUDA, ROCm, CoreML, etc.).
 *
 * <p><b>Supported GPU Backends:</b>
 * <ul>
 *   <li>{@link CudaGpuAdapter} - NVIDIA GPUs (CUDA)</li>
 *   <li>{@link RocmGpuAdapter} - AMD GPUs (ROCm)</li>
 *   <li>{@link CoreMlGpuAdapter} - Apple Silicon (Metal via CoreML)</li>
 *   <li>{@link WebGpuAdapter} - Cross-platform (WebGPU/Vulkan)</li>
 * </ul>
 *
 * <p><b>Usage Example:</b>
 * <pre>{@code
 * GpuAdapter adapter = GpuAdapterFactory.createAdapter();
 * if (adapter.isAvailable()) {
 *     adapter.configure(sessionOptions, deviceId);
 * }
 * }</pre>
 */
public interface GpuAdapter {

  /**
   * Gets the provider type for this adapter.
   *
   * @return OrtProvider type (e.g., CUDA, ROCM, CORE_ML)
   */
  OrtProvider getProviderType();

  /**
   * Checks if this GPU adapter is available on the current system.
   *
   * @return true if the GPU hardware and drivers are available
   */
  boolean isAvailable();

  /**
   * Configures the ONNX Runtime session options for this GPU.
   *
   * @param options Session options to configure
   * @param deviceId GPU device ID (0 for first GPU, 1 for second, etc.)
   * @throws OrtException if configuration fails
   */
  void configure(OrtSession.SessionOptions options, int deviceId) throws OrtException;

  /**
   * Gets a human-readable name for this GPU adapter.
   *
   * @return GPU adapter name (e.g., "NVIDIA CUDA", "AMD ROCm")
   */
  String getName();

  /**
   * Gets GPU device information for debugging.
   *
   * @param deviceId GPU device ID
   * @return Device information string
   */
  String getDeviceInfo(int deviceId);

  /**
   * Gets the number of available GPU devices.
   *
   * @return Number of GPUs, or 0 if not available
   */
  int getDeviceCount();

  /**
   * Gets the recommended memory arena size for this GPU (in MB).
   *
   * <p>This is used to configure memory allocation strategies.
   *
   * @return Recommended arena size in MB, or -1 for default
   */
  default int getRecommendedArenaSize() {
    return -1; // Use ONNX Runtime default
  }

  /**
   * Checks if this adapter supports FP16 (half-precision) inference.
   *
   * @return true if FP16 is supported
   */
  default boolean supportsFp16() {
    return false; // Conservative default
  }

  /**
   * Checks if this adapter supports INT8 quantization.
   *
   * @return true if INT8 is supported
   */
  default boolean supportsInt8() {
    return true; // Most GPUs support INT8
  }
}
