package com.badu.ai.onnx.gpu;

import lombok.Builder;
import lombok.Value;

/**
 * Represents a single GPU device with metadata.
 *
 * <p>Encapsulates GPU properties including device ID, name, memory,
 * availability status, and capabilities.
 *
 * <p><b>Usage Example:</b>
 * <pre>{@code
 * GpuDevice gpu = GpuDevice.builder()
 *     .deviceId(0)
 *     .name("NVIDIA RTX 4090")
 *     .totalMemoryMB(24576)
 *     .available(true)
 *     .supportsFp16(true)
 *     .build();
 * }</pre>
 */
@Value
@Builder(toBuilder = true)
public class GpuDevice {

  /**
   * GPU device ID (0 for first GPU, 1 for second, etc.).
   */
  int deviceId;

  /**
   * GPU name (e.g., "NVIDIA RTX 4090", "AMD Radeon RX 7900 XTX").
   */
  String name;

  /**
   * Driver version (e.g., "535.104.05" for NVIDIA, "23.30" for AMD).
   */
  String driverVersion;

  /**
   * Total GPU memory in MB.
   */
  long totalMemoryMB;

  /**
   * Free GPU memory in MB (may change over time).
   */
  @Builder.Default
  long freeMemoryMB = 0;

  /**
   * GPU utilization percentage (0-100).
   */
  @Builder.Default
  int utilizationPercent = 0;

  /**
   * Whether this GPU is available for inference.
   */
  @Builder.Default
  boolean available = true;

  /**
   * Whether this GPU supports FP16 (half-precision).
   */
  @Builder.Default
  boolean supportsFp16 = false;

  /**
   * Whether this GPU supports INT8 quantization.
   */
  @Builder.Default
  boolean supportsInt8 = true;

  /**
   * GPU adapter type (CUDA, ROCm, CoreML, etc.).
   */
  String adapterType;

  /**
   * Checks if this GPU has sufficient free memory for inference.
   *
   * @param requiredMemoryMB required memory in MB
   * @return true if sufficient memory available
   */
  public boolean hasSufficientMemory(long requiredMemoryMB) {
    return freeMemoryMB >= requiredMemoryMB;
  }

  /**
   * Checks if this GPU is idle (low utilization).
   *
   * @param maxUtilization maximum acceptable utilization (0-100)
   * @return true if utilization is below threshold
   */
  public boolean isIdle(int maxUtilization) {
    return utilizationPercent <= maxUtilization;
  }

  /**
   * Creates a short display string for this GPU.
   *
   * @return display string (e.g., "GPU 0: NVIDIA RTX 4090 (24GB)")
   */
  public String toDisplayString() {
    return String.format("GPU %d: %s (%dGB)",
        deviceId, name, totalMemoryMB / 1024);
  }

  /**
   * Creates a detailed info string including memory and utilization.
   *
   * @return detailed info string
   */
  public String toDetailedString() {
    return String.format("GPU %d: %s [%s] - Memory: %d/%dGB, Util: %d%%, %s",
        deviceId, name, adapterType,
        (totalMemoryMB - freeMemoryMB) / 1024, totalMemoryMB / 1024,
        utilizationPercent,
        available ? "Available" : "Unavailable");
  }
}
