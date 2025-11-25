package com.badu.ai.onnx.gpu;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for GpuDeviceManager.
 *
 * <p>Tests GPU discovery, device querying, and load balancing logic.
 *
 * <p><b>Note:</b> These tests are designed to work on systems with or without GPUs.
 * If no GPUs are available, tests verify graceful degradation.
 */
@DisplayName("GpuDeviceManager Tests")
public class GpuDeviceManagerTest {

  @Test
  @DisplayName("1. GPU Discovery")
  void testGpuDiscovery() {
    GpuDeviceManager manager = new GpuDeviceManager();
    List<GpuDevice> gpus = manager.discoverGpus();

    assertNotNull(gpus, "GPU list should not be null");

    if (gpus.isEmpty()) {
      System.out.println("  No GPUs detected (this is expected on CPU-only systems)");
    } else {
      System.out.println("  Discovered " + gpus.size() + " GPU(s):");
      for (GpuDevice gpu : gpus) {
        System.out.println("    " + gpu.toDetailedString());

        // Verify basic properties
        assertTrue(gpu.getDeviceId() >= 0, "Device ID should be non-negative");
        assertNotNull(gpu.getName(), "GPU name should not be null");
        assertNotNull(gpu.getAdapterType(), "Adapter type should not be null");
      }
    }
  }

  @Test
  @DisplayName("2. Query Specific GPU Device")
  void testQueryGpuDevice() {
    GpuDeviceManager manager = new GpuDeviceManager();

    // Try to query GPU 0
    try {
      GpuDevice gpu = manager.queryGpuDevice(0);
      assertNotNull(gpu, "GPU device should not be null");
      assertEquals(0, gpu.getDeviceId(), "Device ID should be 0");

      System.out.println("  GPU 0: " + gpu.toDisplayString());

      // Verify capabilities
      System.out.println("    FP16 support: " + gpu.isSupportsFp16());
      System.out.println("    INT8 support: " + gpu.isSupportsInt8());

    } catch (Exception e) {
      // GPU 0 doesn't exist - this is fine for CPU-only systems
      System.out.println("  GPU 0 not available (CPU-only system)");
    }
  }

  @Test
  @DisplayName("3. Update GPU Stats")
  void testUpdateGpuStats() {
    GpuDeviceManager manager = new GpuDeviceManager();
    List<GpuDevice> gpus = manager.discoverGpus();

    if (gpus.isEmpty()) {
      System.out.println("  Skipping stats update test (no GPUs available)");
      return;
    }

    GpuDevice gpu = gpus.get(0);
    System.out.println("  Initial state: " + gpu.toDetailedString());

    GpuDevice updated = manager.updateGpuStats(gpu);
    assertNotNull(updated, "Updated GPU should not be null");
    assertEquals(gpu.getDeviceId(), updated.getDeviceId(), "Device ID should match");

    System.out.println("  Updated state: " + updated.toDetailedString());
  }

  @Test
  @DisplayName("4. Find Least Loaded GPU")
  void testFindLeastLoadedGpu() {
    // Create mock GPU devices for testing
    List<GpuDevice> gpus = List.of(
        GpuDevice.builder()
            .deviceId(0)
            .name("GPU 0")
            .adapterType("CUDA")
            .utilizationPercent(80)
            .totalMemoryMB(8192)
            .freeMemoryMB(2048)
            .available(true)
            .build(),
        GpuDevice.builder()
            .deviceId(1)
            .name("GPU 1")
            .adapterType("CUDA")
            .utilizationPercent(30)  // Least loaded
            .totalMemoryMB(8192)
            .freeMemoryMB(6144)
            .available(true)
            .build(),
        GpuDevice.builder()
            .deviceId(2)
            .name("GPU 2")
            .adapterType("CUDA")
            .utilizationPercent(60)
            .totalMemoryMB(8192)
            .freeMemoryMB(4096)
            .available(true)
            .build()
    );

    GpuDeviceManager manager = new GpuDeviceManager();
    GpuDevice leastLoaded = manager.findLeastLoadedGpu(gpus);

    assertNotNull(leastLoaded, "Should find least loaded GPU");
    assertEquals(1, leastLoaded.getDeviceId(), "GPU 1 should be least loaded");
    assertEquals(30, leastLoaded.getUtilizationPercent(), "Utilization should be 30%");

    System.out.println("  Least loaded GPU: " + leastLoaded.toDisplayString());
  }

  @Test
  @DisplayName("5. Find GPU with Most Free Memory")
  void testFindGpuWithMostMemory() {
    // Create mock GPU devices for testing
    List<GpuDevice> gpus = List.of(
        GpuDevice.builder()
            .deviceId(0)
            .name("GPU 0")
            .adapterType("CUDA")
            .totalMemoryMB(8192)
            .freeMemoryMB(2048)
            .available(true)
            .build(),
        GpuDevice.builder()
            .deviceId(1)
            .name("GPU 1")
            .adapterType("CUDA")
            .totalMemoryMB(8192)
            .freeMemoryMB(6144)  // Most free
            .available(true)
            .build(),
        GpuDevice.builder()
            .deviceId(2)
            .name("GPU 2")
            .adapterType("CUDA")
            .totalMemoryMB(8192)
            .freeMemoryMB(4096)
            .available(true)
            .build()
    );

    GpuDeviceManager manager = new GpuDeviceManager();
    GpuDevice mostFree = manager.findGpuWithMostMemory(gpus);

    assertNotNull(mostFree, "Should find GPU with most free memory");
    assertEquals(1, mostFree.getDeviceId(), "GPU 1 should have most free memory");
    assertEquals(6144, mostFree.getFreeMemoryMB(), "Free memory should be 6144MB");

    System.out.println("  GPU with most memory: " + mostFree.toDisplayString());
  }

  @Test
  @DisplayName("6. GPU Device Helper Methods")
  void testGpuDeviceHelpers() {
    GpuDevice gpu = GpuDevice.builder()
        .deviceId(0)
        .name("Test GPU")
        .adapterType("CUDA")
        .totalMemoryMB(8192)
        .freeMemoryMB(4096)
        .utilizationPercent(25)
        .available(true)
        .build();

    // Test hasSufficientMemory
    assertTrue(gpu.hasSufficientMemory(2048), "Should have sufficient memory for 2GB");
    assertFalse(gpu.hasSufficientMemory(8192), "Should not have sufficient memory for 8GB");

    // Test isIdle
    assertTrue(gpu.isIdle(30), "GPU should be idle with 25% utilization");
    assertFalse(gpu.isIdle(20), "GPU should not be idle with 25% utilization (threshold 20%)");

    // Test display strings
    String display = gpu.toDisplayString();
    assertNotNull(display);
    assertTrue(display.contains("GPU 0"), "Display should contain device ID");
    System.out.println("  Display: " + display);

    String detailed = gpu.toDetailedString();
    assertNotNull(detailed);
    assertTrue(detailed.contains("25%"), "Detailed should contain utilization");
    System.out.println("  Detailed: " + detailed);
  }
}
