# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Java library implementing **ONNX Runtime I/O Bindings** using **Project Panama Foreign Function & Memory (FFM) API** (JDK 21+). The primary goal is to enable **zero-copy GPU inference** for ONNX models in Java, eliminating high-latency PCIe data transfers between CPU and GPU that currently plague the official ONNX Runtime Java bindings.

**Critical Context**: The official ONNX Runtime Java API does not expose I/O Binding functionality (available in C++/Python/C#), forcing expensive CPU↔GPU memory copies for every inference call. This project bridges that gap using FFM instead of legacy JNI.

## Build & Development Commands

### Standard Build
```bash
# Build with GPU support (default profile)
mvn clean install

# Build with CPU-only support
mvn clean install -P cpu

# Build without GPU profile
mvn clean install -P \!gpu
```

### Testing
```bash
# Run unit tests only (excludes integration tests by default)
mvn test

# Run integration tests (requires model files)
mvn verify -P integration-tests

# Run all tests including integration
mvn clean verify -P integration-tests
```

### Java Version & FFM API
- **Java 21+** is required (uses FFM API from JEP 442)
- Run with native access enabled:
```bash
java --enable-native-access=ALL-UNNAMED -jar target/onnx-java-summarizer-1.0.0-SNAPSHOT.jar
```

### Maven Profiles
- `gpu` (default): Includes `onnxruntime_gpu` dependency with CUDA support
- `cpu`: CPU-only inference (no GPU libraries)
- `integration-tests`: Enables tests requiring actual ONNX model files

## Architecture Overview

### FFM-Based I/O Binding Implementation (Core Objective)

The project implements **three critical FFM wrapper classes** that map to ONNX Runtime C++ API:

1. **`OrtMemoryInfo`**: Specifies device memory allocation (CUDA VRAM vs CPU)
   - Factory methods: `createCuda(deviceId)`, `createCudaPinned()`
   - Wraps native `Ort::MemoryInfo` struct via FFM `MemorySegment`

2. **`OrtDeviceTensor`**: Represents GPU-resident tensors (`Ort::Value` on device)
   - Pre-allocates tensors in GPU VRAM using device allocators
   - Enables zero-copy data binding (data never leaves GPU)

3. **`OrtIoBinding`**: Main I/O binding orchestrator
   - Methods: `bindInput(name, tensor)`, `bindOutput(name, memorySpec)`, `run()`
   - Eliminates PCIe transfers by binding device memory directly to ONNX session

**Key Architectural Principle**: All native C++ pointers (`OrtSession*`, `OrtIoBinding*`, GPU VRAM addresses) are managed as FFM `MemorySegment` objects, NOT raw `jlong` pointers. This provides JVM-managed lifetime tracking and bounds checking, preventing native crashes common in JNI.

### Inference Engine Architecture

The library provides a **multi-engine facade** for different model architectures:

- **`InferenceEngine`** (interface): Common contract for all engines
  - `initialize(modelConfig, generationConfig)`: Loads models, initializers
  - `generate(prompt)`: Blocking generation
  - `generateStreaming(prompt, callback)`: Token-by-token streaming (TTFT optimization)
  - `getMetrics()`: Returns performance data (TTFT, throughput, latency)

- **`InferenceEngineFactory`**: Auto-detects model architecture from files:
  - `T5EncoderDecoderEngine`: For T5/BART/DistilBART (encoder + decoder sessions)
  - `SimpleGenAIEngine`: Legacy single-session generation
  - `DecoderOnlyEngine`: Future GPT-style support

- **`OnnxInference`**: Facade class exposing simple API:
  ```java
  OnnxInference inference = OnnxInference.create(modelConfig, genConfig);
  InferenceResponse response = inference.generate("Summarize: ...");
  ```

### GPU Adapter System

Multi-backend GPU support via adapter pattern:

- **`GpuAdapter`** (interface): Abstracts execution provider configuration
- Implementations: `CudaGpuAdapter`, `RocmGpuAdapter`, `CoreMlGpuAdapter`, `WebGpuAdapter`
- **`GpuDeviceManager`**: Manages device selection and availability detection
- All adapters configure `SessionOptions` for their respective execution providers

### GenAI Integration (LLM-Specific Optimization)

Located in `com.badu.ai.onnx.genai.internal` package:

- **`SimpleGenAI`**: Wrapper around ONNX Runtime GenAI library
  - Implements **KV cache management** for transformer models
  - Provides token streaming via `Consumer<String>` callback
  - Handles prompt encoding/decoding with `Tokenizer` and `TokenizerStream`

- **`Generator`**: Core generative loop implementation
  - Iterates tokens: `computeLogits()` → `generateNextToken()` → `isDone()`
  - Manages `Sequences` for multi-step autoregressive generation

**Important**: GenAI library internally uses I/O Binding-equivalent optimizations for LLM KV cache updates (keeps intermediate tensors GPU-resident across decoding steps).

### Configuration System

- **`ModelConfig`**: Model path, variant (INT8/INT4/FP16), device type
- **`GenerationConfig`**: Temperature, top-k, max tokens, repetition penalty
- **`ModelFileDiscovery`**: Auto-detects encoder/decoder ONNX files and tokenizer configs
- **`ChatTemplate`**: Prompt formatting for different model families (Llama3, Phi3, Qwen3, T5, BART)

### Performance Tracking

- **`PerformanceMetrics`**: Captures timing breakdown:
  - `timeToFirstTokenMs`: Critical for streaming UX
  - `initializationTimeMs`, `tokenizationTimeMs`, `encoderTimeMs`, `decoderTimeMs`
  - `totalTimeMs`, `tokensPerSecond`

- **`PerformanceTracker`**: Singleton metrics collector across sessions

## Critical Implementation Constraints

### FFM API Usage

1. **Never use raw `long` for native pointers**. Always use `MemorySegment`:
   ```java
   // WRONG (JNI-style)
   long nativeHandle = 0xDEADBEEF;

   // CORRECT (FFM-style)
   MemorySegment handle = OrtNativeCalls.createIoBinding(sessionSegment);
   ```

2. **Always implement `AutoCloseable`** for FFM wrappers:
   - Release native resources in `close()` via `OrtNativeCalls.release*(handle)`
   - Prevents GPU VRAM leaks

3. **Use `jextract` for C API binding generation** (conceptual in docs):
   - Target ONNX Runtime C headers: `onnxruntime_c_api.h`
   - Generate `MethodHandle` bindings for functions like `OrtCreateIoBinding`, `OrtBindInput`, etc.

### CUDA Memory Management

From `docs/Optimizations.md` and `docs/ONNX Runtime Java I_O Bindings.md`:

- **CUDA Memory Types**:
  - `Cuda`: Pure GPU VRAM (for compute-heavy tensors like KV cache)
  - `CudaPinned`: Page-locked host memory (for fast async CPU↔GPU transfers)

- **Allocation Strategy**:
  - Input tensors: Pre-allocate in VRAM using `OrtDeviceAllocator`
  - Output tensors: Bind `OrtMemoryInfo` (device spec) → ORT allocates dynamically on GPU

- **Performance Target**: Eliminate PCIe bottleneck (32 GB/s PCIe vs 1000 GB/s GPU VRAM)

### Optimization Priorities (from docs/Optimizations.md)

1. **Runtime I/O Binding**: **Highest priority** (30-70% latency reduction for GPU inference)
2. **Pre-run Quantization**: Load INT8/INT4 models (2-4x speedup, 50-75% size reduction)
3. **Execution Provider Selection**: TensorRT > CUDA > CPU (1.5-2x gain over base CUDA)
4. **Graph Optimization Level**: Set `ORT_ENABLE_ALL` in `SessionOptions`
5. **Thread Management**: Configure intra/inter-op threads for CPU fallback

**NOT Available in Java Yet** (future work):
- KV Cache Management APIs (currently internal to GenAI library)
- CUDA Graph Integration
- Dynamic Quantization APIs (requires offline Python/C++ tooling)

## Testing Strategy

- **Unit Tests**: `src/test/java/com/badu/ai/onnx/`
  - Run by default with `mvn test`
  - No model files required (mocked dependencies)

- **Integration Tests**: Tagged with `@Tag("integration")`
  - Require actual ONNX models in `models/` directory
  - Run with `mvn verify -P integration-tests`
  - Cover end-to-end inference flows

- **Performance Tests**:
  - `PerformanceTrackerTest`: Validates metrics collection
  - Target: TTFT < 50ms, token intervals < 100ms

## Key Files & Locations

### FFM Implementation (Target for Development)
- Native C API wrappers would go in `src/main/java/com/badu/ai/onnx/ffm/`
- Core classes: `OrtMemoryInfo`, `OrtDeviceTensor`, `OrtIoBinding`
- JNI glue code alternative: `src/main/native/` (if JNI bridge used instead)

### Inference Engines
- `src/main/java/com/badu/ai/onnx/engine/`:
  - `InferenceEngine.java`: Core interface
  - `InferenceEngineFactory.java`: Engine selection logic
  - `ModelArchitecture.java`: Enum for T5/GPT/BART architectures

### GPU Backend
- `src/main/java/com/badu/ai/onnx/gpu/`:
  - `GpuAdapter.java`: Base interface
  - `CudaGpuAdapter.java`: NVIDIA CUDA configuration
  - `GpuDeviceManager.java`: Device enumeration and selection

### Configuration
- `src/main/java/com/badu/ai/onnx/config/`:
  - `ModelConfig.java`: Builder pattern for model settings
  - `GenerationConfig.java`: Sampling/search parameters
  - `ModelFileDiscovery.java`: ONNX file detection logic

### GenAI (LLM-Specific)
- `src/main/java/com/badu/ai/onnx/genai/internal/`:
  - `SimpleGenAI.java`: GenAI library wrapper
  - `Generator.java`: Token generation loop
  - `KVCache.java`: Key-value cache management

## Important Design Patterns

1. **Builder Pattern**: Used for `ModelConfig`, `GenerationConfig`
2. **Factory Pattern**: `InferenceEngineFactory`, `GpuAdapterFactory`
3. **Adapter Pattern**: GPU backend abstraction (`GpuAdapter` implementations)
4. **Singleton Pattern**: `OnnxInference.getInstance()`, `PerformanceTracker`
5. **Callback Pattern**: `TokenCallback` for streaming generation

## Code Style & Conventions

- **Java 21** language level (use records, pattern matching, text blocks where appropriate)
- **Lombok** annotations: Use `@Getter`, `@Builder`, `@Slf4j` for boilerplate reduction
- **SLF4J** for logging (configured to `slf4j-simple` in runtime)
- **Package structure**: Organize by feature (`gpu`, `genai`, `engine`, `config`) not layer
- **Null safety**: Validate inputs early, throw `IllegalArgumentException` for programmer errors
- **Resource management**: Always implement `AutoCloseable` for native resource wrappers

## Common Pitfalls to Avoid

1. **Memory Leaks**: Always call `close()` on FFM `MemorySegment` wrappers (GPU VRAM not GC'd)
2. **Thread Safety**: Inference engines are NOT thread-safe; synchronize access if multi-threaded
3. **Model File Paths**: Use absolute paths or verify working directory before loading models
4. **Execution Providers**: Verify CUDA/GPU availability before selecting GPU adapters (graceful CPU fallback)
5. **JNI Anti-Pattern**: Don't use `jlong` for FFM implementation—use `MemorySegment`

## References & Documentation

Project documentation in `docs/`:
- `Project Agenda.md`: Phase-by-phase FFM implementation plan
- `Optimizations.md`: Performance optimization methodologies (quantization, I/O binding, EP selection)
- `ONNX Runtime Java I_O Bindings.md`: Deep dive on I/O binding architecture and JNI vs FFM
- `Adding I_O Bindings Support.md`: Research summary on Java binding gaps and FFM prototype

External References:
- ONNX Runtime I/O Binding: https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html
- Project Panama FFM API (JEP 442): https://openjdk.org/jeps/442
- ONNX Runtime GenAI: https://github.com/microsoft/onnxruntime-genai

## Development Workflow

1. **Adding FFM Bindings**:
   - Generate C API bindings with `jextract` from `onnxruntime_c_api.h`
   - Create Java wrapper classes implementing `AutoCloseable`
   - Write unit tests mocking native calls
   - Validate with integration tests on actual CUDA devices

2. **Adding New Inference Engines**:
   - Implement `InferenceEngine` interface
   - Register in `InferenceEngineFactory`
   - Add model architecture detection logic in `ModelFileDiscovery`

3. **GPU Backend Support**:
   - Implement `GpuAdapter` for new execution provider
   - Add device enumeration in `GpuDeviceManager`
   - Configure `SessionOptions` with provider-specific options

4. **Performance Optimization**:
   - Profile with `PerformanceMetrics` collection
   - Target TTFT < 50ms for streaming UX
   - Optimize tokenization, encoder, and decoder stages independently
