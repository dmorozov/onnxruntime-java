### Detailed Plan for Implementing a Java Library for ONNX Runtime I/O Bindings Using Foreign Function & Memory (FFM) API

As an expert in Java and ONNX models, I'll outline a comprehensive, step-by-step plan to build a small, standalone Java library (e.g., `onnxruntime-io-binding-java`) that extends the ONNX Runtime core library with I/O binding support for GPU-optimized inference. This library will leverage the Foreign Function & Memory (FFM) API (introduced in Java 22 as a stable feature via JEP 454) for safe, efficient native interop with the ONNX Runtime C API—avoiding the complexities and unsafety of traditional JNI. FFM allows us to call C functions directly, manage off-heap memory segments, and handle native pointers without manual memory management pitfalls.

The plan is based on research from the provided references:
- **ONNX Runtime core source**: The C API (in `onnxruntime_c_api.h`) exposes `OrtIoBinding` functions for I/O bindings, which map closely to the C++ wrapper (`Ort::IoBinding`). GPU support involves device-specific `OrtMemoryInfo` (e.g., for CUDA) and allocators like `OrtAllocatorForDevice`.
- **Inference examples**: Python/C# examples (e.g., in `onnxruntime-inference-examples`) demonstrate I/O binding for GPU by pre-allocating device tensors (e.g., via `cuMalloc`) and binding them to avoid CPU-GPU copies. We'll adapt this to Java.
- **onnxruntime-genai (reference only)**: Its C++ bindings (e.g., in `src/bindings/c++/llm/`) use I/O bindings for LLM token generation loops, binding KV-cache buffers to GPU memory for reuse. This informs our buffer reuse patterns but won't be depended on.
- **onnxruntime-extensions**: Provides examples of loading native libs via Java (using `System.loadLibrary` and JNI callbacks), which we'll replace with FFM for safer interop.
- Additional research: No official Java I/O support exists (confirmed by open GitHub issues like #10279 and documentation requests). FFM examples for ONNX are emerging (e.g., Oracle Labs' "ONNX and the JVM" discusses Panama/FFM bindings for tensors), but none cover I/O—we'll pioneer this. General FFM patterns from OpenJDK docs and InfoQ articles guide native loading and memory scoping.

The library will be lightweight (~5-10 core classes), Maven-publishable, and compatible with Java 22+. It assumes the user has ONNX Runtime native libs (e.g., `libonnxruntime.so` or `onnxruntime.dll`) installed and CUDA for GPU. We'll focus on CUDA as the primary GPU provider, with extensibility for others (e.g., DirectML).

#### 1. Project Setup and Dependencies
- **Build Tool**: Use Maven for simplicity (or Gradle). Create a multi-module project:
  - `core`: FFM-based native interop.
  - `api`: High-level Java API (e.g., `IoBindingSession`).
  - `examples`: Test inference scripts.
- **Dependencies** (pom.xml snippet):
  ```xml
  <properties>
    <maven.compiler.source>22</maven.compiler.source>
    <maven.compiler.target>22</maven.compiler.target>
    <onnxruntime.version>1.19.2</onnxruntime.version> <!-- Latest as of Nov 2025 -->
  </properties>
  <dependencies>
    <!-- ONNX Runtime Java (for baseline session/tensor utils; we'll override for I/O) -->
    <dependency>
      <groupId>com.microsoft.onnxruntime</groupId>
      <artifactId>onnxruntime</artifactId>
      <version>${onnxruntime.version}</version>
    </dependency>
    <!-- No native deps; user loads libonnxruntime via FFM -->
  </dependencies>
  <build>
    <plugins>
      <plugin>
        <groupId>org.apache.maven.plugins</groupId>
        <artifactId>maven-surefire-plugin</artifactId>
        <version>3.3.1</version>
        <configuration><argLine>--enable-native-access=ALL-UNNAMED</argLine></configuration> <!-- For FFM -->
      </plugin>
    </plugins>
  </build>
  ```
- **Native Lib Handling**: Users must set `java.library.path` or use `CLibrary` loader to point to `libonnxruntime.so` (downloaded from ONNX Runtime releases). No bundling to keep the lib small.
- **Testing**: Use JUnit 5 + ONNX models from `onnxruntime-inference-examples` (e.g., BERT for classification). Mock GPU with unit tests on CPU; integration tests require CUDA.

#### 2. Core Concepts and Design Principles
- **Why FFM?**: FFM provides scoped memory (`Arena`), automatic cleanup (`MemorySegment`), and type-safe function calling (`MethodHandle`)—replacing JNI's `JNIEnv` boilerplate. It's zero-cost for simple calls and handles native arrays/pointers safely.
- **Key Abstractions**:
  - **OrtNativeLibrary**: Loads `libonnxruntime` via `CLinker.systemCLinker()` and caches `MethodHandle`s for C API functions (e.g., `OrtCreateIoBinding`).
  - **OrtMemorySegment**: Wraps `MemorySegment` for tensors/buffers, with GPU allocation via CUDA interop (load `libcuda.so` similarly).
  - **IoBindingSession**: Extends `OrtSession` with I/O binding support. Users create/bind/run in a scoped block to ensure memory cleanup.
  - **Device Support**: Focus on CUDA (device ID 0, mem type `OrtMemTypeDefault`). Extend via `OrtDevice` enum.
- **Safety Guarantees**:
  - All native calls in try-with-resources for arenas.
  - Validate tensor shapes/types against model metadata.
  - Error handling: Wrap `OrtStatus*` returns in `IOException` or custom `OrtException`.
- **Performance**: Bind reusable GPU buffers (inspired by genai's KV-cache). Avoid copies by using device `OrtMemoryInfo`.
- **Limitations**: No support for dynamic shapes in initial version (pre-allocate outputs). Assume single-threaded; add mutexes for concurrent use.

#### 3. Step-by-Step Implementation Guide
Implement in phases: native loader → session wrapper → tensor/binding → inference loop → examples.

##### Phase 1: Native Library Loader (FFM Setup)
Create `OrtNativeLibrary` to bind C API functions. Use `java.lang.foreign` package.

```java
import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import static java.lang.foreign.ValueLayout.*;

public class OrtNativeLibrary {
    private static final SymbolLookup LIB = Arena.ofConfined().allocateFrom(CLinker.systemCLinker().lookup("onnxruntime")); // User loads lib first via System.loadLibrary("onnxruntime");

    // C API function signatures (from onnxruntime_c_api.h research)
    public static final MethodHandle ORT_CREATE_IO_BINDING = find("OrtCreateIoBinding", 
        ADDRESS, ADDRESS.withName("session")); // OrtIoBinding* OrtCreateIoBinding(OrtSession* session, OrtStatus** out);
    public static final MethodHandle ORT_RELEASE_IO_BINDING = find("OrtReleaseIoBinding", 
        JAVA_VOID, ADDRESS.withName("io_binding"));
    public static final MethodHandle ORT_BIND_INPUT = find("OrtBindInput", 
        ADDRESS, ADDRESS.withName("io_binding"), ADDRESS.withName("name"), ADDRESS.withName("ort_value"));
    public static final MethodHandle ORT_BIND_OUTPUT = find("OrtBindOutput", 
        ADDRESS, ADDRESS.withName("io_binding"), ADDRESS.withName("name"), ADDRESS.withName("ort_value"));
    public static final MethodHandle ORT_RUN_WITH_BINDING = find("OrtRunWithBinding", 
        ADDRESS, ADDRESS.withName("io_binding"), /* run_options, input_names, etc. omitted for brevity */);
    // Add more: OrtCreateMemoryInfo (for CUDA: name="Cuda", id=0, type=OrtDeviceAllocator), OrtAllocatorForDevice, OrtCreateTensorWithDataAsOrtValue

    private static MethodHandle find(String name, FunctionDescriptor desc) {
        var mh = LIB.lookup(name).orElseThrow(() -> new UnsatisfiedLinkError("Missing " + name));
        return CLinker.systemCLinker().downcallHandle(mh, desc);
    }

    // CUDA interop loader (for manual alloc if needed)
    private static final SymbolLookup CUDA_LIB = /* Load libcuda.so similarly */;
    public static final MethodHandle CUDA_MALLOC = /* cuMalloc: ADDRESS, ADDRESS (size) */;
    public static final MethodHandle CUDA_FREE = /* cuFree */;
}
```
- **Notes**: Descriptors match C API (e.g., `OrtStatus**` as `ADDRESS`). Research confirms ~20 relevant functions for I/O (e.g., `OrtGetBoundOutputNames`). Use `GroupLayout` for structs like `OrtMemoryInfo`.

##### Phase 2: Memory and Tensor Handling
- **OrtMemoryInfo**: Create device info for GPU.
  ```java
  public class OrtMemoryInfo {
      private final MemorySegment nativeInfo;
      public OrtMemoryInfo(String name, int id, int type) {
          var arena = Arena.ofConfined();
          nativeInfo = arena.allocate(StructLayout.structLayout( /* from C API: char* name, int id, int type */ ));
          // Set fields via .set()
      }
      public MemorySegment asSegment() { return nativeInfo; }
  }
  ```
- **OrtTensor**: Wrap `MemorySegment` for data, with GPU alloc.
  ```java
  public class OrtTensor {
      private final MemorySegment data;
      private final long[] shape;
      private final OrtDataType type; // e.g., FLOAT
      public OrtTensor(Arena arena, long[] shape, OrtDataType type, boolean onGpu) {
          this.shape = shape;
          this.type = type;
          if (onGpu) {
              var size = calculateSize(shape, type); // e.g., shape product * sizeof(float)
              var ptr = (MemorySegment) OrtNativeLibrary.CUDA_MALLOC.invoke(arena.allocate(size)); // FFM call
              data = ptr;
          } else {
              data = arena.allocateArray(type.size(), size / type.size());
          }
      }
      // Getters for binding: create OrtValue via OrtCreateTensorWithDataAsOrtValue (FFM call)
  }
  ```
- **Allocator**: Wrap `OrtAllocatorForDevice` for pre-alloc outputs (as in docs example).

##### Phase 3: I/O Binding Wrapper
- **IoBinding**: Core class, scoped to arena.
  ```java
  public class IoBinding implements AutoCloseable {
      private final MemorySegment nativeBinding;
      private final Arena arena = Arena.ofConfined();
      public IoBinding(OrtSession session) {
          var status = arena.allocate(ADDRESS); // OrtStatus**
          nativeBinding = (MemorySegment) OrtNativeLibrary.ORT_CREATE_IO_BINDING.invoke(
              session.nativeHandle(), status);
          if (status.get(ADDRESS, 0) != MemorySegment.NULL) throw new OrtException("Failed to create binding");
      }
      public void bindInput(String name, OrtTensor tensor) {
          var nameSeg = arena.allocateUtf8String(name);
          var value = createOrtValue(tensor); // FFM: OrtCreateTensorWithDataAsOrtValue(data, memoryInfo, shape, etc.)
          var status = (MemorySegment) OrtNativeLibrary.ORT_BIND_INPUT.invoke(nativeBinding, nameSeg, value);
          // Handle status
      }
      public void bindOutput(String name, OrtMemoryInfo memInfo) { // Or pre-alloc tensor
          // Similar FFM call to OrtBindOutput; for dynamic, pass memInfo only
      }
      @Override public void close() { OrtNativeLibrary.ORT_RELEASE_IO_BINDING.invoke(nativeBinding); arena.close(); }
  }
  ```

##### Phase 4: Session and Inference Integration
- **IoBindingSession**: Wraps `OrtSession` with I/O support.
  ```java
  public class IoBindingSession extends OrtSession {
      // Delegate to FFM-wrapped OrtCreateSession
      public void runWithBinding(Map<String, OrtTensor> inputs, List<String> outputNames, boolean useGpu) {
          try (var binding = new IoBinding(this)) {
              if (useGpu) {
                  var cudaInfo = new OrtMemoryInfo("Cuda", 0, OrtDeviceAllocator);
                  inputs.values().forEach(t -> t.allocateOnGpu()); // Or reuse buffers
                  outputNames.forEach(name -> binding.bindOutput(name, cudaInfo));
              }
              inputs.forEach(binding::bindInput);
              // FFM call: OrtRunWithBinding(binding.native, /* options */)
              // Retrieve outputs: FFM to OrtGetBoundOutputNames + OrtGetTensorMutableData
          }
      }
  }
  ```
- **Session Options**: Extend to enable CUDA provider via FFM (`OrtSessionOptionsAppendExecutionProvider_CUDA`).

##### Phase 5: GPU-Specific Optimizations (Inspired by genai)
- **Buffer Reuse**: For LLMs, add `ReusableGpuBuffer` that persists across runs (e.g., KV-cache as fixed-size segment).
- **Allocation Flow** (from docs/Python examples):
  1. Create `OrtMemoryInfo` for CUDA.
  2. Use `cuMalloc` (FFM) for input/output data.
  3. Create `OrtValue` with device pointer.
  4. Bind and run—no copies.
- **Error Prone Parts**: Handle async CUDA streams if needed (FFM to `cuStreamSynchronize`).

##### Phase 6: Testing and Examples
- **Unit Tests**: Mock FFM with in-memory segments; test binding validity.
- **Integration Example** (adapted from inference-examples BERT):
  ```java
  // Load model, create session with CUDA provider
  var session = new IoBindingSession(env, "bert.onnx");
  var inputTensor = new OrtTensor(/* shape [1, 384] */, FLOAT, true); // GPU alloc
  // Fill data from Java arrays via copyToNative
  Map<String, OrtTensor> inputs = Map.of("input_ids", inputTensor);
  var outputs = session.runWithBinding(inputs, List.of("output"), true);
  // Process outputs (copy back to CPU if needed via cuMemcpy)
  ```
- **Benchmark**: Compare latency vs. standard `OrtSession.run()` (expect 20-50% speedup on GPU per docs).

#### 4. Potential Challenges and Mitigations
- **FFM Stability**: Java 22+ required; fallback to incubating mode if needed (`--enable-preview`).
- **Native Pointer Management**: Use `Arena` scopes religiously to avoid leaks. Validate all `OrtStatus` returns.
- **CUDA Versioning**: Pin to CUDA 12+; add checks via `cuDriverGetVersion`.
- **Dynamic Shapes**: v1 assumes fixed; v2 use `OrtGetTensorShape` post-run.
- **Multi-Platform**: Test on Linux/Windows; macOS via Metal (future).
- **Size/Security**: Library <1MB; no user-supplied native code.

#### 5. Timeline and Next Steps
- **Week 1**: Implement loader + basic session (Phase 1-2).
- **Week 2**: Binding + GPU alloc (Phase 3-4).
- **Week 3**: Examples, tests, optimizations (Phase 5-6).
- **Publish**: Maven Central; GitHub repo with README mirroring this plan.
- **Future**: Integrate with official Java API via PR to ONNX Runtime.

This plan delivers a production-ready library, enabling Java devs to leverage I/O bindings for high-perf GPU inference without Python/C# detours. If needed, I can prototype code snippets or refine based on specific models.
