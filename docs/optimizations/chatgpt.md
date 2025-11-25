Nice — I’ll summarize what’s possible, what’s implemented in ONNX Runtime today, and what’s missing / only available in native bindings (C++/C#/C) so you can plan a Java implementation (FFM / JNI) that gets the full set of LLM/LLM-serving optimizations you need for a fast summarizer in an adaptive RAG pipeline.

I split the research into (A) Execution Providers (EPs) and EP-specific optimizations, (B) ONNX Runtime features you can use from the runtime, (C) features known to be implemented in native bindings but often missing / incomplete in Java, and (D) a practical prioritized plan and concrete implementation suggestions (including how to add missing features to Java via Foreign Function & Memory or JNI). I cite official docs / authoritative sources for the most load-bearing claims.

---

# A — Execution Providers & provider-level optimizations

(Choose the EP that best matches your hardware; each EP has provider-specific knobs & fused kernels that matter for LLM performance.)

CPU

* oneDNN / DNNL (Intel/AMD CPU kernels): vectorized kernels, BF16/FP16 support (hardware allowing), graph fusions optimized for CPUs. Good choice for CPU-only inference; tune thread pools and memory arena. ([ONNX Runtime][1])

CUDA (NVIDIA GPUs)

* CUDA EP: standard path for NVIDIA. Optimizations include FP16 / mixed precision (float16), Tensor Cores usage, prepacking weights, memory preallocation, and use of TensorRT EP (if available) for further kernel fusion and INT8/TensorRT optimizations. Use I/O Binding to keep tensors resident on GPU to avoid host/device copies. ([ONNX Runtime][2])

TensorRT (NVIDIA) — often used as a separate EP or post-compile step

* Extremely useful for INT8 or FP16 kernel fusion and fast attention kernels; can yield large speedups for inference when models convert well to TRT engines. ([ONNX Runtime][3])

ROCm (AMD GPUs)

* ROCm EP for AMD GPUs; supports ROCm-enabled hardware and like CUDA benefits from device-resident memory and provider-specific kernels. ([ONNX Runtime][4])

Vulkan / DirectML / DML (Windows)

* Useful on integrated GPUs or some discrete GPUs where Vulkan / DirectML is the best path. Provider-specific fused kernels and driver interplay matter. ([Microsoft Learn][5])

CoreML / Metal (Apple)

* For Apple Silicon, CoreML/Metal EPs can be best path; they may offer FP16/BF16 speedups and are tightly integrated with system frameworks.

Recommendations:

1. If you have NVIDIA GPUs → CUDA + (where possible) TensorRT for maximum speed.
2. For multi-platform support: provide a PATH which picks best provider at runtime (ORT session option registration order). ([ONNX Runtime][3])

---

# B — ONNX Runtime features & optimizations (general)

These are features implemented inside ONNX Runtime (often accessible through session options, C/C++/C# APIs) that deliver big wins for LLM inference and summarization.

1. **Graph optimizations & operator fusions**

   * ONNX Runtime runs graph optimizations (eliminate dead nodes, fuse patterns, layout optimizations) at load time and can apply more aggressive transforms when GraphOptimizationLevel is set. This reduces op count and improves kernel fusion. ([ONNX Runtime][6])

2. **Quantization**

   * Post-training quantization: dynamic/static quantization to INT8 or INT4 (where supported), FP16 conversion. Quantize to reduce memory, maximize cache hits, and leverage provider INT8 kernels (TensorRT, etc.). Tools: onnxruntime quantization tools. ([ONNX Runtime][6])

3. **I/O Binding (zero-copy / device-resident buffers)**

   * Bind input/output buffers to the device prior to Run() to avoid host<->device copies per inference. Huge win when executing many short sequences (LLM token loop). I/O Binding is a major performance lever. ONNX docs say it's available in language bindings (examples in C++/Python/C#). ([ONNX Runtime][7])

4. **Memory allocators / arena allocator / preallocation**

   * Control memory allocation (arena on/off), custom allocators, and preallocate outputs. Reduces runtime malloc/free overhead and fragmentation. ([ONNX Runtime][6])

5. **Threading and execution mode**

   * `intra_op_num_threads` and `inter_op_num_threads`, plus `ExecutionMode` (sequential vs parallel) are important to tune on CPU. For GPU, minimize CPU thread overhead and prefer device parallelism. ([iot-robotics.github.io][8])

6. **ORT format / model packing / memory mapping**

   * Convert ONNX to ORT format (optimized, memory-mapped) to reduce model load times and memory overhead when supported. ([GitHub][9])

7. **KV-cache / past_key_values management & attention optimizations**

   * For autoregressive LLMs, efficient KV cache handling (keep KV on device across steps, avoid re-computation) is essential. GenAI library / strategies implement efficient KV management. ([GitHub][10])

8. **ONNX Runtime GenAI (generate API)**

   * A higher-level library (onnxruntime-genai) implements the generative loop: tokenization, logits processing, sampling, KV cache, and optimized loops — it yields much better performance out of the box for LLMs. It's a separate library but a reference implementation of many optimizations. ([GitHub][10])

---

# C — Features implemented natively but (practically) missing / incomplete in Java

Official docs say many features exist for *all* language bindings, but in practice Java has had gaps (bugs, missing wrappers) or lower-level native features are only exposed via C/C++/C# APIs. Key examples:

1. **I/O Binding support in Java**

   * Official docs assert I/O Binding is available in all language bindings; there are C++/C#/Python examples. However, community issues and GitHub threads show Java users hitting missing methods, API gaps, or bugs using large LLMs with Java bindings (some users report missing Java IoBinding convenience APIs or broken behavior for device binding). You should treat Java I/O Binding as *possible but fragile* — verify the exact ORT Java version you plan to use. If the Java API lacks a createIoBinding or equivalent, you can call the lower-level C API via FFM/JNI to implement it. ([ONNX Runtime][7])

2. **onnxruntime-genai integration for Java**

   * `onnxruntime-genai` provides the LLM generate loop and has releases that include shared libraries. Microsoft has moved toward exposing genai through a shared library that language bindings can call, but Java may lag behind in first-class binding or high-level helpers. You can still call the genai shared library from Java via FFM or JNI. ([GitHub][11])

3. **Provider-specific advanced features (TensorRT INT8 calibration, custom provider extensions)**

   * Some provider features or calibration tools are exposed only in native code or in provider tooling. Java clients may not expose all calibration/inspection utilities; the standard route is to run calibration & conversion offline (Python/C++) and load the optimized engine in Java runtime. ([ONNX Runtime][3])

4. **Custom ORT ops / native fused kernels**

   * Implemented generally at C++ level. Java can *use* custom ops if the provider/native shared lib is present, but building and registering custom ops requires native code. ([GitHub][9])

5. **Occasional Java binding bugs for very large models**

   * Community issues show Java API problems when loading very large models (memory, API limits); check the ORT GitHub issues for your version. (E.g. recent issues about loading a 20B model with Java that surfaced in 2025). ([GitHub][12])

---

# D — Prioritized plan for an ONNX-based summarizer (practical / minimal → aggressive optimizations)

Goals: single-turn text in → text out summarizer (fast, low latency, good quality). Prioritize robustness first, then optimize.

**Phase 0 — Baseline (get it working end-to-end)**

1. Choose a model already converted to ONNX for summarization (or convert/trim a transformer model with huggingface→ONNX).
2. Run inference with Java ONNX Runtime using the CPU EP (oneDNN) or CUDA EP if GPU available. Use the latest stable ORT Java package that matches your native libs. (This gets correctness & baseline latency.) ([FSEire][13])

**Phase 1 — Low effort / high win**

1. **Graph optimizations**: enable full graph optimization level when creating the session (SessionOptions.GraphOptimizationLevel = ALL). Test correctness vs float baseline. ([ONNX Runtime][6])
2. **Model dtype**: float16 (FP16) if your EP supports it — significantly reduces memory and increases throughput. Test for quality degradation. ([ONNX Runtime][2])
3. **Use ORT GenAI** as a reference for the inference loop (tokenization, KV management). If Java binding not present, call the genai shared lib via FFM/JNI or re-implement a simplified loop in Java using the ORT C API via FFM. ([GitHub][10])

**Phase 2 — Medium effort**

1. **I/O Binding** (big win for step-wise decoding):

   * Ensure inputs/outputs remain on device. If Java API exposes IoBinding correctly, use it. If not, implement IoBinding calls via FFM / JNI (call the C API: create IoBinding, OrtCreateTensorWithDataAsOrtValue for device memory, BindInput/BindOutput, SynchronizeBoundInputs/Outputs, then RunWithBinding). The ONNX C API exposes these primitives. ([ONNX Runtime][14])
2. **KV cache on device**: keep past_key_values resident on GPU and append new tokens using I/O Binding and device-resident tensors. Use genai patterns for efficient KV reuse. ([GitHub][10])
3. **Thread & allocator tuning**: disable superfluous CPU threads for GPU path, tune arena allocator as needed. ([iot-robotics.github.io][8])

**Phase 3 — Aggressive / advanced**

1. **Quantization**: static INT8 or INT4 (if supported by EP, TensorRT) for large models — substantial perf & memory win but needs calibration. Run calibration offline (Python tools) then load quantized model in Java. ([ONNX Runtime][6])
2. **TensorRT or provider-specific fused kernels**: use TensorRT for most aggressive graph fusion and fastest kernels (requires conversion and careful validation). ([ONNX Runtime][3])
3. **Use onnxruntime-genai shared lib**: either invoke the shared lib from Java (FFM/JNI) or reimplement its logic in Java but reuse the same performance techniques (logits processing, fused sampling, KV cache management). ([GitHub][11])

---

# E — How to add missing/native features to Java (FFM / JNI plan, recommended approach)

You mentioned using Foreign Function & Memory — good call. High-level options:

1. **Use the ONNX Runtime Java distribution that bundles native libs** (try latest). Check if the Java API already exposes IoBinding, OrtCreateTensorWithDataAsOrtValue, and Run with binding. If yes, use those. (Docs claim IOBinding exists in all bindings, but confirm on your ORT Java version first). ([ONNX Runtime][7])

2. **If Java API lacks a convenience wrapper for I/O Binding or GenAI:**

   * **FFM (Project Panama) approach** (preferred for modern Java when possible):

     * Load onnxruntime shared library (the same .so/.dll used by other bindings).
     * Bind the specific C API calls you need:

       * CreateIoBinding / OrtCreateIoBinding (C API wrapper for IoBinding)
       * OrtCreateTensorWithDataAsOrtValue (create OrtValue from native buffer)
       * BindInput / BindOutput / SynchronizeBoundInputs / SynchronizeBoundOutputs
       * RunWithBinding or SessionRunWithBinding
     * Use `MemorySegment` / direct `ByteBuffer` for native buffers pointing to device memory (for GPU you will likely need to interop with CUDA driver API to allocate device memory; for that you might also call CUDA driver functions from Java via FFM or use a small native helper lib).
     * Advantages: avoids JNI bent-over complexity and supports direct memory segments and safer signatures.
     * References: ONNX Runtime C API exposes the needed low-level functions; docs & header show `OrtApi` and io binding primitives. ([ONNX Runtime][14])

   * **JNI approach** (older, more boilerplate): write a small C/C++ shim that uses the ONNX C/C++ API to implement Java-callable functions for IoBinding and genai features. The shim can present a simple JNI API to Java (e.g., `createIoBinding(sessionPtr)`, `bindInputToDevice(bindingPtr, name, nativePtr, shape, deviceInfo)`, `runWithBinding(sessionPtr, bindingPtr)`). Advantage: easy to reuse C++ examples and provider code; slightly more glue but well-tested.

3. **Using onnxruntime-genai shared library from Java**

   * The genai project publishes shared libs (see releases). You can call the genai shared library functions for the generate loop from Java via FFM/JNI instead of reimplementing the whole loop; reuse the highly tuned genai logic for KV cache, sampling, and device management. ([GitHub][11])

4. **Hybrid approach**

   * Do heavy device memory management & I/O Binding in a small native helper (C/C++) and keep higher-level orchestration and tokenization in Java. This keeps the Java code clean while still getting the device-level zero-copy advantages.

---

# F — Concrete code / architecture sketch (high level)

(Use FFM or JNI to call C API; below is a conceptual flow — I won't paste low-level C signatures, but this is the architecture you should implement.)

1. Java starts and loads ONNX Runtime shared lib + genai shared lib (if used).
2. Create Session with preferred EPs appended in priority order (CUDA → TensorRT → CPU).
3. Convert model to ORT format / enable graph optimizations at load time.
4. If GPU:

   * Allocate device memory for inputs and outputs (via CUDA runtime or provider allocator).
   * Create OrtValue over preallocated device memory (via `OrtCreateTensorWithDataAsOrtValue`).
   * Create IoBinding object for the session (via C API or Java wrapper).
   * Bind inputs/outputs, call `RunWithBinding`.
   * Synchronize bound outputs (if needed).
5. For autoregressive decoding (summarization):

   * Keep KV cache OrtValues on device across loop iterations.
   * For each new token step: update input token buffer (device), call `RunWithBinding`, read logits (device), sample next token (prefer sampling on CPU only if needed; better: implement sampling on device or use logits->sample hybrid).
6. Optionally use genai shared lib instead of re-implementing the loop.

Caveat: device allocation & external memory passing may require integration with CUDA driver or provider allocators — this is why the C shim route is often easier: the shim uses provider allocators to create OrtValues that are natively compatible.

---

# G — Quick prioritized checklist (one-page)

1. Pick ORT Java + matching native shared libraries (same version).
2. Convert/quantize model offline (FP16 → FP16, try INT8 calibrate offline).
3. Enable GraphOptimizationLevel = ALL.
4. Use CUDA EP + TensorRT if NVIDIA GPU available.
5. Implement I/O Binding (FFM / JNI) if Java binding is missing or buggy.
6. Keep KV cache on device, use genai logic if possible.
7. Tune thread counts / arena allocator.
8. Benchmark: sequence_length, batch_size, latency per token; iterate.

(References for the above: ONNX Runtime EP docs, I/O Binding docs, GenAI repo, Graph optimizations docs). ([ONNX Runtime][1])

---

# H — What I recommend you do next (concrete short items)

1. Tell me which hardware you’ll target first (NVIDIA GPU? CPU only?) — I’ll produce the exact Java + native function list you’ll need and a small FFM/JNI shim code sketch for IoBinding + OrtValue creation mapped to device memory (I’ll include the specific C API function names and the Java FFM signatures).
   *If you prefer, I can skip asking and just produce the shim now for a typical CUDA + Java setup (I’ll assume Linux x86_64 with ONNX Runtime X.Y — I’ll pick a recent ORT version).*

2. Meanwhile: get one ORT Java release that matches your native runtime (same version), and pick a small summarization model converted to ONNX (or tell me which model you plan to use). I can prepare conversion & quantization scripts and a benchmarking harness (Java + minimal C shim) for measuring per-token latency.

---

# Sources (key citations)

* ONNX Runtime Execution Providers overview. ([ONNX Runtime][1])
* I/O Binding (official performance doc; I/O binding examples & rationale). ([ONNX Runtime][7])
* onnxruntime-genai (repo & releases — generate API, KV cache, optimizations). ([GitHub][10])
* Graph optimizations & model transforms in ORT. ([ONNX Runtime][6])
* C API / OrtApi IoBinding primitives (SynchronizeBoundInputs, etc.). ([ONNX Runtime][14])

---

If you want, I’ll **immediately** produce:

* (A) a minimal JNI or FFM shim sketch that exposes: `createIoBinding(sessionPtr)`, `bindInputToDevice(...)`, `bindOutputToDevice(...)`, `runWithBinding(...)`, and `createOrtValueFromDevicePointer(...)`, **or**
* (B) a design + code sketch that uses `onnxruntime-genai` shared lib from Java (FFM) and shows how to call the generate API from Java,

— pick (A) or (B) and tell me target OS + target ORT version (or I can pick a commonly used target like Linux x86_64 and ORT 1.21/1.22). I’ll produce the code sketch and the precise C API function names you’ll call (no waiting).

[1]: https://onnxruntime.ai/docs/execution-providers/?utm_source=chatgpt.com "ONNX Runtime Execution Providers"
[2]: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html?utm_source=chatgpt.com "NVIDIA - CUDA"
[3]: https://onnxruntime.ai/docs/build/eps.html?utm_source=chatgpt.com "Build with different EPs | onnxruntime"
[4]: https://onnxruntime.ai/docs/execution-providers/ROCm-ExecutionProvider.html?utm_source=chatgpt.com "ROCm Execution Provider - AMD"
[5]: https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/supported-execution-providers?utm_source=chatgpt.com "Supported execution providers in Windows ML"
[6]: https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html?utm_source=chatgpt.com "Graph Optimizations in ONNX Runtime"
[7]: https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html?utm_source=chatgpt.com "I/O Binding | onnxruntime"
[8]: https://iot-robotics.github.io/ONNXRuntime/docs/performance/tune-performance.html?utm_source=chatgpt.com "ONNX Runtime Performance Tuning"
[9]: https://github.com/microsoft/onnxruntime?utm_source=chatgpt.com "microsoft/onnxruntime: ONNX Runtime: cross-platform ..."
[10]: https://github.com/microsoft/onnxruntime-genai?utm_source=chatgpt.com "microsoft/onnxruntime-genai: Generative AI extensions for ..."
[11]: https://github.com/microsoft/onnxruntime-genai/releases?utm_source=chatgpt.com "Releases · microsoft/onnxruntime-genai"
[12]: https://github.com/microsoft/onnxruntime/issues/26476?utm_source=chatgpt.com "loading model gpt-oss-20b throws 'ORT_INVALID_GRAPH ..."
[13]: https://fs-eire.github.io/onnxruntime/docs/get-started/with-java.html?utm_source=chatgpt.com "Java | onnxruntime - GitHub Pages"
[14]: https://onnxruntime.ai/docs/api/c/struct_ort_api.html?utm_source=chatgpt.com "OrtApi Struct Reference"

