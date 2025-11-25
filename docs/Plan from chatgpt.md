# Quick findings (evidence)

* ONNX Runtime provides an **I/O binding** mechanism to place inputs/outputs on device and pre-allocate outputs for better GPU performance — documents and examples exist for C++, Python, and C#. ([ONNX Runtime][1])
* The official Java binding exists but (today) does **not** expose the high-level IO binding surface that other languages do; there are community/issue requests noting this gap. (This is the motivation for the project.) ([ONNX Runtime][2])
* ONNX Runtime source (C/C++) is on GitHub and includes C APIs and headers you must consult when mapping to FFM. Use it as the canonical reference. ([GitHub][3])
* Java’s FFM API (JEP 454 et al.) is the modern safe way to call native code from Java (avoid JNI), and is suitable for this task (Java 21+ / Java 22+ recommended). ([OpenJDK][4])

---

# Goals for the library

1. Provide a small, idiomatic Java API that mirrors the ONNX I/O binding capabilities: bind inputs/outputs to host or device memory; run with binding; access bound outputs (optionally keep them on device); copy outputs to CPU.
2. Implement the native calls using **FFM** (no JNI), so the library is "pure Java" code that dynamically links to the ONNX Runtime shared library at runtime.
3. Keep the implementation small and focused so it can be shipped as a tiny artifact (pom) plus the prebuilt native ONNX shared libs (or rely on the user to provide them).

---

# High-level architecture

1. **Java Layer (api package)** — user-facing API:

   * `OrtIoBindingFFM` (main class) — high-level wrapper implementing methods:

     * `bindInput(String name, OrtValueOrBuffer buffer, DeviceSpec device)`
     * `bindOutput(String name, MemoryDescriptor desc, DeviceSpec device)`
     * `bindOutputToDevice(...)`
     * `runWithBindings() : Map<String, OrtValue>`
     * `getBoundOutputValues(...)`
     * `copyOutputsToCpu()`
   * Small value types: `DeviceSpec` (device name, deviceType, deviceId), `OrtValueRef` (opaque handle), `MemoryDescriptor` (shape, datatype).
   * `OrtSessionFFM` (light wrapper) to couple with existing `ai.onnxruntime.OrtSession` or the C session pointer.

2. **FFM Native Layer (internal package)** — direct FFM bindings mapped to the ONNX C API functions:

   * `OrtApiFFM` — class that uses `Linker`/`SymbolLookup` to get downcall `MethodHandle`s to C functions such as:

     * `OrtCreateIoBinding(OrtSession*, OrtIoBinding**)` (names to be confirmed from onnxruntime C header)
     * `OrtIoBinding_BindInput(...)`
     * `OrtIoBinding_BindOutput(...)`
     * `OrtIoBinding_GetBoundOutputValues(...)`
     * `OrtRunWithBinding(...)` / `OrtRun` variant that accepts IOBinding
     * `OrtApiRelease/OrtReleaseIoBinding`
   * `NativeMemory` utilities to allocate `MemorySegment`s that represent device pointers or pinned host buffers.

3. **Resource management**

   * Use `ResourceScope`/`Arena` (ScopedMemory) to ensure native memory is freed deterministically.
   * The Java wrapper objects implement `AutoCloseable` and make sure to call release functions (`OrtReleaseIoBinding`) and close ResourceScopes.

4. **Integration with existing com.microsoft.onnxruntime Java package**

   * Option A: Keep the library completely separate and accept/consume `OrtSession` handles (via reading the native pointer inside OrtSession or by creating session via FFM too).
   * Option B: Offer a drop-in extension that cooperates with the existing `ai.onnxruntime` Java classes by utilizing the same native `OrtSession` pointer (requires reading the package internals or providing a companion `OrtSessionFFM#createFromOrtSession()` method).
   * Prefer Option B for ergonomics but design code so it can also operate directly using the C API.

---

# Mapping ONNX Runtime native API → FFM

**Steps to produce accurate function bindings:**

1. Open `onnxruntime/include/onnxruntime_c_api.h` (and related headers) in the ONNX repo and confirm C function names and signatures. (This repo is authoritative.) ([GitHub][3])

2. For each needed API, create a `MethodHandle` downcall using the FFM linker:

   * Example pattern:

     * Use `SymbolLookup.loaderLookup()` (or `CLinker.systemLookup()` / `Linker.nativeLinker`) to find `OrtCreateIoBinding`.
     * Build a `FunctionDescriptor` that matches the C signature (pointer types -> `Addressable` / `MemoryAddress` / `MemorySegment` in FFM; ints -> `C_INT`, `C_LONG` etc).
     * Obtain a `MethodHandle` via `Linker.getDowncallHandle(symbol, functionDescriptor)`.

3. **Example (informal)** — creating a downcall for `OrtCreateIoBinding(OrtSession* session, OrtIoBinding** out)` (confirm exact signature in headers):

```java
// pseudo-code — check the exact types and function name from headers
SymbolLookup lookup = SymbolLookup.loaderLookup(); 
Linker linker = Linker.nativeLinker();
MemoryAddress symbol = lookup.lookup("OrtCreateIoBinding").orElseThrow();
FunctionDescriptor fd = FunctionDescriptor.of(C_POINTER, C_POINTER, C_POINTER); // return OrtStatus*? (C_POINTER)
MethodHandle mh = linker.downcallHandle(symbol, fd);
```

You must confirm the real return type (many ORT functions return `const OrtApi*` or `OrtStatus*`) and adjust descriptors accordingly.

> **Important:** always read `onnxruntime_c_api.h` for exact types and ownership semantics. The repo has the headers you need. ([GitHub][3])

---

# Memory layout and OrtValue handling

ONNX Runtime uses `OrtValue` objects and has C APIs to create `OrtValue` that wrap existing memory (e.g., `CreateTensorWithDataAsOrtValue`). Two key approaches:

1. **Host-backed tensors** — create an `OrtValue` that views a pinned host buffer:

   * Allocate a native (off-heap) `MemorySegment` and fill it with host data, then call `CreateTensorWithDataAsOrtValue` (C API) with that pointer. Use FFM to call that C API.
   * Use `ResourceScope` so Java can free/close the buffer when appropriate.

2. **Device-backed binding** — for device memory (CUDA pointer / VA memory):

   * ONNX Runtime supports binding device pointers via `OrtIoBinding_BindInputToDevice` / `BindOutputToDevice` APIs (see docs). You will need to pass the device pointer and device type/ID per the C header. For CUDA, that might be a `void*` GPU pointer or a special `OrtMemoryInfo` describing device location.
   * The Java side must only pass opaque native pointers (as `MemoryAddress`) and not dereference them.

**Key FFM types:**

* Java `MemorySegment` for buffers.
* `MemoryAddress` for raw native pointers.
* `ResourceScope` / `Arena` for lifetime management.
* `MemoryLayout` only if you need to construct structs (like `OrtMemoryInfo`); alternatively, create helper allocation helpers to fill the struct in native memory.

---

# API design (suggested)

Public API (concise):

```java
public class OrtIoBindingFFM implements AutoCloseable {
    public OrtIoBindingFFM(OrtSession session) { /* wrap/obtain native session pointer */ }
    public void bindInput(String name, MemorySegment hostBuffer, long[] shape, OnnxTensorElementType type, DeviceSpec device) { ... }
    public void bindOutput(String name, long[] shape, OnnxTensorElementType type, DeviceSpec device) { ... }
    public Map<String, OrtValueRef> runWithBindings(Map<String,String> runOptions) { ... }
    public Map<String, MemorySegment> getBoundOutputCpuValues() { ... }
    public void copyOutputsToCpu() { ... }
    @Override public void close() { /* release native IOBinding */ }
}
```

Design notes:

* `OrtValueRef` is an opaque wrapper for a native `OrtValue*` pointer (stored as `MemoryAddress` or `MemorySegment`).
* Keep API surface small and synchronous (first pass). Add async later if needed.

---

# Implementation steps (detailed)

1. **Familiarize & identify native functions**

   * Inspect `onnxruntime/include/onnxruntime_c_api.h` and I/O binding docs for the exact functions and signatures (e.g., `CreateIoBinding`, `BindInput`, `BindOutput`, `OrtRunWithBinding` variants). Use these signatures in FFM descriptors. ([GitHub][3])

2. **Project skeleton**

   * Create Gradle/Maven project `onnxruntime-io-binding-ffm`.
   * Modules:

     * `api` — user API and wrappers
     * `ffm` — low-level FFM bindings (internal)

3. **Implement native loader**

   * Locate ONNX Runtime native shared libs at runtime (LD_LIBRARY_PATH / PATH etc). Provide an environment variable override (e.g., `ORT_NATIVE_LIB_PATH`).
   * Use `System.loadLibrary` to load the ONNX shared library early (so `SymbolLookup.loaderLookup()` can find symbols); or use `SymbolLookup` techniques that accept the library handle.

4. **Create FFM bindings**

   * For each native function:

     * Create `MethodHandle` via `Linker.nativeLinker().downcallHandle(...)`.
     * Wrap that handle in a typed Java method that marshal/unmarshal `MemorySegment` / Java arrays → native representations.
   * Provide helpers:

     * `createOrtIoBinding(MemoryAddress sessionPtr) -> MemoryAddress ioBindingPtr`
     * `bindInputToDevice(ioBindingPtr, name, ortValuePtr, deviceInfoPtr)`
     * `bindOutput(ioBindingPtr, name, type, shape, deviceInfoPtr)`
     * `runWithIoBinding(ioBindingPtr)`
     * `getBoundOutputValues(ioBindingPtr, outArrayPtr)`

5. **Memory management**

   * Always use `ResourceScope` when allocating segments.
   * Document that any `MemorySegment` passed into `bindInput` must be kept alive until `runWithBindings` completes (or the library will copy it or take ownership depending on chosen semantics).

6. **Interoperability with `ai.onnxruntime` Java package**

   * Two options:

     * Extract native pointer from `ai.onnxruntime` `OrtSession` (requires either reflection or the existing Java package to expose `getNativeHandle()` API). If not accessible, provide an API to create an `OrtSession` from your FFM layer and then use it.
     * Initially implement a self-contained `OrtSessionFFM` that opens models using the C API and can be used by the new I/O binding wrapper. Later integrate with com.microsoft.onnxruntime Java objects.

7. **Testing & benchmarks**

   * Unit tests to verify:

     * `bindInput` with host memory works and inference returns expected values.
     * `bindOutput` preallocates device memory and `getBoundOutputValues` returns device pointers.
     * `copyOutputsToCpu` moves data back and accessible from Java.
   * Performance tests:

     * Measure end-to-end throughput with and without I/O binding for a representative model (LLM KV cache or transformer ops).
     * Compare against Python/C# implementations of IO binding using the same ONNX Runtime and providers.

8. **Documentation**

   * Usage guide, minimal sample that runs a model on CUDA using I/O binding.
   * Document required Java version (Java 21+ or 22+ recommended) and required ONNX Runtime native build (version compatibility).

---

# Example: FFM sketch (concrete-ish)

> **WARNING**: below is *illustrative* code. You **must** verify C function names and exact signatures in `onnxruntime_c_api.h` and adapt descriptors. Use the repo headers as source-of-truth. ([GitHub][3])

```java
// ffm/OrtApiFFM.java (sketch)
import java.lang.foreign.*;
import java.lang.invoke.*;

public final class OrtApiFFM {
    private final Linker linker = Linker.nativeLinker();
    private final SymbolLookup lookup = SymbolLookup.loaderLookup();
    private final MethodHandle createIoBindingMH;
    private final MethodHandle ioBindingBindInputMH;
    // ... other MHs

    public OrtApiFFM() {
        MemoryAddress cfn = lookup.lookup("OrtCreateIoBinding").orElseThrow();
        // Suppose OrtCreateIoBinding returns OrtStatus* and has signature (OrtSession*, OrtIoBinding**)
        FunctionDescriptor fd = FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS);
        createIoBindingMH = linker.downcallHandle(cfn, fd);
        // similarly create handles for BindInput, RunWithBinding, etc.
    }

    public MemoryAddress createIoBinding(MemoryAddress sessionPtr, ResourceScope scope) throws Throwable {
        MemorySegment out = MemorySegment.allocateNative(ValueLayout.ADDRESS, scope);
        var status = (MemoryAddress) createIoBindingMH.invoke(sessionPtr, out.address());
        // check status for errors (call OrtGetErrorMessage via FFM)
        return out.get(ValueLayout.ADDRESS, 0); // returns OrtIoBinding*
    }

    // wrappers for bindInput, bindOutput...
}
```

Key points:

* Use `ValueLayout.ADDRESS`/`C_POINTER` equivalents for pointers.
* Many ORT functions return `OrtStatus*`; if non-null, you must fetch the error message and throw a Java exception.
* Use `ResourceScope` so allocations are deterministically freed.

---

# Testing checklist

* Unit tests that run with CPU provider (no GPU) — verifies logic and FFM correctness on CI.
* Integration tests on GPU (CUDA) provider — verifies device pointer passing & outputs-in-device semantics.
* Cross-language comparison tests: same model run from Python with IO Binding vs Java-FFM IO binding — outputs must match.
* Stress tests: repeated runs to confirm no leaks (native memory, device memory).

---

# Build & distribution

* Build: Maven/Gradle Java artifact that depends on `org.openjdk:panama-ffm` only if needed; for Java 21+ you don’t need external artifacts, use the JDK FFM API.
* Native dependency: the ONNX Runtime shared libraries must be installed and discoverable (LD_LIBRARY_PATH / PATH). Provide docs and small scripts to download matching ONNX Runtime builds (for Linux x86_64, aarch64, Windows).
* Packaging: publish the Java API jar to Maven Central (or private repo). Do **not** bundle ONNX native libs; instead provide instructions and optional helper `download-native.sh`.

---

# Risks & mitigations

1. **Function signatures mismatch / header changes** — always parse and validate the header `onnxruntime_c_api.h` at build time or maintain a small codegen step that produces FFM descriptors from the header. Mitigation: automated test against the ONNX runtime native library. ([GitHub][3])

2. **FFM API availability / Java compatibility** — FFM evolved across Java versions; choose a stable Java baseline (Java 21 / 22 LTS) and document requirement. Mitigation: gate features and fall back to JNI if absolutely necessary. ([OpenJDK][4])

3. **Device pointer ownership / lifetime complexity** — device memory must not be freed while ORT is using it. Mitigation: clearly document ownership semantics; use explicit `close()` and ResourceScopes; provide copy semantics when user buffers are transient.

4. **Symbol lookup across platforms** — `SymbolLookup.loaderLookup()` works when native lib is loaded. Mitigation: provide robust native loader and fallbacks; check `dlopen` name per OS.

5. **Thread-safety** — ONNX runtime sessions may or may not be thread-safe for some operations. Mitigation: follow ONNX runtime recommended session options and concurrency patterns.

---

# Minimal roadmap (no time estimates)

1. Inspect `onnxruntime_c_api.h` → list required functions and exact signatures. (FFM descriptors depend on this.) ([GitHub][3])
2. Prototype FFM `MethodHandle` lookup for 2 functions (`OrtCreateIoBinding`, `OrtIoBinding_BindInput`) and call them against a locally-built ONNX Runtime to verify downcall works.
3. Implement wrappers for creating `OrtValue` from host `MemorySegment` and binding it. Add tests on CPU.
4. Extend to device-binding API and test with CUDA provider (requires CUDA-enabled ONNX build).
5. Add higher-level Java API, docs, and integration tests.
6. Publish artifact + sample apps (CPU & GPU samples).

---

# References & links (use these for exact signatures & examples)

* ONNX Runtime I/O Binding docs (concept + code in C++/Python/C#). ([ONNX Runtime][1])
* ONNX Runtime Java getting started / API docs. ([ONNX Runtime][2])
* ONNX Runtime repository (C headers and native code) — canonical source for function names and signatures. ([GitHub][3])
* OrtIoBinding class docs (C# API) — shows semantics to mirror in Java. ([ONNX Runtime][5])
* Microsoft `onnxruntime-genai` repo — reference implementation and examples of I/O binding usage in C++/Python/C#. Use it for correctness and semantics, but **do not** depend on it as a compiled dependency (as you said). ([GitHub][6])
* JEP 454 / Project Panama (FFM API) — recommended Java API to use. ([OpenJDK][4])
* Community notes / issues indicating missing Java I/O binding (motivation). ([GitHub][7])

---

# Final notes & next steps (practical)

* **Next technical step you can do immediately**: open `onnxruntime/include/onnxruntime_c_api.h` from the ONNX repo and paste (or inspect) the exact function signatures for the `IoBinding` related functions (`OrtCreateIoBinding`, `OrtIoBinding_BindInput`, `OrtIoBinding_BindOutput`, `OrtRun` with binding, `GetBoundOutputValues`, etc.). Once you have the exact C signatures I will produce copy-paste-ready FFM `FunctionDescriptor` and `MethodHandle` code for each function and a small runnable Java example that uses `OrtIoBindingFFM` on CPU.

If you want, I can:

* produce the **first concrete FFM Java file** that maps the `OrtCreateIoBinding` and `OrtIoBinding_BindInput` functions (I will need the exact C signatures from the header — I can fetch them now if you want me to), and/or
* draft the **user-facing README + sample** that shows how to run a sample model using the new Java I/O binding wrapper.

Which would you like me to do next? (I can start by extracting the actual C function signatures from the ONNX repo and generate exact FFM descriptors & code.)

[1]: https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html?utm_source=chatgpt.com "I/O Binding | onnxruntime"
[2]: https://onnxruntime.ai/docs/get-started/with-java.html?utm_source=chatgpt.com "Get Started with ORT for Java"
[3]: https://github.com/microsoft/onnxruntime "GitHub - microsoft/onnxruntime: ONNX Runtime: cross-platform, high performance ML inferencing and training accelerator"
[4]: https://openjdk.org/jeps/454?utm_source=chatgpt.com "JEP 454: Foreign Function & Memory API"
[5]: https://onnxruntime.ai/docs/api/csharp/api/Microsoft.ML.OnnxRuntime.OrtIoBinding.html?utm_source=chatgpt.com "Class OrtIoBinding"
[6]: https://github.com/microsoft/onnxruntime-genai?utm_source=chatgpt.com "microsoft/onnxruntime-genai: Generative AI extensions for ..."
[7]: https://github.jpy.wang/microsoft/onnxruntime/issues/26209?utm_source=chatgpt.com "[Feature Request] IOBinding in Java API · Issue #26209 - jpy"


