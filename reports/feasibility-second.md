# Second attempt: restoration works; compile setup still incomplete

The second bounded attempt again passed both stock GPU self-checks. Standalone compilation failed because the wheel include path shadowed the toolkit's matching runtime headers:

```text
macro "__cudaLaunch" passed 2 arguments, but takes just 1
```

The generated nvcc stub used a runtime interface incompatible with the wheel's `crt/host_runtime.h`. The next fix places the CUDA toolkit include directory before the wheel include directory. This preserves access to `cusparse.h` without selecting the wrong runtime header.

**Restoration succeeded this time:** the original approved 0.85 containers restarted, the model responded, and model/draft path, config-hash and shard-size manifests matched the pre-run records. No candidate was installed in the serving image and no weight data was copied.

To avoid downtime for compiler setup failures, the probe now supports `PROBE_COMPILE_ONLY=1`. The runner compiles/imports both modules without GPU access, under CPU/RAM limits, before stopping the service. Only successful precompilation proceeds to isolated GPU checks and timing.

A separate compile-only qualification is in progress. Until it passes and GPU evaluation completes, standalone kernel research remains unqualified. These failures are build/harness issues, not evidence that the E2 kernels or the model are incorrect.
