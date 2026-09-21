# NNUE backend benchmark machines

Use `backend_bench` for CPU backend comparisons. Search NPS results are
single-threaded unless the result row says otherwise with its `threads`
field.

Recommended command:

```bash
cargo build --release --locked --bin backend_bench
./target/release/backend_bench \
  --refresh-loops=4000 \
  --update-loops=400 \
  --search-depth=10 \
  --search-repeats=3 \
  --search-threads=1,4 \
  --json=/tmp/backend-bench.json
```

Pass an external native, compact `ECN1`, or Ember V2 network through the same
loader used by UCI with `--nnue=/path/to/network.nnue`. The `search` results then
compare every available backend against that network. Low-level refresh and
incremental timings are emitted only for native Ember networks because those
measure the native `NNUENet` accumulator directly.

Benchmark host classes used for the backend dispatch checks:

| Provider | Machine type | Arch | vCPU | Memory | CPU model | Relevant CPU features |
| --- | --- | --- | ---: | ---: | --- | --- |
| Hetzner Cloud | cx53 | x86_64 | 8 | 16 GiB | AMD EPYC-Rome Processor | x86-64-v3 / AVX2, no AVX512 |
| Google Cloud | c4a-standard-4 | aarch64 | 4 | 16 GiB | Neoverse-V2 | ASIMD, SVE, SVE2, dot-product, i8mm, bf16 |
| Google Cloud | t2a-standard-4 | aarch64 | 4 | 16 GiB | Neoverse-N1 | ASIMD, dot-product |
| Google Cloud | c4-standard-4 | x86_64 | 4 | 16 GiB | Intel Xeon Platinum 8581C | x86-64-v3 / AVX2, AVX512F/BW/DQ/VL |
| Google Cloud | c4d-standard-4 | x86_64 | 4 | 16 GiB | AMD EPYC 9B45 | x86-64-v3 / AVX2, AVX512F/BW/DQ/VL |

Measured depth-10 search NPS, three repeats, using `--search-threads=1,4`:

| Machine type | Backend | NPS, 1 thread | NPS, 4 threads |
| --- | --- | ---: | ---: |
| cx53 | scalar | 243826 | 778543 |
| cx53 | x86-v3 | 641644 | 1584260 |
| c4a-standard-4 | scalar | 437704 | 1600505 |
| c4a-standard-4 | aarch64-simd128 | 414786 | 1612705 |
| c4a-standard-4 | aarch64-simd256 | 435036 | 1463641 |
| c4a-standard-4 | aarch64-simd512 | 423804 | 1559227 |
| t2a-standard-4 | scalar | 186448 | 683342 |
| t2a-standard-4 | aarch64-simd128 | 184602 | 719073 |
| t2a-standard-4 | aarch64-simd256 | 186844 | 716214 |
| t2a-standard-4 | aarch64-simd512 | 187433 | 693142 |
| c4-standard-4 | scalar | 388006 | 793705 |
| c4-standard-4 | x86-v3 | 842456 | 1677253 |
| c4-standard-4 | x86-avx512 | 942605 | 1880189 |
| c4d-standard-4 | scalar | 423924 | 960329 |
| c4d-standard-4 | x86-v3 | 1345010 | 2708587 |
| c4d-standard-4 | x86-avx512 | 1620514 | 3129727 |

The x86 binary contains scalar, x86-v3, and x86-avx512 search backends.
The aarch64 binary contains scalar, aarch64-simd128, aarch64-simd256, and
aarch64-simd512 search backends. `setoption name NNUEBackend value ...` can
force a supported backend for comparison.

The earlier rows above predate the native Ember V2 AArch64 dense kernels. A
2026-09-19 remeasurement on `t2a-standard-4` used an interleaved eight-position
depth-10 corpus, Hash 128 MiB, book disabled, and PGO static-musl packages. The
candidate (`32695d5`) measured 1.774x the old scalar package (`c34cadf`) at one
thread (five repeats), and 1.648x at four threads (three repeats). All 40
single-thread repeat-position pairs had identical node counts. Raw logs and
binary hashes for this comparison were retained under
`results/arm64-simd-opt/remote-package-comparison/`. The earlier plain-build
1.810x/1.749x/1.711x comparisons were not found in the local archive during
review and are not used here as auditable evidence.
Separate plain-build comparisons with the archived V1 dense and compact
networks measured 3.127x and 3.119x the same-revision scalar backend,
respectively. At four threads the dense V1 comparison measured 2.994x; its
raw logs are under `results/arm64-simd-opt/remote-v1/`.
These results establish a gain on this DOTPROD-capable N1. The current
automatic backend is `aarch64-simd256` on all AArch64 platforms, but its speed
on non-DOTPROD hardware, Apple/Windows targets, and Cortex-A55/A76 Android
phones remains unmeasured. The backend keeps a baseline NEON dense-kernel
path and selects DOTPROD at runtime when available.

Generic ARM64 correctness and PGO validation:

```bash
nix run .#aarch64-qemu-tests v2_
EMBER_QEMU_CPU=max nix run .#aarch64-qemu-tests v2_
nix build .#ember-linux-arm64 --out-link result-linux-arm64
nix build .#ember-linux-arm64-plain --out-link result-linux-arm64-plain
nix develop .#release-ci --command python3 tools/verify_arm64_release.py \
  --plain result-linux-arm64-plain/bin/ember \
  --pgo result-linux-arm64/bin/ember \
  --out-dir results/arm64-release-parity
```

The default QEMU test CPU is `cortex-a53` (no DOTPROD); `max` also exercises
DOTPROD. The release check compares exact single-thread scores, PVs, node
counts and bench signatures across both CPU models, scalar/auto dispatch,
and same-target plain/PGO packages. It also exercises completed two-thread
searches and process shutdown. Use a new result directory for each run.

The generic Linux ARM64 PGO derivation trains both CPU models and requires
nonzero execution counts for all four native dense-kernel entry points.
Its `training/` output preserves workload logs and coverage counts. QEMU
timings cannot establish a performance improvement; native paired NPS is
still required when adopting a changed profile. These checks also do not
replace Android GUI integration, big/little affinity, sustained thermal,
tournament clock-safety, and paired playing-strength measurements.

A follow-up native check at `c38064d` rebuilt the final plain and PGO packages
on the same four-core N1 and compared the revised profile with the `32695d5`
PGO package. All runs used the eight-position corpus, Hash 128 MiB, no book,
and interleaved sampling after warm-up:

| Depth | Threads | Repeats | Revised / previous PGO NPS |
| ---: | ---: | ---: | ---: |
| 10 | 1 | 5 | 0.999x |
| 10 | 4 | 3 | 0.963x |
| 12 | 1 | 5 | 1.009x |
| 12 | 4 | 7 | 0.997x |

The shallow SMP result prompted the deeper confirmation with an A/A control:
the previous binary under a second label measured 1.001x at depth 12. Its
per-repeat ratios ranged from 0.892x to 1.104x, showing substantial SMP
variation. These measurements support approximately unchanged throughput;
they do not establish a speed gain from the profile change. The revised PGO
package measured 1.061x the same-source plain package in the depth-10
single-thread comparison.

All 80 single-thread position/repeat groups across depths 10 and 12 retained
exact scores, PVs, moves and node counts. Final-package scalar/auto and
plain/PGO parity also passed on both QEMU CPU models, including the lifecycle
checks, and natively against the previous package. Both PGO training models
reproduced the depth-12/14 signatures and covered all four native kernels.
Raw transcripts, source/binary/profile/network hashes and run configurations
were preserved in the private runner artifact directory. These results remain
specific to N1; the Android and other-platform limitations above still apply.
