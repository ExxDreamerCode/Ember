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
aarch64-simd512 search backends. The automatic aarch64 default is scalar
because the tested Arm machines did not show a stable single-thread search
win from portable SIMD. `setoption name NNUEBackend value ...` can force a
supported backend for comparison.
