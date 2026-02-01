# Assembly Dot Product Implementation Using AVX2 Vector Registers

Hand-optimized x64 assembly implementation benchmarked vs Intel MKL library.

## Description

`dot_ymm.asm` is a hand-optimized assembly implementation that computes dot product of arbitrary-size vectors using Intel AVX2 256-bit YMM registers.

## Benchmarks

The implementation is timed vs:
- **Intel MKL BLAS library**
- **Assembly scalar** - simple implementation using scalar registers
- **C scalar** - standard C implementation with O2/O3 compiler optimization

## Results

`dot_ymm.asm` performance:
- **Faster** than Intel MKL in most runs
- **up to 5-6× faster** than optimized C implementation

## Requirements

- **CPU**: Intel or AMD processor with AVX2 support
- **Intel MKL**: Required for benchmarking only (optional)

## Example Results
<img width="672" height="382" alt="dot" src="https://github.com/user-attachments/assets/db835a87-3b65-43f3-b4f8-61ac48626103" />

## Notes
Floating-point accumulation results differ slightly between implementations due to expected rounding error.


