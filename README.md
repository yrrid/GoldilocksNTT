# Goldilocks NTT Exploration

## Introduction

This project explores a new idea for Goldilocks NTT on the GPU.
It stores each Goldilocks point as 4 signed int32 values.  24 bits
of each 32-bit stores data, the remaining 8 bits stores carries and
borrows.

This new approach is used to compute 1024-point NTTs.  

## GPU Card Requirements

The code uses a large shared memory of 97KB, and therefore only runs on Ampere and Ada cards.

## Files

|File                      | Description           |
|--------------------------|-----------------------|
|BluePrint-NTT.c           | Blue print of a 1024-point NTT computation.  GPU code was modeled on this implementation. |
|compat.c                  | CPU version of GPU PTX instructions.  Allows for debugging low level GPU operations on the CPU. |
|HardRandom.cpp            | Tester used to develop the low level shiftRoot32 and normalize routines. |
|Tester.cu                 | Runs many 1024-point NTTs |
|DataOnly.cu               | GPU tester that does all the reads and writes but no computation. |
|Kernel.cu                 | Implements the __global__ kernel |
|NTTEngine32.cu            | Implements components for NTT-1024 and NTT-32. |
|Sampled96.cu              | Implements the Goldilocks math primitives. |
|asm.cu                    | Low level ptx bindings. |

## Compiling and Running

To compile and run the GPU code:

```
  nvcc -arch=sm_89 Tester.cu -o gpu_tester
  ./gpu_tester 1024 10 >gpu.out
````

To compile and run the CPU code and verify the results:

```
  gcc Correct.c -o cpu_tester
  ./cpu_tester 1024 10 >cpu.out
  diff cpu.out gpu.out
```

To compile and run the GPU code for compute only:

```
  nvcc -arch=sm_89 Tester.cu -o compute_only -DCOMPUTE_ONLY
  ./compute_only 1024 100
```

To compile and run the GPU code for data loading/storing only (no compute):

```
  nvcc -arch=sm_89 DataOnly.cu -o data_only
  ./data_only 1024 100
```

The data only and compute only builds are useful for determining if we're compute or memory bound.

### Performance

I am running on an RTX 4070 Ti, and I see the best performance with `./gpu_tester 3072 100`, with a run time of
4.713 ms.  This is `3*1024*100 / 0.004713` = 65.2 million 1024-point NTTs per second.   Note, this performance
is achieved by remaining in L2 cache.  Global memory bandwidth on my card is about 2.5x too slow to sustain this rate.

If the software were further developed, I believe it should be possible to achieve about 24,000 1M-point NTTs/sec on an
RTX 4070 Ti.


