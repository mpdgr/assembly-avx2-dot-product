#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <windows.h>
#include <mkl.h>                // intel MKL required to use as benchmark
#include <mkl_cblas.h>

#define VECTOR_LEN 10000000     // length of the vector to test
#define TEST_ITERS 100          // number of iterations for average result

extern float dot_basic(const float* vectors, long long len);
extern float dot_ymm(const float* vectors, long long len);
float dot_test(const float* vectors, long long len);
void pin_thread_to_core(int core_id);

// asm functions
typedef float (*dot_function)(const float *arr, long long size);

// MKL wrapper
float dot_mkl(const float* vectors, long long len);

// test item
typedef struct {
    dot_function function;
    char* name;
} asm_dot_test;

void run_function_test(asm_dot_test test, const float* vectors, long long arr_len, float expected);

int main()
{
    pin_thread_to_core(0);  // test should run on one core

    // input
    long long vector_len = VECTOR_LEN;
    long long arr_len = vector_len * 2;
    volatile float expected;

    // 32-bit aligned input buffer
    float* vectors = (float*)_aligned_malloc(arr_len * sizeof(float), 32);
    if (vectors == NULL) {
        printf("Memory allocation failed!\n");
        return 1;
    }
    // init data
    srand((unsigned int)time(NULL));
    for (long long i = 0; i < arr_len; i++) {
        vectors[i] = ((float)rand() / RAND_MAX) * 0.01f;
    }

    // compute expected result
    expected = dot_test(vectors, arr_len);

    // run tests
    asm_dot_test tests[] = {
        {dot_test, "C - scalar loop"},
        {dot_basic, "ASM - unoptimized - scalar loop"},
        {dot_ymm, "ASM - optimized - YMM registers SIMD loop"},
        {dot_mkl, "Intel Math Kernel Library (oneMKL)"},
    };

    printf("\n==============================================================================================\n");
    printf("\nDOT PRODUCT BENCHMARK\n");
    printf("compute dot product of two vectors of n = %lld float elements\n",
        vector_len);
    printf("(average of n=%d runs)\n\n", TEST_ITERS);

    for (int i = 0; i < sizeof(tests)/sizeof(tests[0]); i++)
    {
        run_function_test(tests[i], vectors, arr_len, expected);
    }

    printf("==============================================================================================\n");

    _aligned_free(vectors);
    return 0;
}

void run_function_test(asm_dot_test test, const float* vectors, long long arr_len, float expected)
{
    LARGE_INTEGER frequency, start, end;
    QueryPerformanceFrequency(&frequency);

    dot_function f = test.function;
    int iters = TEST_ITERS;
    volatile float result;

    QueryPerformanceCounter(&start);        // start timing

    for (int i = 0; i < iters; i++) {
        result = f(vectors, arr_len);
    }

    QueryPerformanceCounter(&end);          // end timing

    double elapsed_s = (double)(end.QuadPart - start.QuadPart) / (frequency.QuadPart * iters);
    double elapsed_ms = elapsed_s * 1000.0;
    double elapsed_us = elapsed_s * 1000000.0;

    printf("Test: %s\n", test.name);
    printf("Time: %.3f ms, %.1f us\n", elapsed_ms, elapsed_us);
    printf("Dot product result: %.2f\n", result);
    printf("Dot product test result: %.2f\n", expected);
    printf("Accumulated floating point error: %.8f\n\n", expected - result);

}

// MKL wrapper for dot product
float dot_mkl(const float* vectors, long long len)
{
    float result = 0.0f;

    // set MKL to use only 1 thread (but more threads won't improve speed anyway)
    mkl_set_num_threads(1);

    MKL_INT n = (MKL_INT)(len / 2);

    result = cblas_sdot(n, vectors, 1, vectors + n, 1);

    return result;
}

float dot_test(const float* vectors, long long len)
{
    float res = 0;
    for (long long i = 0; i < len / 2; ++i)
    {
        res += vectors[i] * vectors[len / 2 + i];
    }
    return res;
}

void pin_thread_to_core(int core_id)
{
    HANDLE thread = GetCurrentThread();
    DWORD_PTR mask = ((DWORD_PTR)1 << core_id);  // pin to core_id
    SetThreadAffinityMask(thread, mask);
}
