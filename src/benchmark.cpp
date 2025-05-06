#include <HPCHighDimensionFlatArray.hpp>
#include <SparseDS.hpp>
#include <benchmark/benchmark.h>
#include <cmath>
#include <libmorton/morton.h>
// #include <omp.h>
#include <sse2neon.h>
#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>
#include <vector>

constexpr std::size_t m = 1 << 13;
constexpr std::size_t n = 1 << 15;
std::vector<float> arr(n);

#define N (1024 * 1024)

constexpr int nblur = 8;
constexpr std::size_t nx = 1 << 13;
constexpr std::size_t ny = 1 << 13;
hpc::HPCHighDimensionFlatArray<2, float, nblur> a(nx, ny);
hpc::HPCHighDimensionFlatArray<2, float> b(nx, ny);

static void BM_AOS_partical(benchmark::State &bm) {
  struct AOS {
    float x;
    float y;
    float z;
  };

  std::vector<AOS> arr(n);
  for (auto _ : bm) {
#pragma omp parallel for
    for (std::size_t i = 0; i < n; ++i) {
      arr[i].x = arr[i].x + arr[i].y;
    }
    benchmark::DoNotOptimize(arr);
  }
}

static void BM_SOA_partical(benchmark::State &bm) {
  std::vector<float> x(n);
  std::vector<float> y(n);
  std::vector<float> z(n);

  for (auto _ : bm) {
#pragma omp parallel for
    for (std::size_t i = 0; i < n; ++i) {
      x[i] = x[i] * y[i];
    }
    benchmark::DoNotOptimize(x);
    benchmark::DoNotOptimize(y);
    benchmark::DoNotOptimize(z);
  }
}

static void BM_AOSOA_partical(benchmark::State &bm) {
  struct AOSOA {
    float x[1024];
    float y[1024];
    float z[1024];
  };

  std::vector<AOSOA> arr(n / 1024);

  for (auto _ : bm) {
#pragma omp parallel for
    for (std::size_t i = 0; i < n / 1024; ++i) {
#pragma omp simd
      for (std::size_t j = 0; j < 1024; ++j) {
        arr[i].x[j] = arr[i].x[j] + arr[i].y[j];
      }
    }
    benchmark::DoNotOptimize(arr);
  }
}

static void BM_AOS_all_properties(benchmark::State &bm) {
  struct AOS {
    float x, y, z;
  };
  std::vector<AOS> arr(n);

  for (auto _ : bm) {
#pragma omp parallel for
    for (std::size_t i = 0; i < n; ++i) {
      arr[i].x += sin(i);
      arr[i].y += sin(i + 1);
      arr[i].z += sin(i + 2);
    }
    benchmark::DoNotOptimize(arr);
  }
}

static void BM_SOA_all_properties(benchmark::State &bm) {
  std::vector<float> x(n);
  std::vector<float> y(n);
  std::vector<float> z(n);

  for (auto _ : bm) {
#pragma omp parallel for
    for (std::size_t i = 0; i < n; ++i) {
      x[i] += sin(i);
      y[i] += sin(i + 1);
      z[i] += sin(i + 2);
    }
    benchmark::DoNotOptimize(x);
    benchmark::DoNotOptimize(y);
    benchmark::DoNotOptimize(z);
  }
}

static void BM_AOSOA_all_properties(benchmark::State &bm) {
  struct AOSOA {
    float x[1024];
    float y[1024];
    float z[1024];
  };
  std::vector<AOSOA> arr(n / 1024);

  for (auto _ : bm) {
#pragma omp parallel for
    for (std::size_t i = 0; i < n / 1024; ++i) {
#pragma omp simd
      for (std::size_t j = 0; j < 1024; ++j) {
        arr[i].x[j] += sin(j);
        arr[i].y[j] += sin(j);
        arr[i].z[j] += sin(j);
      }
    }
    benchmark::DoNotOptimize(arr);
  }
}

static void BM_ordered(benchmark::State &bm) {
  for (auto _ : bm) {
#pragma omp parallel for
    for (std::size_t i = 0; i < n; ++i) {
      benchmark::DoNotOptimize(arr[i]);
    }
    benchmark::DoNotOptimize(arr);
  }
}

static void BM_random(benchmark::State &bm) {
  for (auto _ : bm) {
#pragma omp parallel for
    for (std::size_t i = 0; i < n; ++i) {
      benchmark::DoNotOptimize(arr[rand() % n]);
    }
    benchmark::DoNotOptimize(arr);
  }
}

static void BM_random_64B(benchmark::State &bm) {
  for (auto _ : bm) {
#pragma omp parallel for
    for (std::size_t i = 0; i < n / 16; ++i) {
      auto r = rand() % (n / 16);
      for (std::size_t j = 0; j < 16; ++j) {
        benchmark::DoNotOptimize(arr[16 * r + j]);
      }
    }
    benchmark::DoNotOptimize(arr);
  }
}

static void BM_random_64B_prefetch(benchmark::State &bm) {
  for (auto _ : bm) {
#pragma omp parallel for
    for (std::size_t i = 0; i < n / 16; ++i) {
      auto r = rand() % (n / 16);
      _mm_prefetch(reinterpret_cast<const char *>(&arr[16 * (r + 1)]),
                   _MM_HINT_T0);
      for (std::size_t j = 0; j < 16; ++j) {
        benchmark::DoNotOptimize(arr[16 * r + j]);
      }
    }
    benchmark::DoNotOptimize(arr);
  }
}

constexpr uint64_t block = 4096;
static void BM_random_4096B(benchmark::State &bm) {
  for (auto _ : bm) {
#pragma omp parallel for
    for (std::size_t i = 0; i < n / block; ++i) {
      auto r = rand() % (n / block);
#pragma omp simd
      for (std::size_t j = 0; j < block; ++j)
        benchmark::DoNotOptimize(arr[block * r + j]);

      benchmark::DoNotOptimize(arr);
    }
  }
}

static void BM_random_4KB_align(benchmark::State &bm) {
  float *a = (float *)_mm_malloc(n * sizeof(float), block);
  memset(a, 0, n * sizeof(float));

  for (auto _ : bm) {
#pragma omp parallel for
    for (std::size_t i = 0; i < n / block; ++i) {
      auto r = rand() % (n / block);
#pragma omp simd
      for (std::size_t j = 0; j < block; ++j)
        benchmark::DoNotOptimize(a[block * r + j]);

      benchmark::DoNotOptimize(arr);
    }
  }
  _mm_free(a);
}

static void BM_write(benchmark::State &bm) {
  for (auto _ : bm) {
#pragma omp parallel for
    for (std::size_t i = 0; i < n; ++i) {
      arr[i] = 1;
    }
    benchmark::DoNotOptimize(arr);
  }
}

static void BM_write_streamed(benchmark::State &bm) {
  for (auto _ : bm) {
#pragma omp parallel for
    for (std::size_t i = 0; i < n; ++i) {
      float value = 1;
      _mm_stream_si32((int *)&arr[i], *(int *)&value);
    }
    benchmark::DoNotOptimize(arr);
  }
}

static void BM_write_streamed_and_read(benchmark::State &bm) {
  for (auto _ : bm) {
#pragma omp parallel for
    for (std::size_t i = 0; i < n; ++i) {
      float value = 1;
      _mm_stream_si32((int *)&arr[i], *(int *)&value);
      benchmark::DoNotOptimize(arr[i]); // read again
    }
    benchmark::DoNotOptimize(arr);
  }
}

static void BM_read_and_write(benchmark::State &bm) {
  for (auto _ : bm) {
#pragma omp parallel for
    for (std::size_t i = 0; i < n; ++i) {
      arr[i] = arr[i] + 1;
    }
    benchmark::DoNotOptimize(arr);
  }
}

static void BM_write_zero(benchmark::State &bm) {
  for (auto _ : bm) {
#pragma omp parallel for
    for (std::size_t i = 0; i < n; ++i) {
      arr[i] = 0;
    }
    benchmark::DoNotOptimize(arr);
  }
}

static void BM_write_one(benchmark::State &bm) {
  for (auto _ : bm) {
#pragma omp parallel for
    for (std::size_t i = 0; i < n; ++i) {
      arr[i] = 1;
    }
    benchmark::DoNotOptimize(arr);
  }
}

static void BM_origin(benchmark::State &bm) {
  int *m = new int[n];
  for (auto _ : bm) {
    for (std::size_t i = 0; i < n; ++i) {
      m[i] = 1;
    }
  }
  delete[] m;
}

static void BM_init(benchmark::State &bm) {
  int *m = new int[n]{};
  for (auto _ : bm) {
    for (std::size_t i = 0; i < n; ++i) {
      m[i] = 1;
    }
  }
  delete[] m;
}

static void BM_java_style(benchmark::State &bm) {
  for (auto _ : bm) {
    std::vector<std::vector<std::vector<float>>> dimension(
        n, std::vector<std::vector<float>>(n, std::vector<float>(n)));
    benchmark::DoNotOptimize(dimension);
  }
}

static void BM_flat(benchmark::State &bm) {
  std::vector<float> flat(n * n * n);
  for (auto _ : bm) {
    std::vector<float> flat(n * n * n);
    benchmark::DoNotOptimize(flat);
  }
}

static void BM_XY(benchmark::State &bm) {
  std::vector<float> matrix2d(nx * ny);
  for (auto _ : bm) {
#pragma omp parallel for collapse(2)
    for (std::size_t x = 0; x < nx; ++x) {
      for (std::size_t y = 0; y < ny; ++y) {
        matrix2d[x + y * nx] = 1.f;
      }
    }
    benchmark::DoNotOptimize(matrix2d);
  }
}

static void BM_YX(benchmark::State &bm) {
  std::vector<float> matrix2d(nx * ny);
  for (auto _ : bm) {
#pragma omp parallel for collapse(2)
    for (std::size_t y = 0; y < ny; ++y) {
      for (std::size_t x = 0; x < nx; ++x) {
        matrix2d[x + y * nx] = 1.f;
      }
    }
    benchmark::DoNotOptimize(matrix2d);
  }
}

static void BM_x_blur(benchmark::State &bm) {
  for (auto _ : bm) {
#pragma omp parallel for collapse(2)
    for (int y = 0; y < ny; ++y) {
      for (int x = 0; x < nx; ++x) {
        float res = {0.f};
        for (int blur = -nblur; blur <= nblur; ++blur) {
          res += a(y, x + blur);
        }
        b(y, x) = res;
      }
    }
    benchmark::DoNotOptimize(a);
  }
}

static void BM_x_blur_prefetch(benchmark::State &bm) {
  for (auto _ : bm) {
#pragma omp parallel for collapse(2)
    for (int y = 0; y < ny; ++y) {
      for (int x = 0; x < nx; ++x) {
        float res = {0.f};
        _mm_prefetch((const char *)&a(y, x + 2 * nblur), _MM_HINT_T0);
        for (int blur = -nblur; blur <= nblur; ++blur) {
          res += a(y, x + blur);
        }
        b(y, x) = res;
      }
    }
    benchmark::DoNotOptimize(a);
  }
}

static void BM_x_blur_cond_prefetch(benchmark::State &bm) {
  for (auto _ : bm) {
#pragma omp parallel for collapse(2)
    for (int y = 0; y < ny; ++y) {
      for (int x = 0; x < nx; ++x) {
        float res = {0.f};
        if (x % 16) {
          _mm_prefetch((const char *)&a(y, x + 2 * nblur), _MM_HINT_T0);
        }
        for (int blur = -nblur; blur <= nblur; ++blur) {
          res += a(y, x + blur);
        }
        b(y, x) = res;
      }
    }
    benchmark::DoNotOptimize(a);
  }
}

static void BM_x_blur_tiling_prefetch(benchmark::State &bm) {
  for (auto _ : bm) {
#pragma omp parallel for collapse(2)
    for (int y = 0; y < ny; ++y) {
      for (int xBase = 0; xBase < static_cast<int>(nx); xBase += 2 * nblur) {
        float res = {0.f};
        _mm_prefetch((const char *)&a(y, xBase + 2 * nblur), _MM_HINT_T0);
        for (int x = xBase; x < xBase + 2 * nblur; ++x) {
          for (int blur = -nblur; blur <= nblur; ++blur) {
            res += a(y, x + blur);
          }
          b(y, x) = res;
        }
      }
    }
    benchmark::DoNotOptimize(a);
  }
}

static void BM_x_blur_tiling_simd_prefetch(benchmark::State &bm) {
  for (auto _ : bm) {
#pragma omp parallel for collapse(2)
    for (int y = 0; y < ny; ++y) {
      for (int xBase = 0; xBase < static_cast<int>(nx); xBase += 2 * nblur) {
        _mm_prefetch((const char *)&a(y, xBase + 2 * nblur), _MM_HINT_T0);
        for (int x = xBase; x < xBase + 2 * nblur; x += 4) {
          __m128 res = _mm_setzero_ps();
          for (int blur = -nblur; blur <= nblur; ++blur)
            res = _mm_add_ps(_mm_loadu_ps(&a(y, x + blur)), res);
          _mm_stream_ps(&b(y, x), res);
        }
      }
    }
    benchmark::DoNotOptimize(a);
  }
}

constexpr int blockSize = 64;
static void BM_y_blur(benchmark::State &bm) {
  for (auto _ : bm) {
#pragma omp parallel for collapse(2)
    for (int y = 0; y < ny; ++y) {
      for (int x = 0; x < nx; ++x) {
        float res = {0.f};
        for (int blur = -nblur; blur <= nblur; ++blur)
          res += a(y + blur, x);
        b(y, x) = res;
      }
    }
    benchmark::DoNotOptimize(a);
  }
}

static void BM_y_blur_tiling(benchmark::State &bm) {
  for (auto _ : bm) {
#pragma omp parallel for collapse(2)
    for (int yBase = 0; yBase < ny; yBase += blockSize) {
      for (int xBase = 0; xBase < nx; xBase += blockSize) {
        for (int y = yBase; y < yBase + blockSize; ++y) {
          for (int x = xBase; x < xBase + blockSize; ++x) {
            float res = {};
            for (int blur = -nblur; blur <= nblur; ++blur)
              res += a(y + blur, x);
            b(y, x) = res;
          }
        }
      }
    }
    benchmark::DoNotOptimize(a);
  }
}

static void BM_XYx_blur_tiling(benchmark::State &bm) {
  for (auto _ : bm) {
#pragma omp parallel for collapse(2)
    for (int xBase = 0; xBase < nx; xBase += blockSize) {
      for (int y = 0; y < ny; ++y) {
        for (int x = xBase; x < xBase + blockSize; ++x) {
          float res = {};
          for (int blur = -nblur; blur <= nblur; ++blur)
            res += a(y + blur, x);
          b(y, x) = res;
        }
      }
    }
    benchmark::DoNotOptimize(a);
  }
}

static void BM_YXx_blur_tiling(benchmark::State &bm) {
  for (auto _ : bm) {
#pragma omp parallel for collapse(2)
    for (int y = 0; y < ny; ++y) {
      for (int xBase = 0; xBase < nx; xBase += blockSize) {
        for (int x = xBase; x < xBase + blockSize; ++x) {
          float res = {};
          for (int blur = -nblur; blur <= nblur; ++blur)
            res += a(y + blur, x);
          b(y, x) = res;
        }
      }
    }
    benchmark::DoNotOptimize(a);
  }
}

static void BM_YXx_blur_tiling_prefetch(benchmark::State &bm) {
  for (auto _ : bm) {
#pragma omp parallel for collapse(2)
    for (int y = 0; y < ny; ++y) {
      for (int xBase = 0; xBase < nx; xBase += blockSize) {
        _mm_prefetch((const char *)&a(y + nblur, xBase), _MM_HINT_T0);
        for (int x = xBase; x < xBase + blockSize; ++x) {
          float res = {};
          for (int blur = -nblur; blur <= nblur; ++blur)
            res += a(y + blur, x);
          b(y, x) = res;
        }
      }
    }
    benchmark::DoNotOptimize(a);
  }
}

static void BM_YXx_blur_tiling_prefetch_streamed(benchmark::State &bm) {
  for (auto _ : bm) {
#pragma omp parallel for collapse(2)
    for (int y = 0; y < ny; ++y) {
      for (int xBase = 0; xBase < nx; xBase += blockSize) {
        _mm_prefetch((const char *)&a(y + nblur, xBase), _MM_HINT_T0);
        for (int x = xBase; x < xBase + blockSize; ++x) {
          float res = {};
          for (int blur = -nblur; blur <= nblur; ++blur)
            res += a(y + blur, x);
          _mm_stream_si32((int *)&b(y, x), res);
        }
      }
    }
    benchmark::DoNotOptimize(a);
  }
}

static void BM_YXx_blur_tiling_prefetch_streamed_merged(benchmark::State &bm) {
  for (auto _ : bm) {
#pragma omp parallel for collapse(2)
    for (int y = 0; y < ny; ++y) {
      for (int xBase = 0; xBase < nx; xBase += blockSize) {
        _mm_prefetch((const char *)&a(y + nblur, xBase), _MM_HINT_T0);
        for (int x = xBase; x < xBase + blockSize; x += 16) {
          __m128 res[4];
          for (int offset = 0; offset < 4; ++offset) {
            for (int blur = -nblur; blur <= nblur; ++blur) {
              res[offset] = _mm_setzero_ps();
              res[offset] = _mm_add_ps(
                  res[offset],
                  _mm_load_ps((const float *)&a(y + blur, x + offset * 4)));
            }
          }
          for (int offset = 0; offset < 4; offset++) {
            _mm_stream_ps(&b(y, x + offset * 4), res[offset]);
          }
        }
      }
    }
    benchmark::DoNotOptimize(a);
  }
}

static void BM_YXx_blur_tiling_prefetch_streamed_IPL(benchmark::State &bm) {
  for (auto _ : bm) {
#pragma omp parallel for collapse(2)
    for (int y = 0; y < ny; ++y) {
      for (int xBase = 0; xBase < nx; xBase += blockSize) {
        _mm_prefetch((const char *)&a(y + nblur, xBase), _MM_HINT_T0);
        for (int x = xBase; x < xBase + blockSize; x += 16) {
          __m128 res[4];
          for (int offset = 0; offset < 4; ++offset) {
            res[offset] = _mm_setzero_ps();
          }
          for (int blur = -nblur; blur <= nblur; ++blur) {
            for (int offset = 0; offset < 4; ++offset) {
              res[offset] = _mm_add_ps(
                  res[offset],
                  _mm_load_ps((const float *)&a(y + blur, x + offset * 4)));
            }
          }
          for (int offset = 0; offset < 4; offset++) {
            _mm_stream_ps(&b(y, x + offset * 4), res[offset]);
          }
        }
      }
    }
    benchmark::DoNotOptimize(a);
  }
}

// hpc::HPCHighDimensionFlatArray<2, float, nblur, nblur, 32> a_avx(nx, ny);
// hpc::HPCHighDimensionFlatArray<2, float, 0, 0, 32> b_avx(nx, ny);

// static void BM_YXx_blur_tiling_prefetch_streamed_AVX2(benchmark::State &bm) {
//   for (auto _ : bm) {
// #pragma omp parallel for collapse(2)
//     for (int y = 0; y < ny; ++y) {
//       for (int x = 0; x < nx - 32 + 1; x += 32) {
//         __m256 res[4];
//         _mm_prefetch((const char *)&a_avx(y + nblur, x), _MM_HINT_T0);
//         _mm_prefetch((const char *)&a_avx(y + nblur, x + 16), _MM_HINT_T0);
//         for (int offset = 0; offset < 4; ++offset) {
//           res[offset] = _mm256_setzero_ps();
//         }

//         for (int blur = -nblur; blur <= nblur; ++blur) {
//           for (int offset = 0; offset < 4; ++offset) {
//             res[offset] =
//                 _mm256_add_ps(res[offset], _mm256_load_ps((const float
//                 *)&a_avx(
//                                                y + blur, x + offset * 8)));
//           }
//         }
//         for (int offset = 0; offset < 4; offset++) {
//           _mm256_stream_ps(&b_avx(y, x + offset * 8), res[offset]);
//         }
//       }
//     }
//     benchmark::DoNotOptimize(a);
//   }
// }

// static void
// BM_YXx_blur_tiling_prefetch_streamed_AVX2_in_advance(benchmark::State &bm) {
//   for (auto _ : bm) {
// #pragma omp parallel for collapse(2)
//     for (int y = 0; y < ny; ++y) {
//       for (int x = 0; x < nx - 32 + 1; x += 32) {
//         __m256 res[4];
//         _mm_prefetch((const char *)&a_avx(y + nblur, x + 32), _MM_HINT_T0);
//         _mm_prefetch((const char *)&a_avx(y + nblur, x + 16 + 32),
//         _MM_HINT_T0);

//         for (int offset = 0; offset < 4; ++offset) {
//           res[offset] = _mm256_setzero_ps();
//         }

//         for (int blur = -nblur; blur <= nblur; ++blur) {

//           for (int offset = 0; offset < 4; ++offset) {
//             res[offset] =
//                 _mm256_add_ps(res[offset], _mm256_load_ps((const float
//                 *)&a_avx(
//                                                y + blur, x + offset * 8)));
//           }
//         }

//         for (int offset = 0; offset < 4; offset++) {
//           _mm256_stream_ps(&b_avx(y, x + offset * 8), res[offset]);
//         }
//       }
//     }
//     benchmark::DoNotOptimize(a);
//   }
// }

hpc::HPCHighDimensionFlatArray<2, float> a_t(nx, ny);
hpc::HPCHighDimensionFlatArray<2, float> b_t(nx, ny);

static void BM_transpose(benchmark::State &bm) {
  for (auto _ : bm) {
#pragma omp parallel for collapse(2)
    for (int y = 0; y < ny; ++y) {
      for (int x = 0; x < nx; ++x) {
        b_t(x, y) = a_t(y, x);
      }
    }
    benchmark::DoNotOptimize(b_t);
  }
}

static void BM_transpose_tiling(benchmark::State &bm) {
  for (auto _ : bm) {
#pragma omp parallel for collapse(2)
    for (int yBase = 0; yBase < ny; yBase += blockSize) {
      for (int xBase = 0; xBase < nx; xBase += blockSize) {
        for (int y = yBase; y < yBase + blockSize; ++y) {
          for (int x = xBase; x < xBase + blockSize; ++x) {
            b_t(x, y) = a_t(y, x);
          }
        }
      }
    }
    benchmark::DoNotOptimize(b_t);
  }
}

static void BM_transpose_tiling_morton2d(benchmark::State &bm) {
  for (auto _ : bm) {
#pragma omp parallel for
    for (int t = 0; t < (nx * ny) / (blockSize * blockSize); ++t) {
      uint_fast16_t xBase{}, yBase{};
      libmorton::morton2D_32_decode(t, xBase, yBase);
      xBase *= blockSize;
      yBase *= blockSize;
      for (auto y = yBase; y < yBase + blockSize; ++y) {
        for (auto x = xBase; x < xBase + blockSize; ++x) {
          b_t(x, y) = a_t(y, x);
        }
      }
    }
    benchmark::DoNotOptimize(b_t);
  }
}

static void BM_transpose_tiling_morton2d_stream(benchmark::State &bm) {
  for (auto _ : bm) {
#pragma omp parallel for
    for (int t = 0; t < (nx * ny) / (blockSize * blockSize); ++t) {
      uint_fast16_t xBase{}, yBase{};
      libmorton::morton2D_32_decode(t, xBase, yBase);
      xBase *= blockSize;
      yBase *= blockSize;
      for (auto y = yBase; y < yBase + blockSize; ++y) {
        for (auto x = xBase; x < xBase + blockSize; x++) {
          _mm_stream_si32((int *)&b_t(x, y), (int &)a_t(y, x));
        }
      }
    }
    benchmark::DoNotOptimize(b_t);
  }
}

static void BM_transpose_tiling_tbb(benchmark::State &bm) {
  for (auto _ : bm) {
    tbb::parallel_for(
        tbb::blocked_range2d<std::size_t>(0, nx, blockSize, 0, ny, blockSize),
        [](tbb::blocked_range2d<std::size_t> &r) {
          for (auto y = r.cols().begin(); y != r.cols().end(); ++y) {
            for (auto x = r.rows().begin(); x != r.rows().end(); ++x) {
              b_t(x, y) = a_t(y, x);
            }
          }
        },
        tbb::simple_partitioner{});
    benchmark::DoNotOptimize(b_t);
  }
}

constexpr int size = 1 << 10;
constexpr int matrix_block = 32;
hpc::HPCHighDimensionFlatArray<2, float> ma(size, size);
hpc::HPCHighDimensionFlatArray<2, float> mb(size, size);
hpc::HPCHighDimensionFlatArray<2, float> mc(size, size);

static void BM_matrix_mul(benchmark::State &bm) {
  for (auto _ : bm) {
    for (int y = 0; y < size; ++y) {
      for (int x = 0; x < size; ++x) {
        for (int t = 0; t < size; ++t) {
          ma(y, x) += mc(y, t) * mb(t, x);
        }
      }
    }
    benchmark::DoNotOptimize(ma);
  }
}

static void BM_matrix_mul_blocked(benchmark::State &bm) {
  for (auto _ : bm) {
    for (int y = 0; y < size; y++) {
      for (int xBase = 0; xBase < size; xBase += matrix_block) {
        for (int t = 0; t < size; ++t) {
          for (int x = xBase; x < xBase + matrix_block; ++x) {
            ma(y, x) += mc(y, t) * mb(t, x);
          }
        }
      }
    }
    benchmark::DoNotOptimize(ma);
  }
}

constexpr int conv_block = 4;
constexpr int conv_n = 1 << 10;
constexpr int nkern = 16;
hpc::HPCHighDimensionFlatArray<2, float> ka(conv_n, conv_n);
hpc::HPCHighDimensionFlatArray<2, float, nkern> kb(conv_n, conv_n);
hpc::HPCHighDimensionFlatArray<2, float> kc(nkern, nkern);

static void BM_conv(benchmark::State &bm) {
  for (auto _ : bm) {
    for (int y = 0; y < conv_n; y++) {
      for (int x = 0; x < conv_n; ++x) {
        for (int l = 0; l < nkern; ++l) {
          for (int k = 0; k < nkern; ++k) {
            ka(y, x) += kb(y + l, x + k) * kc(l, k);
          }
        }
      }
    }
    benchmark::DoNotOptimize(ma);
  }
}

static void BM_conv_block(benchmark::State &bm) {
  for (auto _ : bm) {
    for (int yBase = 0; yBase < conv_n; yBase += conv_block)
      for (int xBase = 0; xBase < conv_n; xBase += conv_block)
        for (int l = 0; l < nkern; ++l)
          for (int k = 0; k < nkern; ++k)
            for (int y = yBase; y < yBase + conv_block; ++y)
              for (int x = xBase; x < xBase + conv_block; ++x)
                ka(y, x) += kb(y + l, x + k) * kc(l, k);
  }
}

static void BM_conv_block_unroll(benchmark::State &bm) {
  for (auto _ : bm) {
    for (int yBase = 0; yBase < conv_n; yBase += conv_block)
      for (int xBase = 0; xBase < conv_n; xBase += conv_block)
        for (int l = 0; l < nkern; ++l)
          for (int k = 0; k < nkern; ++k)
            for (int y = yBase; y < yBase + conv_block; ++y)
              for (int x = xBase; x < xBase + conv_block; ++x)
                ka(y, x) += kb(y + l, x + k) * kc(l, k);
  }
}

constexpr int line = 1 << 23;
std::vector<float> false_sharing(line);

// static void BM_false_sharing(benchmark::State &bm) {
//   for (auto _ : bm) {
//     std::vector<int> temp(omp_get_max_threads());
// #pragma omp parallel for
//     for (int i = 0; i < line; ++i) {
//       temp[omp_get_thread_num()] += false_sharing[i];
//       benchmark::DoNotOptimize(temp);
//     }
//     benchmark::DoNotOptimize(temp);
//   }
// }

// static void BM_no_false_sharing(benchmark::State &bm) {
//   for (auto _ : bm) {
//     std::vector<int> temp(omp_get_max_threads() * 4096);
// #pragma omp parallel for
//     for (int i = 0; i < line; ++i) {
//       temp[omp_get_thread_num() * 4096] += false_sharing[i];
//       benchmark::DoNotOptimize(temp);
//     }
//     benchmark::DoNotOptimize(temp);
//   }
// }

static void BM_grid_unordered_block_XY_o1(benchmark::State &bm) {
  for (auto _ : bm) {
    auto grid = std::make_shared<sparse::RootGrid<
        bool, sparse::HashBlock<sparse::DenseBlock<16, bool>>>>();
    float px = -100.f, py = 100.f;
    float vx = 0.2f, vy = -0.6f;

    for (std::size_t time = 0; time < N; ++time) {
      px += vx;
      py += vy;
      grid->write(px, py, true);
    }

    std::size_t counter{};
    grid->foreach ([&counter](auto x, auto y, auto &value) {
      if (value)
        counter++;
    });
    benchmark::DoNotOptimize(counter);
  }
}

static void BM_PointerGrid_SpinLock(benchmark::State &bm) {
  for (auto _ : bm) {
    auto grid = std::make_shared<sparse::RootGrid<
        bool,
        sparse::PointerBlock<1 << 11, sparse::DenseBlock<1 << 4, bool>>>>();
    float px = -100.f, py = 100.f;
    float vx = 0.2f, vy = -0.6f;

    // #pragma omp parallel for
    for (long long time = 0; time < N; ++time)
      grid->write(static_cast<std::intptr_t>(px + vx * time),
                  static_cast<std::intptr_t>(py + vy * time), true);

    std::atomic<std::size_t> counter{};
    grid->foreach ([&counter](auto x, auto y, auto &value) {
      if (value)
        counter++;
    });
    benchmark::DoNotOptimize(counter);
  }
}

static void BM_int64_t(benchmark::State &bm) {
  for (auto _ : bm) {
    std::vector<int64_t> arr(N);
    for (std::size_t i = 0; i < N; ++i) {
      arr[i] = i % 2;
    }
    benchmark::DoNotOptimize(arr);
  }
}
static void BM_int32_t(benchmark::State &bm) {
  for (auto _ : bm) {
    std::vector<int32_t> arr(N);
    for (std::size_t i = 0; i < N; ++i) {
      arr[i] = i % 2;
    }
    benchmark::DoNotOptimize(arr);
  }
}
static void BM_int8_t(benchmark::State &bm) {
  for (auto _ : bm) {
    std::vector<int8_t> arr(N);
    for (std::size_t i = 0; i < N; ++i) {
      arr[i] = i % 2;
    }
    benchmark::DoNotOptimize(arr);
  }
}

static void BM_8bit(benchmark::State &bm) {
  for (auto _ : bm) {
    std::vector<int8_t> arr(N / 8);
    for (std::size_t ib = 0; ib < N / 8; ++ib) {
      int8_t result = {};
      for (std::size_t di = 0; di < 8; ++di) {
        auto index = ib << 3 + di;
        result |= (index & 1) << di; // index % 2 = index & 1
      }
      arr[ib] = result;
    }
    benchmark::DoNotOptimize(arr);
  }
}

static void BM_double_calc(benchmark::State &bm) {
  for (auto _ : bm) {
    std::vector<double> arr(N);
    for (std::size_t i = 0; i < N; ++i) {
      arr[i] = i * 3.14;
    }
    benchmark::DoNotOptimize(arr);
  }
}

static void BM_float_calc(benchmark::State &bm) {
  for (auto _ : bm) {
    std::vector<float> arr(N);
    for (std::size_t i = 0; i < N; ++i) {
      arr[i] = i * 3.14f;
    }
    benchmark::DoNotOptimize(arr);
  }
}

static void BM_floatingpoint(benchmark::State &bm) {
  for (auto _ : bm) {
    std::vector<float> arr(N);
#pragma omp parallel for
    for (std::size_t i = 0; i < N; ++i)
      arr[i] = (i % 32) * 3.14f;

    float ret = 0;
#pragma omp parallel for reduction(max : ret)
    for (std::size_t i = 0; i < N; ++i)
      ret = std::max(ret, arr[i]);

    benchmark::DoNotOptimize(ret);
  }
}

template <typename IntegarType>
static IntegarType float2integar(const float f, const IntegarType k) {
  return static_cast<IntegarType>(f * static_cast<float>(k));
}

template <typename IntegarType>
static float integar2float(const IntegarType f, const std::intptr_t k) {
  return static_cast<float>(f / (1.0f * k));
}

static void BM_fixedpoint_32(benchmark::State &bm) {
  for (auto _ : bm) {
    std::vector<int32_t> arr(N);

#pragma omp parallel for
    for (std::size_t i = 0; i < N; ++i)
      arr[i] = float2integar((i & 31) * 3.14f, int32_t(100));

    float ret = 0;
#pragma omp parallel for reduction(max : ret)
    for (std::size_t i = 0; i < N; ++i) {
      ret = std::max(ret, integar2float(arr[i], 100));
    }
    benchmark::DoNotOptimize(ret);
  }
}

static void BM_fixedpoint_16(benchmark::State &bm) {
  for (auto _ : bm) {
    std::vector<int16_t> arr(N);

#pragma omp parallel for
    for (std::size_t i = 0; i < N; ++i)
      arr[i] = float2integar((i & 31) * 3.14f, int16_t(100));

    float ret = 0;
#pragma omp parallel for reduction(max : ret)
    for (std::size_t i = 0; i < N; ++i) {
      ret = std::max(ret, integar2float(arr[i], 100));
    }
    benchmark::DoNotOptimize(ret);
  }
}

static void BM_fixedpoint_uint8(benchmark::State &bm) {
  for (auto _ : bm) {
    std::vector<uint8_t> arr(N);
#pragma omp parallel for
    for (std::size_t i = 0; i < N; ++i)
      arr[i] = float2integar((i & 31) * 3.14f, uint8_t(2));

    float ret = 0;
#pragma omp parallel for reduction(max : ret)
    for (std::size_t i = 0; i < N; ++i)
      ret = std::max(ret, integar2float(arr[i], uint8_t(2)));

    benchmark::DoNotOptimize(ret);
  }
}

BENCHMARK(BM_AOS_partical);
BENCHMARK(BM_SOA_partical);
BENCHMARK(BM_AOSOA_partical);
BENCHMARK(BM_AOS_all_properties);
BENCHMARK(BM_SOA_all_properties);

BENCHMARK(BM_ordered);
BENCHMARK(BM_random_64B);
BENCHMARK(BM_random_4096B);
BENCHMARK(BM_random_4KB_align);
BENCHMARK(BM_random_64B);
BENCHMARK(BM_random_64B_prefetch);

BENCHMARK(BM_read_and_write);
BENCHMARK(BM_write);
// BENCHMARK(BM_write_streamed);
// BENCHMARK(BM_write_streamed_and_read);
// BENCHMARK(BM_write_zero);
// BENCHMARK(BM_write_one);
// BENCHMARK(BM_java_style);
// BENCHMARK(BM_flat);

// BENCHMARK(BM_x_blur);
// BENCHMARK(BM_x_blur_prefetch);
// BENCHMARK(BM_x_blur_cond_prefetch);
// BENCHMARK(BM_x_blur_tiling_prefetch);
// BENCHMARK(BM_x_blur_tiling_simd_prefetch);

// BENCHMARK(BM_y_blur);
// BENCHMARK(BM_y_blur_tiling);
// BENCHMARK(BM_XYx_blur_tiling);
// BENCHMARK(BM_YXx_blur_tiling);
// BENCHMARK(BM_YXx_blur_tiling_prefetch);
// BENCHMARK(BM_YXx_blur_tiling_prefetch_streamed);
// BENCHMARK(BM_YXx_blur_tiling_prefetch_streamed_merged);
// BENCHMARK(BM_YXx_blur_tiling_prefetch_streamed_IPL);
// BENCHMARK(BM_YXx_blur_tiling_prefetch_streamed_AVX2);
// BENCHMARK(BM_YXx_blur_tiling_prefetch_streamed_AVX2_in_advance);

// BENCHMARK(BM_transpose);
// BENCHMARK(BM_transpose_tiling);
// BENCHMARK(BM_transpose_tiling_morton2d);
// BENCHMARK(BM_transpose_tiling_morton2d_stream);
// BENCHMARK(BM_transpose_tiling_tbb);

// BENCHMARK(BM_matrix_mul);
// BENCHMARK(BM_matrix_mul_blocked);

// BENCHMARK(BM_conv);
// BENCHMARK(BM_conv_block);
// BENCHMARK(BM_conv_block_unroll);

// BENCHMARK(BM_false_sharing);
// BENCHMARK(BM_no_false_sharing);
//
// BENCHMARK(BM_grid_unordered_block_XY_o1);
// BENCHMARK(BM_PointerGrid_SpinLock);

BENCHMARK(BM_int64_t);
BENCHMARK(BM_int32_t);
BENCHMARK(BM_int8_t);
BENCHMARK(BM_8bit);

BENCHMARK(BM_double_calc);
BENCHMARK(BM_float_calc);

BENCHMARK(BM_floatingpoint);
BENCHMARK(BM_fixedpoint_32);
BENCHMARK(BM_fixedpoint_16);
BENCHMARK(BM_fixedpoint_uint8);
BENCHMARK_MAIN();
