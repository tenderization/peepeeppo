#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <pybind11/pybind11.h>

// LEETCODERS AVERT YOUR GAZE
at::Tensor longest_of_cpu(
    const at::Tensor& input,
    int64_t marker,
    int64_t stop) {
    TORCH_CHECK(input.scalar_type() == at::kFloat);
    TORCH_CHECK(input.dim() == 3);
    int batch_size = input.size(0);
    int h = input.size(1);
    int w = input.size(2);
    std::array<int64_t, 1> batchsize = {batch_size};
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
    at::Tensor result = at::zeros(batchsize, options);
    auto inp = input.accessor<float, 3>();
    auto r = result.accessor<float, 1>();
    for (int b = 0; b < batch_size; b++) {
        for (int transpose = 0; transpose < 2; transpose++) {
            int range0 = transpose ? w : h;
            int range1 = transpose ? h : w;
            for (int idx0 = 0; idx0 < range0; idx0++) {
                int curr_longest = 0;
                for (int idx1 = 0; idx1 < range1; idx1++) {
                    int curr_state = transpose ? inp[b][idx1][idx0] : inp[b][idx0][idx1];
                    if (curr_state == marker) {
                        curr_longest++;
                        if (curr_longest > r[b]) {
                            r[b] = curr_longest;
                            if (r[b] >= stop) {
                                goto next;
                            }

                        }
                    } else {
                        curr_longest = 0;
                    }
                }
            }
            auto left = transpose;
            for (int x0 = 0; x0 < w; x0++) {
                int curr_longest = 0;
                for (int i = 0; i < h; i++) {
                    int x = left ? x0 + i : x0 - i;
                    int y = h - 1 - i;
                    if (x >= w || y >= h || x < 0 || y < 0) {
                        break;
                    }
                    int curr_state = inp[b][y][x];
                    if (curr_state == marker) {
                        curr_longest++;
                        if (curr_longest > r[b]) {
                            r[b] = curr_longest;
                            if (r[b] >= stop) {
                                goto next;
                            }
                        }
                    } else {
                        curr_longest = 0;
                    }
                }
                curr_longest = 0;
                for (int i = 0; i < h; i++) {
                    int x = left ? x0 + i : x0 - i;
                    int y = i;
                    if (x >= w || y >= h || x < 0 || y < 0) {
                        break;
                    }
                    int curr_state = inp[b][y][x];
                    if (curr_state == marker) {
                        curr_longest++;
                        if (curr_longest > r[b]) {
                            r[b] = curr_longest;
                            if (r[b] >= stop) {
                                goto next;
                            }
                        }
                    } else {
                        curr_longest = 0;
                    }
                }

            }
        }
        next:;
    }
    return result;
}

TORCH_LIBRARY(peepeeppocpp, m) {
  m.def("longest_of(Tensor input, int marker,  int stop) -> Tensor");
}

TORCH_LIBRARY_IMPL(peepeeppocpp, CPU, m) {
  m.impl("longest_of", &longest_of_cpu);
}

PYBIND11_MODULE(peepeeppocpp, m) {
  m.def("longest_of",
  static_cast<at::Tensor (*)(const at::Tensor&, int64_t, int64_t)>(&longest_of_cpu));
}
