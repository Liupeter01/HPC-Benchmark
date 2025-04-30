#pragma once
#include <bitset>
#include <iostream>
#include <map>
#include <tuple>
#include <vector>
#define N (1024 * 1024)

namespace basic {
struct Vector {
  float m_data[N];
  float &at(const std::size_t index) { return m_data[index]; }

  template <typename Func> void foreach (Func &&func) {
    for (int x = 0; x < N; ++x)
      func(x);
  }
};

// struct Matrix {
//           float m_data[N][N];
//           float& at(const std::size_t x, const std::size_t y) { return
//           m_data[x][y]; }

//          template<typename Func>
//          void foreach(Func&& func) {
//                    for (int x = 0; x < N; ++x)
//                              for (int y = 0; y < N; ++y)
//                                        func(x, y);
//          }
//};
} // namespace basic

namespace Random {
struct Matrix {
  std::map<std::pair<std::size_t, std::size_t>, float> m_data;

  float read(std::size_t x, std::size_t y) const {
    const auto key = std::make_pair(x, y);
    auto it = m_data.find(key);
    return it != m_data.end() ? it->second : 0.f;
  }
  void modify(std::size_t x, std::size_t y, float value) {
    const auto key = std::make_pair(x, y);
    auto it = m_data.find(key);
    if (it != m_data.end()) {
      it->second = value;
    }
  }
  void create(std::size_t x, std::size_t y, float value) {
    m_data.emplace(std::make_pair(x, y), value);
  }

  float &at(std::size_t x, std::size_t y) {
    return m_data[std::make_pair(x, y)];
  }
};
} // namespace Random

namespace Sequential {
struct Matrix {
  std::vector<std::tuple<std::size_t, std::size_t, float>> m_data;
  void create(std::size_t x, std::size_t y, float value) {
    m_data.emplace_back(std::make_tuple(x, y, value));
  }
  template <typename Func> void foreach (Func &&func) {
    for (auto &[x, y, value] : m_data) {
      func(x, y, value);
    }
  }
};
} // namespace Sequential

namespace CRS {
struct Matrix {
  using ColLists = std::tuple</*col_ind=*/std::size_t, /*value=*/float>;
  std::vector<ColLists> m_data;
  std::size_t col_offset[N]{};

  void create(const std::size_t row, std::vector<ColLists> &&lists) {
    col_offset[row] = m_data.size();
    for (auto &&item : lists)
      m_data.push_back(std::move(item));
  }
  template <typename Func> void foreach (Func &&func) {
    for (std::size_t row = 0; row < N; ++row) {
      bool flag = row == N - 1; // last line
      std::size_t end = flag ? m_data.size() : col_offset[row + 1];
      for (std::size_t col = col_offset[row]; col < end; ++col) {
        auto &[y, value] = m_data[col];
        func(x, y, value);
      }
    }
  };
};
} // namespace CRS
