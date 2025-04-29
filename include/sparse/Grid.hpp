#pragma once
#include <iostream>
#include <vector>
#include <bitset>
#include <map>
#include <tuple>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <tbb/spin_mutex.h>
#define N (1024 * 1024)

namespace logn {
          template<typename _Ty = bool>
          struct Grid {
                    using Coord2D = std::pair<std::intptr_t, std::intptr_t>;

                    void create(const std::size_t x, const std::size_t y, const _Ty&  value) {
                              m_data[std::make_pair(x, y)] = value;
                    }
                    template<typename Func> void foreach(Func&& func) {
                              for (auto& [key, value] : m_data) {
                                        func(key.first, key.second, value);
                              }
                    }

                    std::map<Coord2D, _Ty > m_data;
          };
}

namespace o1 {
          template<typename _Ty>
          struct Grid {
                    using Coord2D = std::pair<std::intptr_t, std::intptr_t>;
                    struct Hasher {
                              std::intptr_t operator()(const Coord2D& key) const {
                                        return (key.first * 2718281828) ^ (key.second * 3141592653);
                              }
                    };

                    void create(const std::intptr_t x, const std::intptr_t y, const _Ty& value) {
                              m_data[std::make_pair(x, y)] = value;
                    }
                    template<typename Func> void foreach(Func&& func) {
                              for (auto& [key, value] : m_data) {
                                        func(key.first, key.second, value);
                              }
                    }
                    std::unordered_map< Coord2D, _Ty, Hasher> m_data;
          };
}

namespace o1 {
          template<typename _Ty, std::intptr_t BlockSize>
          struct BlockGrid {
                    static_assert((BlockSize& (BlockSize - 1)) == 0, "BlockSize must be a power of 2");

                    using Coord2D = std::pair<std::intptr_t, std::intptr_t>;
                    struct Hasher {
                              std::intptr_t operator()(const Coord2D& key) const {
                                        return (key.first * 2718281828) ^ (key.second * 3141592653);
                              }
                    };
                 
                    template<typename _Ty, std::intptr_t BlockSize>
                    struct BlockSquare {
                              static constexpr std::intptr_t absmod(const std::intptr_t a, const std::intptr_t divv) {
                                        return a & (divv - 1);
                              }
                              _Ty& operator()(const std::intptr_t x, const std::intptr_t y) {
                                        return m_block[absmod(x, BlockSize)][absmod(y, BlockSize)];
                              }
                              _Ty m_block[BlockSize][BlockSize];
                    };

                    static constexpr std::intptr_t constexpr_log2(std::intptr_t n) {
                              return (n < 2) ? 0 : 1 + constexpr_log2(n >> 1);
                    }
                    static constexpr std::intptr_t right_shift = constexpr_log2(BlockSize);

                    void create(const std::intptr_t x, const std::intptr_t y, const _Ty& value) {
                              auto& block = m_data[std::make_pair(x >> right_shift, y >> right_shift)];
                              block(x, y) = value;
                    }
                    template<typename Func> void foreach(Func&& func) {
                              for (auto& [key, block] : m_data) {
                                        auto& [xB, yB] = key;
                                        for (std::size_t dx = 0; dx < BlockSize; ++dx) {
                                                  for (std::size_t dy = 0; dy < BlockSize; ++dy) {
                                                            func(dx + BlockSize * xB, dy + BlockSize * yB, block(dx, dy));
                                                  }
                                        }
                              }
                    }
                    std::unordered_map<Coord2D, BlockSquare<_Ty, BlockSize>, Hasher> m_data;
          };
}

template<typename _Ty, std::intptr_t GridSize, std::intptr_t BlockSize>
struct PointerGrid {
          template<typename _Ty, std::intptr_t BlockSize>
          struct BlockSquare {
                    static constexpr std::intptr_t constexpr_log2(std::intptr_t n) { return (n < 2) ? 0 : 1 + constexpr_log2(n >> 1); }

                    static constexpr std::intptr_t B = BlockSize;
                    static constexpr std::intptr_t BShift = constexpr_log2(BlockSize);	
                    static constexpr std::intptr_t BMask = BlockSize - 1;		      

                    _Ty& operator()(const std::intptr_t x, const std::intptr_t y) { return m_block[x & BMask][y & BMask]; }
                    _Ty m_block[BlockSize][BlockSize];
          };

          static_assert((BlockSize& (BlockSize - 1)) == 0, "BlockSize must be a power of 2");
          static_assert((GridSize& (GridSize - 1)) == 0, "GridSize must be a power of 2");

          static constexpr std::intptr_t constexpr_log2(std::intptr_t n) {
                    return (n < 2) ? 0 : 1 + constexpr_log2(n >> 1);
          }

          static constexpr std::intptr_t BlockShift = constexpr_log2(BlockSize);
          static constexpr std::intptr_t BlockMask = BlockSize - 1;

          static constexpr std::intptr_t GridShift = constexpr_log2(GridSize);
          static constexpr std::intptr_t GridMask = GridSize - 1;

          struct WriteAccessor {
                    WriteAccessor(PointerGrid & grid) :m_global(grid)
                    {}


                    void write(const std::intptr_t x, const std::intptr_t y, const _Ty& value) {
                              auto x_coord = (x >> BlockShift) & GridMask;
                              auto y_coord = (y >> BlockShift) & GridMask;
                              auto key = std::pair(x_coord, y_coord);
                              auto it = m_cache.find(key);
                              if (it != m_cache.end()) {
                                        (*it->second)(x, y) = value;
                                        return;
                              }
                              auto& block = m_global.m_data[x_coord][y_coord];
                              if (!block) {
                                        std::lock_guard<tbb::spin_mutex> _lck(m_global.m_spinlock[x_coord][y_coord]);
                                        if (!block)
                                                  block = std::make_unique< BlockSquare<_Ty, BlockSize>>();
                              }
                              (*block)(x, y) = value;
                              m_cache.try_emplace(key, block.get());
                    }

                    using Coord2D = std::pair<std::intptr_t, std::intptr_t>;
                    PointerGrid<_Ty, GridSize, BlockSize>& m_global;
                    std::map<Coord2D, BlockSquare<_Ty, BlockSize>*> m_cache;
          };

          WriteAccessor access() { return { *this }; }

          const _Ty& read(const std::intptr_t x, const std::intptr_t y) const {
                    auto& block = m_data[(x >> BlockShift) & GridMask][(y >> BlockShift) & GridMask];
                    return !block ? {} : (*block)(x, y);
          }
          void create(const std::intptr_t x, const std::intptr_t y, const _Ty& value) {
                    auto& block = m_data[(x >> BlockShift) & GridMask][(y >> BlockShift) & GridMask];
                    if (!block) {
                              block = std::make_unique< BlockSquare<_Ty, BlockSize>>();
                    }
                    (*block)(x, y) = value;
          }
          void createByMutexLock(const std::intptr_t x, const std::intptr_t y, const _Ty& value) {
                    auto& block = m_data[(x >> BlockShift) & GridMask][(y >> BlockShift) & GridMask];
                    {
                              std::lock_guard<std::mutex> _lck(m_lock[(x >> BlockShift) & GridMask][(y >> BlockShift) & GridMask]);
                              if (!block) 
                                        block = std::make_unique< BlockSquare<_Ty, BlockSize>>();
                    }
                    (*block)(x, y) = value;
          }
          void createByDoubleCheckLock(const std::intptr_t x, const std::intptr_t y, const _Ty& value) {
                    auto& block = m_data[(x >> BlockShift) & GridMask][(y >> BlockShift) & GridMask];
                    if (!block) {
                              std::lock_guard<std::mutex> _lck(m_lock[(x >> BlockShift) & GridMask][(y >> BlockShift) & GridMask]);
                              if (!block)
                                        block = std::make_unique< BlockSquare<_Ty, BlockSize>>();
                    }
                    (*block)(x, y) = value;
          }
          void createBySpinLock(const std::intptr_t x, const std::intptr_t y, const _Ty& value) {
                    auto& block = m_data[(x >> BlockShift) & GridMask][(y >> BlockShift) & GridMask];
                    if (!block) {
                              std::lock_guard<tbb::spin_mutex> _lck(m_spinlock[(x >> BlockShift) & GridMask][(y >> BlockShift) & GridMask]);
                              if (!block)
                                        block = std::make_unique< BlockSquare<_Ty, BlockSize>>();
                    }
                    (*block)(x, y) = value;
          }
          template<typename Func> void foreach(Func&& func) {
#pragma omp parallel for collapse(2)
                    for (std::size_t xB = 0; xB < GridSize; ++xB) {
                              for (std::size_t yB = 0; yB < GridSize; ++yB) {
                                        auto& block = m_data[xB & GridMask][yB & GridMask];
                                        if (!block) continue;         //empty, because of nullptr

                                        for (std::size_t dx = 0; dx < BlockSize; ++dx) {
                                                  for (std::size_t dy = 0; dy < BlockSize; ++dy) {
                                                            func(dx + xB << GridShift, dy + yB << GridShift, (*block)(dx, dy));
                                                  }
                                        }
                              }
                    }
          }

          void garbageCollection() {
                    for (std::size_t xB = 0; xB < GridSize; ++xB) {
                              for (std::size_t yB = 0; yB < GridSize; ++yB) {
                                        auto& block = m_data[xB & GridMask][yB & GridMask];
                                        if (!block) continue;         //empty, because of nullptr

                                        for (std::size_t dx = 0; dx < BlockSize; ++dx) {
                                                  for (std::size_t dy = 0; dy < BlockSize; ++dy) {
                                                            //has data
                                                            if ((*block)(dx, dy) != _Ty{}) {
                                                                      goto actived_block;
                                                            }
                                                  }
                                        }
deactived_block:
                                        block.reset(nullptr);
actived_block:
                                        ;
                              }
                    }
          }
          std::mutex m_lock[GridSize][GridSize];
          tbb::spin_mutex m_spinlock[GridSize][GridSize];
          std::unique_ptr<BlockSquare<_Ty, BlockSize>> m_data[GridSize][GridSize];
};