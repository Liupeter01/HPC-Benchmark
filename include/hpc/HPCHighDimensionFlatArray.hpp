/**
 * @file HPCHighDimensionFlatArray.hpp
 * @brief Definition of a high-performance, cache-aligned, padded, N-dimensional
 * flat array.
 */

#pragma once
#ifndef _HPC_HIGH_DIMENSION_FLAT_ARRAY_HPP_
#define _HPC_HIGH_DIMENSION_FLAT_ARRAY_HPP_

#include <array>
#include <hpc/AlignedAlloc.hpp>
#include <vector>

namespace hpc {

/**
 * @brief A high-performance, flat N-dimensional array with padding (ghost
 * cells) and aligned memory.
 *
 * This class stores N-dimensional data using a flat, contiguous memory buffer
 * (row-major layout), optimized for HPC and SIMD workloads. It supports
 * boundary padding (ghost cells) on each side of each dimension via the
 * `Low_Bound` and `High_Bound` template parameters. Memory alignment is
 * guaranteed via a customizable allocator (default is `AlignedAllocator` with
 * configurable alignment).
 *
 * @tparam Dimension     Number of dimensions (must be > 0)
 * @tparam _Ty           Value type stored in the array (e.g. float, double,
 * int)
 * @tparam Low_Bound     Number of ghost cells (padding) on the lower side of
 * each dimension
 * @tparam High_Bound    Number of ghost cells (padding) on the upper side of
 * each dimension (default = Low_Bound)
 * @tparam Alignment     Memory alignment in bytes (default = 16). Should be
 * 16/32 for SSE/AVX.
 * @tparam Alloc         Custom STL-compatible allocator (default =
 * AlignedAllocator<_Ty, Alignment>)
 *
 * @note Indexing follows row-major order: the last dimension changes fastest.
 * @note The array does not perform bounds checking unless using `at()` or
 * `safe_linearize`.
 * @note Indexing assumes padded bounds: valid access indices range from
 * `-Low_Bound` to `dim[i] + High_Bound - 1`.
 *
 * @example
 * @code
 * HPCHighDimensionFlatArray<2, float, 1> array(64, 64);  // 2D array with 1
 * ghost cell padding array(0, 0) = 1.0f; float val = array.at({-1, 0});  //
 * safe access with boundary check
 * @endcode
 */
template <std::size_t Dimension, typename _Ty, std::size_t Low_Bound = 0,
          std::size_t High_Bound = Low_Bound, std::size_t Alignment = 16,
          class Alloc = AlignedAllocator<_Ty, Alignment>>
class HPCHighDimensionFlatArray {
  static_assert(Dimension > 0, "Dimension must larger than zero");

#if _HAS_CXX20
  static_assert(std::is_same_v<std::remove_cvref_t<_Ty>, _Ty>,
                "_Ty must not be cvref");
#else
  static_assert(
      std::is_same_v<std::remove_cv_t<std::remove_reference_t<_Ty>>, _Ty>,
      "_Ty must not be cvref");
#endif

public:
  /**
   * @brief Constructs the array with given per-dimension sizes.
   *
   * @tparam DimForEachLayer Variadic dimension sizes, must match Dimension.
   * @param dims Sizes for each logical dimension (excluding padding).
   */
  template <typename... DimForEachLayer,
            std::enable_if_t<
                (sizeof...(DimForEachLayer) == Dimension &&
                 std::conjunction_v<std::is_integral<DimForEachLayer>...>),
                int> = 0>
  explicit HPCHighDimensionFlatArray(const DimForEachLayer &...dims)
      : HPCHighDimensionFlatArray(std::array<std::size_t, Dimension>{
            static_cast<std::size_t>(dims)...}) {}

  /**
   * @brief Shrinks internal vector capacity to fit its size.
   */
  void shrink_to_fit() { _flat.shrink_to_fit(); }

  /**
   * @brief Returns a mutable pointer to the underlying flat buffer.
   */
  constexpr _Ty *data() noexcept { return _flat.data(); }

  /**
   * @brief Returns a const pointer to the underlying flat buffer.
   */
  constexpr const _Ty *data() const noexcept { return _flat.data(); }

  /**
   * @brief Accesses an element using a dimension array with bounds checking.
   *
   * @param indices Array of indices (with possible ghost cell range).
   * @return Reference to the element at the given location.
   * @throws std::out_of_range if any index is outside padded bounds.
   */
  _Ty &at(const std::array<std::intptr_t, Dimension> &indices) {
    return data()[safe_linearize(indices)];
  }

  /**
   * @brief Direct element access without bounds checking.
   *
   * @tparam Indicies Variadic integral indices (must match Dimension).
   * @param idxs Indices per dimension.
   * @return Reference to the element at the given location.
   */
  template <
      typename... Indicies,
      std::enable_if_t<(sizeof...(Indicies) == Dimension &&
                        std::conjunction_v<std::is_integral<Indicies>...>),
                       int> = 0>
  _Ty &operator()(const Indicies &...idxs) noexcept {
    return data()[unsafe_linearize({static_cast<std::intptr_t>(idxs)...})];
  }

protected:
  /**
   * @brief Resizes the array using a dimension array, preserving ghost cell
   * configuration.
   *
   * @param dim Dimension sizes per axis.
   * @param value Initial value to fill the flat buffer.
   */
  void resize(const std::array<std::size_t, Dimension> &dim,
              const _Ty &value = _Ty{}) {
    const auto [stride, total] = compute_stride_and_total(dim);
    assert(total > 0 && "resize() attempted to create zero-sized flat array!");

    _dim = dim;
    _stride = stride;
    _flat.clear();
    _flat.resize(total, value);
  }

  /**
   * @brief Applies ghost cell offset to an input index.
   * @param val Logical index (can be negative for ghost cells).
   * @return Internal flat index (with offset).
   */
  std::intptr_t padded_index(std::intptr_t val) const noexcept {
    return val + static_cast<std::intptr_t>(Low_Bound);
  }

  /**
   * @brief Computes stride and total number of flat cells (with padding).
   *
   * @param dims Dimension sizes.
   * @return A pair containing the stride array and total element count.
   */
  static constexpr auto compute_stride_and_total(
      const std::array<std::size_t, Dimension> &dims) noexcept
      -> std::pair<std::array<std::size_t, Dimension>, std::size_t> {
    std::array<std::size_t, Dimension> stride = {};
    std::size_t total = 1;
    for (std::size_t i = Dimension; i-- > 0;) {
      stride[i] = total;
      total *= dims[i] + Low_Bound + High_Bound;
    }
    return {stride, total};
  }

  /**
   * @brief Converts a multi-dimensional padded index to a flat offset (without
   * bounds check).
   *
   * @param insert Index per dimension (including ghost access).
   * @return Flat memory offset.
   */
  std::intptr_t unsafe_linearize(
      const std::array<std::intptr_t, Dimension> &insert) const noexcept {
    std::intptr_t result = 0;
    for (std::size_t i = Dimension; i-- > 0;) {
      result += _stride[i] * padded_index(insert[i]);
    }
    return result;
  }

  /**
   * @brief Converts a multi-dimensional padded index to a flat offset (with
   * bounds check).
   *
   * @param insert Index per dimension.
   * @return Flat memory offset.
   * @throws std::out_of_range if any index exceeds bounds.
   */
  std::intptr_t
  safe_linearize(const std::array<std::intptr_t, Dimension> &insert) const {
    std::intptr_t result = 0;
    for (std::size_t i = Dimension; i-- > 0;) {
      if (insert[i] < -static_cast<std::intptr_t>(Low_Bound) ||
          insert[i] >= static_cast<std::intptr_t>(_dim[i] + High_Bound)) {
        throw std::out_of_range("invalid index, out of boundary");
      }
      result += _stride[i] * padded_index(insert[i]);
    }
    return result;
  }

private:
  /// Internal flat buffer, allocated with alignment and padding.
  std::vector<_Ty, Alloc> _flat;

  /// Logical dimensions (excluding ghost cells).
  std::array<std::size_t, Dimension> _dim;

  /// Row-major stride for each dimension (including ghost cells).
  std::array<std::size_t, Dimension> _stride;

  /// Private constructor used by delegating constructors.
  HPCHighDimensionFlatArray(const std::array<std::size_t, Dimension> &array) {
    resize(array);
  }
};

} // namespace hpc

#endif // _HPC_HIGH_DIMENSION_FLAT_ARRAY_HPP_
