"""
Implements N-Dimensional Complex Array
Last updated: 2025-01-26
"""

from algorithm import parallelize, vectorize
import builtin.bool as builtin_bool
import builtin.math as builtin_math
from builtin.type_aliases import Origin
from collections import Dict
from collections.optional import Optional
from memory import UnsafePointer, memset_zero, memcpy
from python import Python, PythonObject
from sys import simdwidthof
from utils import Variant

from numojo.core.complex.complex_simd import ComplexSIMD
from numojo.core.datatypes import TypeCoercion, _concise_dtype_str
from numojo.core.flags import Flags
from numojo.core.item import Item
from numojo.core.ndshape import NDArrayShape
from numojo.core.ndstrides import NDArrayStrides
from numojo.core.utility import (
    _get_offset,
    _traverse_iterative,
    _traverse_iterative_setter,
    to_numpy,
    bool_to_numeric,
)
from numojo.core._math_funcs import Vectorized
import numojo.routines.bitwise as bitwise
from numojo.routines.io.formatting import (
    format_floating_precision,
    format_floating_scientific,
    format_value,
    PrintOptions,
    GLOBAL_PRINT_OPTIONS,
)
import numojo.routines.linalg as linalg
from numojo.routines.linalg.products import matmul
import numojo.routines.logic.comparison as comparison
from numojo.routines.logic.truth import any
from numojo.routines.manipulation import reshape, ravel
import numojo.routines.math.rounding as rounding
import numojo.routines.math.arithmetic as arithmetic
from numojo.routines.math.extrema import max, min
from numojo.routines.math.products import prod, cumprod
from numojo.routines.math.sums import sum, cumsum
import numojo.routines.sorting as sorting
from numojo.routines.statistics.averages import mean


# ===----------------------------------------------------------------------===#
# ComplexNDArray
# ===----------------------------------------------------------------------===#
@value
struct ComplexNDArray[
    cdtype: CDType, *, dtype: DType = CDType.to_dtype[cdtype]()
](Stringable, Representable, CollectionElement, Sized, Writable):
    """
    Represents a Complex N-Dimensional Array.

    Parameters:
        cdtype: Complex data type.
        dtype: Real data type.
    """

    """FIELDS"""
    var _re: NDArray[dtype]
    var _im: NDArray[dtype]

    # It's redundant, but better to have it as fields.
    var ndim: Int
    """Number of Dimensions."""
    var shape: NDArrayShape
    """Size and shape of ComplexNDArray."""
    var size: Int
    """Size of ComplexNDArray."""
    var strides: NDArrayStrides
    """Contains offset, strides."""
    var flags: Flags
    "Information about the memory layout of the array."

    """LIFETIME METHODS"""

    @always_inline("nodebug")
    fn __init__(mut self, owned re: NDArray[dtype], owned im: NDArray[dtype]):
        self._re = re
        self._im = im
        self.ndim = re.ndim
        self.shape = re.shape
        self.size = re.size
        self.strides = re.strides
        self.flags = re.flags

    @always_inline("nodebug")
    fn __init__(
        mut self,
        shape: NDArrayShape,
        order: String = "C",
    ) raises:
        """
        Initialize a ComplexNDArray with given shape.

        The memory is not filled with values.

        Args:
            shape: Variadic shape.
            order: Memory order C or F.

        Example:
        ```mojo
        from numojo.prelude import *
        var A = nm.ComplexNDArray[cf32](Shape(2,3,4))
        ```
        """
        self._re = NDArray[dtype](shape, order)
        self._im = NDArray[dtype](shape, order)
        self.ndim = self._re.ndim
        self.shape = self._re.shape
        self.size = self._re.size
        self.strides = self._re.strides
        self.flags = self._re.flags

    @always_inline("nodebug")
    fn __init__(
        mut self,
        shape: List[Int],
        order: String = "C",
    ) raises:
        """
        (Overload) Initialize a ComplexNDArray with given shape (list of integers).

        Args:
            shape: List of shape.
            order: Memory order C or F.
        """
        self._re = NDArray[dtype](shape, order)
        self._im = NDArray[dtype](shape, order)
        self.ndim = self._re.ndim
        self.shape = self._re.shape
        self.size = self._re.size
        self.strides = self._re.strides
        self.flags = self._re.flags

    @always_inline("nodebug")
    fn __init__(
        mut self,
        shape: VariadicList[Int],
        order: String = "C",
    ) raises:
        """
        (Overload) Initialize a ComplexNDArray with given shape (variadic list of integers).

        Args:
            shape: Variadic List of shape.
            order: Memory order C or F.
        """
        self._re = NDArray[dtype](shape, order)
        self._im = NDArray[dtype](shape, order)
        self.ndim = self._re.ndim
        self.shape = self._re.shape
        self.size = self._re.size
        self.strides = self._re.strides
        self.flags = self._re.flags

    fn __init__(
        mut self,
        shape: List[Int],
        offset: Int,
        strides: List[Int],
    ) raises:
        """
        Extremely specific ComplexNDArray initializer.
        """
        self._re = NDArray[dtype](shape, offset, strides)
        self._im = NDArray[dtype](shape, offset, strides)
        self.ndim = self._re.ndim
        self.shape = self._re.shape
        self.size = self._re.size
        self.strides = self._re.strides
        self.flags = self._re.flags

    fn __init__(
        mut self,
        shape: NDArrayShape,
        ref buffer_re: UnsafePointer[Scalar[dtype]],
        ref buffer_im: UnsafePointer[Scalar[dtype]],
        offset: Int,
        strides: NDArrayStrides,
    ) raises:
        """
        Extremely specific ComplexNDArray initializer.
        """
        self._re = NDArray(shape, buffer_re, offset, strides)
        self._im = NDArray(shape, buffer_re, offset, strides)
        self.ndim = self._re.ndim
        self.shape = self._re.shape
        self.size = self._re.size
        self.strides = self._re.strides
        self.flags = self._re.flags

    @always_inline("nodebug")
    fn __copyinit__(mut self, other: Self):
        """
        Copy other into self.
        """
        self._re = other._re
        self._im = other._im
        self.ndim = other.ndim
        self.shape = other.shape
        self.size = other.size
        self.strides = other.strides
        self.flags = other.flags

    @always_inline("nodebug")
    fn __moveinit__(mut self, owned existing: Self):
        """
        Move other into self.
        """
        self._re = existing._re^
        self._im = existing._im^
        self.ndim = existing.ndim
        self.shape = existing.shape
        self.size = existing.size
        self.strides = existing.strides
        self.flags = existing.flags

    # Explicity deallocation
    # @always_inline("nodebug")
    # fn __del__(owned self):
    #     """
    #     Deallocate memory.
    #     """
    #     self._re.__del__()
    #     self._im.__del__()

    # ===-------------------------------------------------------------------===#
    # Indexing and slicing
    # Getter and setter dunders and other methods
    # ===-------------------------------------------------------------------===#

    fn _setitem(self, *indices: Int, val: ComplexSIMD[cdtype, dtype=dtype]):
        """
        (UNSAFE! for internal use only.)
        Get item at indices and bypass all boundary checks.
        """
        var index_of_buffer: Int = 0
        for i in range(self.ndim):
            index_of_buffer += indices[i] * self.strides._buf[i]
        self._re._buf.ptr[index_of_buffer] = val.re
        self._im._buf.ptr[index_of_buffer] = val.im

    fn __setitem__(mut self, idx: Int, val: Self) raises:
        """
        Set a slice of ComplexNDArray with given ComplexNDArray.

        Example:
        ```mojo
        import numojo as nm
        var A = nm.random.rand[nm.i16](3, 2)
        var B = nm.random.rand[nm.i16](3)
        A[1:4] = B
        ```
        """
        if self.ndim == 0 and val.ndim == 0:
            self._re._buf.ptr.store(0, val._re._buf.ptr.load(0))
            self._im._buf.ptr.store(0, val._im._buf.ptr.load(0))

        var slice_list = List[Slice]()
        if idx >= self.shape[0]:
            var message = String(
                "Error: Slice value exceeds the array shape!\n"
                "The {}-th dimension is of size {}.\n"
                "The slice goes from {} to {}"
            ).format(
                0,
                self.shape[0],
                idx,
                idx + 1,
            )
            raise Error(message)
        slice_list.append(Slice(idx, idx + 1))
        if self.ndim > 1:
            for i in range(1, self.ndim):
                var size_at_dim: Int = self.shape[i]
                slice_list.append(Slice(0, size_at_dim))

        var n_slices: Int = len(slice_list)
        var ndims: Int = 0
        var count: Int = 0
        var spec: List[Int] = List[Int]()
        for i in range(n_slices):
            if slice_list[i].step is None:
                raise Error(String("Step of slice is None."))
            var slice_len: Int = (
                (slice_list[i].end.value() - slice_list[i].start.value())
                / slice_list[i].step.or_else(1)
            ).__int__()
            spec.append(slice_len)
            if slice_len != 1:
                ndims += 1
            else:
                count += 1
        if count == slice_list.__len__():
            ndims = 1

        var nshape: List[Int] = List[Int]()
        var ncoefficients: List[Int] = List[Int]()
        var nstrides: List[Int] = List[Int]()
        var nnum_elements: Int = 1

        var j: Int = 0
        count = 0
        for _ in range(ndims):
            while spec[j] == 1:
                count += 1
                j += 1
            if j >= self.ndim:
                break
            var slice_len: Int = (
                (slice_list[j].end.value() - slice_list[j].start.value())
                / slice_list[j].step.or_else(1)
            ).__int__()
            nshape.append(slice_len)
            nnum_elements *= slice_len
            ncoefficients.append(
                self.strides[j] * slice_list[j].step.or_else(1)
            )
            j += 1

        # TODO: We can remove this check after we have support for broadcasting
        for i in range(ndims):
            if nshape[i] != val.shape[i]:
                var message = String(
                    "Error: Shape mismatch!\n"
                    "Cannot set the array values with given array.\n"
                    "The {}-th dimension of the array is of shape {}.\n"
                    "The {}-th dimension of the value is of shape {}."
                ).format(nshape[i], val.shape[i])
                raise Error(message)

        var noffset: Int = 0
        if self.flags["C_CONTIGUOUS"]:
            noffset = 0
            for i in range(ndims):
                var temp_stride: Int = 1
                for j in range(i + 1, ndims):
                    temp_stride *= nshape[j]
                nstrides.append(temp_stride)
            for i in range(slice_list.__len__()):
                noffset += slice_list[i].start.value() * self.strides[i]
        elif self.flags["F_CONTIGUOUS"]:
            noffset = 0
            nstrides.append(1)
            for i in range(0, ndims - 1):
                nstrides.append(nstrides[i] * nshape[i])
            for i in range(slice_list.__len__()):
                noffset += slice_list[i].start.value() * self.strides[i]

        var index = List[Int]()
        for _ in range(ndims):
            index.append(0)

        _traverse_iterative_setter[dtype](
            val._re, self._re, nshape, ncoefficients, nstrides, noffset, index
        )
        _traverse_iterative_setter[dtype](
            val._im, self._im, nshape, ncoefficients, nstrides, noffset, index
        )

    fn __setitem__(
        mut self, index: Item, val: ComplexSIMD[cdtype, dtype=dtype]
    ) raises:
        """
        Set the value at the index list.
        """
        if index.__len__() != self.ndim:
            var message = String(
                "Error: Length of `index` does not match the number of"
                " dimensions!\n"
                "Length of indices is {}.\n"
                "The array dimension is {}."
            ).format(index.__len__(), self.ndim)
            raise Error(message)
        for i in range(index.__len__()):
            if index[i] >= self.shape[i]:
                var message = String(
                    "Error: `index` exceeds the size!\n"
                    "For {}-th dimension:\n"
                    "The index value is {}.\n"
                    "The size of the corresponding dimension is {}"
                ).format(i, index[i], self.shape[i])
                raise Error(message)
        var idx: Int = _get_offset(index, self.strides)
        self._re._buf.ptr.store(idx, val.re)
        self._im._buf.ptr.store(idx, val.im)

    fn __setitem__(
        mut self,
        mask: ComplexNDArray[cdtype, dtype=dtype],
        value: ComplexSIMD[cdtype, dtype=dtype],
    ) raises:
        """
        Set the value of the array at the indices where the mask is true.
        """
        if (
            mask.shape != self.shape
        ):  # this behaviour could be removed potentially
            raise Error("Mask and array must have the same shape")

        for i in range(mask.size):
            if mask._re._buf.ptr.load[width=1](i):
                self._re._buf.ptr.store(i, value.re)
            if mask._im._buf.ptr.load[width=1](i):
                self._im._buf.ptr.store(i, value.im)

    fn __setitem__(mut self, owned *slices: Slice, val: Self) raises:
        """
        Retreive slices of an ComplexNDArray from variadic slices.

        Example:
            `arr[1:3, 2:4]` returns the corresponding sliced ComplexNDArray (2 x 2).
        """
        var slice_list: List[Slice] = List[Slice]()
        for i in range(slices.__len__()):
            slice_list.append(slices[i])
        # self.__setitem__(slices=slice_list, val=val)
        self[slice_list] = val

    fn __setitem__(mut self, owned slices: List[Slice], val: Self) raises:
        """
        Sets the slices of an ComplexNDArray from list of slices and ComplexNDArray.

        Example:
            `arr[1:3, 2:4]` returns the corresponding sliced ComplexNDArray (2 x 2).
        """
        var n_slices: Int = len(slices)
        var ndims: Int = 0
        var count: Int = 0
        var spec: List[Int] = List[Int]()
        var slice_list: List[Slice] = self._adjust_slice(slices)
        for i in range(n_slices):
            if (
                slice_list[i].start.value() >= self.shape[i]
                or slice_list[i].end.value() > self.shape[i]
            ):
                var message = String(
                    "Error: Slice value exceeds the array shape!\n"
                    "The {}-th dimension is of size {}.\n"
                    "The slice goes from {} to {}"
                ).format(
                    i,
                    self.shape[i],
                    slice_list[i].start.value(),
                    slice_list[i].end.value(),
                )
                raise Error(message)
            # if slice_list[i].step is None:
            #     raise Error(String("Step of slice is None."))
            var slice_len: Int = (
                (slice_list[i].end.value() - slice_list[i].start.value())
                / slice_list[i].step.or_else(1)
            ).__int__()
            spec.append(slice_len)
            if slice_len != 1:
                ndims += 1
            else:
                count += 1
        if count == slice_list.__len__():
            ndims = 1

        var nshape: List[Int] = List[Int]()
        var ncoefficients: List[Int] = List[Int]()
        var nstrides: List[Int] = List[Int]()
        var nnum_elements: Int = 1

        var j: Int = 0
        count = 0
        for _ in range(ndims):
            while spec[j] == 1:
                count += 1
                j += 1
            if j >= self.ndim:
                break
            var slice_len: Int = (
                (slice_list[j].end.value() - slice_list[j].start.value())
                / slice_list[j].step.or_else(1)
            ).__int__()
            nshape.append(slice_len)
            nnum_elements *= slice_len
            ncoefficients.append(
                self.strides[j] * slice_list[j].step.or_else(1)
            )
            j += 1

        # TODO: We can remove this check after we have support for broadcasting
        for i in range(ndims):
            if nshape[i] != val.shape[i]:
                var message = String(
                    "Error: Shape mismatch!\n"
                    "For {}-th dimension: \n"
                    "The size of the array is {}.\n"
                    "The size of the input value is {}."
                ).format(i, nshape[i], val.shape[i])
                raise Error(message)

        var noffset: Int = 0
        if self.flags["C_CONTIGUOUS"]:
            noffset = 0
            for i in range(ndims):
                var temp_stride: Int = 1
                for j in range(i + 1, ndims):  # temp
                    temp_stride *= nshape[j]
                nstrides.append(temp_stride)
            for i in range(slice_list.__len__()):
                noffset += slice_list[i].start.value() * self.strides[i]
        elif self.flags["F_CONTIGUOUS"]:
            noffset = 0
            nstrides.append(1)
            for i in range(0, ndims - 1):
                nstrides.append(nstrides[i] * nshape[i])
            for i in range(slice_list.__len__()):
                noffset += slice_list[i].start.value() * self.strides[i]

        var index = List[Int]()
        for _ in range(ndims):
            index.append(0)

        _traverse_iterative_setter[dtype](
            val._re, self._re, nshape, ncoefficients, nstrides, noffset, index
        )
        _traverse_iterative_setter[dtype](
            val._im, self._im, nshape, ncoefficients, nstrides, noffset, index
        )

    ### compiler doesn't accept this.
    # fn __setitem__(self, owned *slices: Variant[Slice, Int], val: NDArray[dtype]) raises:
    #     """
    #     Get items by a series of either slices or integers.
    #     """
    #     var n_slices: Int = slices.__len__()
    #     if n_slices > self.ndim:
    #         raise Error("Error: No of slices greater than rank of array")
    #     var slice_list: List[Slice] = List[Slice]()

    #     var count_int = 0
    #     for i in range(len(slices)):
    #         if slices[i].isa[Slice]():
    #             slice_list.append(slices[i]._get_ptr[Slice]()[0])
    #         elif slices[i].isa[Int]():
    #             count_int += 1
    #             var int: Int = slices[i]._get_ptr[Int]()[0]
    #             slice_list.append(Slice(int, int + 1))

    #     if n_slices < self.ndim:
    #         for i in range(n_slices, self.ndim):
    #             var size_at_dim: Int = self.shape[i]
    #             slice_list.append(Slice(0, size_at_dim))

    #     self.__setitem__(slices=slice_list, val=val)

    fn __setitem__(self, index: NDArray[DType.index], val: Self) raises:
        """
        Returns the items of the ComplexNDArray from an array of indices.

        Refer to `__getitem__(self, index: List[Int])`.
        """

        for i in range(len(index)):
            self._re.store(
                Int(index.load(i)), rebind[Scalar[dtype]](val._re.load(i))
            )
            self._im.store(
                Int(index.load(i)), rebind[Scalar[dtype]](val._im.load(i))
            )

    fn __setitem__(
        mut self,
        mask: ComplexNDArray[cdtype, dtype=dtype],
        val: ComplexNDArray[cdtype, dtype=dtype],
    ) raises:
        """
        Set the value of the ComplexNDArray at the indices where the mask is true.
        """
        if (
            mask.shape != self.shape
        ):  # this behavious could be removed potentially
            var message = String(
                "Shape of mask does not match the shape of array."
            )
            raise Error(message)

        for i in range(mask.size):
            if mask._re._buf.ptr.load(i):
                self._re._buf.ptr.store(i, val._re._buf.ptr.load(i))
            if mask._im._buf.ptr.load(i):
                self._im._buf.ptr.store(i, val._im._buf.ptr.load(i))

    # ===-------------------------------------------------------------------===#
    # Getter dunders and other getter methods
    # ===-------------------------------------------------------------------===#

    fn _getitem(self, *indices: Int) -> ComplexSIMD[cdtype, dtype=dtype]:
        """
        (UNSAFE! for internal use only.)
        Get item at indices and bypass all boundary checks.
        """
        var index_of_buffer: Int = 0
        for i in range(self.ndim):
            index_of_buffer += indices[i] * self.strides._buf[i]
        return ComplexSIMD[cdtype, dtype=dtype](
            re=self._re._buf.ptr.load[width=1](index_of_buffer),
            im=self._im._buf.ptr.load[width=1](index_of_buffer),
        )

    fn __getitem__(self, idx: Int) raises -> Self:
        """
        Retreive a slice of the ComplexNDArray corresponding to the index at the first dimension.

        Example:
            `arr[1]` returns the second row of the ComplexNDArray.
        """

        var slice_list = List[Slice]()
        slice_list.append(Slice(idx, idx + 1))

        # 0-d array always return itself
        if self.ndim == 0:
            return self

        if self.ndim > 1:
            for i in range(1, self.ndim):
                var size_at_dim: Int = self.shape[i]
                slice_list.append(Slice(0, size_at_dim))

        var narr: Self = self[slice_list]

        if self.ndim == 1:
            narr.ndim = 0
            narr.shape._buf[0] = 0

        return narr

    fn __getitem__(
        self, index: Item
    ) raises -> ComplexSIMD[cdtype, dtype=dtype]:
        """
        Get the value at the index list.
        """
        if index.__len__() != self.ndim:
            var message = String(
                "Error: Length of `index` do not match the number of"
                " dimensions!\n"
                "Length of indices is {}.\n"
                "The number of dimensions is {}."
            ).format(index.__len__(), self.ndim)
            raise Error(message)
        for i in range(index.__len__()):
            if index[i] >= self.shape[i]:
                var message = String(
                    "Error: `index` exceeds the size!\n"
                    "For {}-the mension:\n"
                    "The index is {}.\n"
                    "The size of the dimensions is {}"
                ).format(i, index[i], self.shape[i])
                raise Error(message)
        var idx: Int = _get_offset(index, self.strides)
        return ComplexSIMD[cdtype, dtype=dtype](
            re=self._re._buf.ptr.load[width=1](idx),
            im=self._im._buf.ptr.load[width=1](idx),
        )

    fn _adjust_slice(self, slice_list: List[Slice]) raises -> List[Slice]:
        """
        Adjusts the slice values to lie within 0 and dim.
        """
        var n_slices: Int = slice_list.__len__()
        var slices = List[Slice]()
        for i in range(n_slices):
            if i >= self.ndim:
                raise Error("Error: Number of slices exceeds array dimensions")

            var start: Int = 0
            var end: Int = self.shape[i]
            var step: Int = 1
            if slice_list[i].start is not None:
                start = slice_list[i].start.value()
                if start < 0:
                    # start += self.shape[i]
                    raise Error(
                        "Error: Negative indexing in slices not supported"
                        " currently"
                    )

            if slice_list[i].end is not None:
                end = slice_list[i].end.value()
                if end < 0:
                    # end += self.shape[i] + 1
                    raise Error(
                        "Error: Negative indexing in slices not supported"
                        " currently"
                    )
            step = slice_list[i].step.or_else(1)
            if step == 0:
                raise Error("Error: Slice step cannot be zero")

            slices.append(
                Slice(
                    start=Optional(start),
                    end=Optional(end),
                    step=Optional(step),
                )
            )

        return slices^

    fn __getitem__(self, owned *slices: Slice) raises -> Self:
        """
        Retreive slices of a ComplexNDArray from variadic slices.

        Example:
            `arr[1:3, 2:4]` returns the corresponding sliced ComplexNDArray (2 x 2).
        """

        var n_slices: Int = slices.__len__()
        if n_slices > self.ndim:
            raise Error("Error: No of slices exceed the array dimensions.")
        var slice_list: List[Slice] = List[Slice]()
        for i in range(len(slices)):
            slice_list.append(slices[i])

        if n_slices < self.ndim:
            for i in range(n_slices, self.ndim):
                slice_list.append(Slice(0, self.shape[i]))

        var narr: Self = self[slice_list]
        return narr

    fn __getitem__(self, owned slice_list: List[Slice]) raises -> Self:
        """
        Retreive slices of a ComplexNDArray from list of slices.

        Example:
            `arr[1:3, 2:4]` returns the corresponding sliced ComplexNDArray (2 x 2).
        """

        var n_slices: Int = slice_list.__len__()
        if n_slices > self.ndim or n_slices < self.ndim:
            raise Error("Error: No of slices do not match shape")

        var ndims: Int = 0
        var spec: List[Int] = List[Int]()
        var count: Int = 0

        var slices: List[Slice] = self._adjust_slice(slice_list)
        for i in range(slices.__len__()):
            if (
                slices[i].start.value() >= self.shape[i]
                or slices[i].end.value() > self.shape[i]
            ):
                raise Error("Error: Slice value exceeds the array shape")
            var slice_len: Int = len(
                range(
                    slices[i].start.value(),
                    slices[i].end.value(),
                    slices[i].step.or_else(1),
                )
            )
            spec.append(slice_len)
            if slice_len != 1:
                ndims += 1
            else:
                count += 1
        if count == slices.__len__():
            ndims = 1

        var nshape: List[Int] = List[Int]()
        var ncoefficients: List[Int] = List[Int]()
        var nstrides: List[Int] = List[Int]()
        var nnum_elements: Int = 1

        var j: Int = 0
        count = 0
        for _ in range(ndims):
            while spec[j] == 1:
                count += 1
                j += 1
            if j >= self.ndim:
                break
            var slice_len: Int = len(
                range(
                    slices[j].start.value(),
                    slices[j].end.value(),
                    slices[j].step.or_else(1),
                )
            )
            nshape.append(slice_len)
            nnum_elements *= slice_len
            ncoefficients.append(self.strides[j] * slices[j].step.value())
            j += 1

        if count == slices.__len__():
            nshape.append(1)
            nnum_elements = 1
            ncoefficients.append(1)

        var noffset: Int = 0
        if self.flags["C_CONTIGUOUS"]:
            noffset = 0
            for i in range(ndims):
                var temp_stride: Int = 1
                for j in range(i + 1, ndims):  # temp
                    temp_stride *= nshape[j]
                nstrides.append(temp_stride)
            for i in range(slices.__len__()):
                noffset += slices[i].start.value() * self.strides[i]

        elif self.flags["F_CONTIGUOUS"]:
            noffset = 0
            nstrides.append(1)
            for i in range(0, ndims - 1):
                nstrides.append(nstrides[i] * nshape[i])
            for i in range(slices.__len__()):
                noffset += slices[i].start.value() * self.strides[i]

        var narr = Self(
            offset=noffset,
            shape=nshape,
            strides=nstrides,
        )

        var index = List[Int]()
        for _ in range(ndims):
            index.append(0)

        _traverse_iterative[dtype](
            self._re,
            narr._re,
            nshape,
            ncoefficients,
            nstrides,
            noffset,
            index,
            0,
        )
        _traverse_iterative[dtype](
            self._im,
            narr._im,
            nshape,
            ncoefficients,
            nstrides,
            noffset,
            index,
            0,
        )

        return narr

    fn __getitem__(self, owned *slices: Variant[Slice, Int]) raises -> Self:
        """
        Get items by a series of either slices or integers.

        Args:
            slices: A series of either Slice or Int.

        Returns:
            A ComplexNDArray with a smaller or equal dimension of the original one.
        """

        var n_slices: Int = slices.__len__()
        if n_slices > self.ndim:
            raise Error(
                String(
                    "Error: number of slices {} \n"
                    "is greater than number of dimension of array {}!"
                ).format(n_slices, self.ndim)
            )
        var slice_list: List[Slice] = List[Slice]()

        var count_int = 0  # Count the number of Int in the argument

        for i in range(len(slices)):
            if slices[i].isa[Slice]():
                slice_list.append(slices[i]._get_ptr[Slice]()[0])
            elif slices[i].isa[Int]():
                count_int += 1
                var int: Int = slices[i]._get_ptr[Int]()[0]
                slice_list.append(Slice(int, int + 1))

        if n_slices < self.ndim:
            for i in range(n_slices, self.ndim):
                var size_at_dim: Int = self.shape[i]
                slice_list.append(Slice(0, size_at_dim))

        var narr: Self = self[slice_list]

        if count_int == self.ndim:
            narr.ndim = 0
            narr.shape._buf[0] = 0

        return narr

    fn __getitem__(self, index: List[Int]) raises -> Self:
        """
        Get items of ComplexNDArray from a list of indices.

        It always gets the first dimension.
        ```

        Args:
            index: List[Int].

        Returns:
            ComplexNDArray with items from the list of indices.
        """

        # Shape of the result should be
        # Number of indice * shape from dim-1
        # So just change the first number of the ndshape
        var ndshape = self.shape
        ndshape[0] = len(index)
        ndsize = 1
        for i in range(ndshape.ndim):
            ndsize *= Int(ndshape._buf[i])
        var result = ComplexNDArray[cdtype, dtype=dtype](ndshape)
        var size_per_item = ndsize // len(index)

        # Fill in the values
        for i in range(len(index)):
            for j in range(size_per_item):
                result._re._buf.ptr.store(
                    i * size_per_item + j, self[Int(index[i])].item(j).re
                )
                result._im._buf.ptr.store(
                    i * size_per_item + j, self[Int(index[i])].item(j).im
                )

        return result

    fn __getitem__(self, index: NDArray[DType.index]) raises -> Self:
        """
        Get items of ComplexNDArray from an array of indices.

        Refer to `__getitem__(self, index: List[Int])`.
        """

        var new_index = List[Int]()
        for i in index:
            new_index.append(Int(i.item(0)))

        return self[new_index]

    fn __getitem__(self, mask: NDArray[DType.bool]) raises -> Self:
        """
        Get items of ComplexNDArray corresponding to a mask.

        Example:
            ```
            var A = numojo.core.NDArray[numojo.i16](6, random=True)
            var mask = A > 0
            print(A)
            print(mask)
            print(A[mask])
            ```

        Args:
            mask: NDArray with Dtype.bool.

        Returns:
            ComplexNDArray with items from the mask.
        """
        var true: List[Int] = List[Int]()
        for i in range(mask.size):
            if mask._buf.ptr.load[width=1](i):
                true.append(i)

        var result = Self(Shape(true.__len__()))
        for i in range(true.__len__()):
            result._re._buf.ptr.store(i, self._re.load(true[i]))
            result._im._buf.ptr.store(i, self._im.load(true[i]))

        return result

    fn __pos__(self) raises -> Self:
        """
        Unary positive returns self unless boolean type.
        """
        if self.dtype is DType.bool:
            raise Error(
                "complex_ndarray:ComplexNDArray:__pos__: pos does not accept"
                " bool type arrays"
            )
        return self

    fn __neg__(self) raises -> Self:
        """
        Unary negative returns self unless boolean type.

        For bolean use `__invert__`(~)
        """
        if self.dtype is DType.bool:
            raise Error(
                "complex_ndarray:ComplexNDArray:__neg__: neg does not accept"
                " bool type arrays"
            )
        return self * ComplexSIMD[cdtype, dtype=dtype](-1.0, -1.0)

    @always_inline("nodebug")
    fn __eq__(self, other: Self) raises -> NDArray[DType.bool]:
        """
        Itemwise equivalence.
        """
        return comparison.equal[dtype](
            self._re, other._re
        ) and comparison.equal[dtype](self._im, other._im)

    @always_inline("nodebug")
    fn __eq__(
        self, other: ComplexSIMD[cdtype, dtype=dtype]
    ) raises -> NDArray[DType.bool]:
        """
        Itemwise equivalence between scalar and ComplexNDArray.
        """
        return comparison.equal[dtype](self._re, other.re) and comparison.equal[
            dtype
        ](self._im, other.im)

    @always_inline("nodebug")
    fn __ne__(self, other: Self) raises -> NDArray[DType.bool]:
        """
        Itemwise non-equivalence.
        """
        return comparison.not_equal[dtype](
            self._re, other._re
        ) and comparison.not_equal[dtype](self._im, other._im)

    @always_inline("nodebug")
    fn __ne__(
        self, other: ComplexSIMD[cdtype, dtype=dtype]
    ) raises -> NDArray[DType.bool]:
        """
        Itemwise non-equivalence between scalar and ComplexNDArray.
        """
        return comparison.not_equal[dtype](
            self._re, other.re
        ) and comparison.not_equal[dtype](self._im, other.im)

    """ ARITHMETIC OPERATIONS """

    fn __add__(self, other: ComplexSIMD[cdtype, dtype=dtype]) raises -> Self:
        """
        Enables `ComplexNDArray + ComplexSIMD`.
        """
        var real: NDArray[dtype] = math.add[dtype](self._re, other.re)
        var imag: NDArray[dtype] = math.add[dtype](self._im, other.im)
        return Self(real, imag)

    fn __add__(self, other: Scalar[dtype]) raises -> Self:
        """
        Enables `ComplexNDArray + Scalar`.
        """
        var real: NDArray[dtype] = math.add[dtype](self._re, other)
        var imag: NDArray[dtype] = math.add[dtype](self._im, other)
        return Self(real, imag)

    fn __add__(self, other: Self) raises -> Self:
        """
        Enables `ComplexNDArray + ComplexNDArray`.
        """
        var real: NDArray[dtype] = math.add[dtype](self._re, other._re)
        var imag: NDArray[dtype] = math.add[dtype](self._im, other._im)
        return Self(real, imag)

    fn __add__(self, other: NDArray[dtype]) raises -> Self:
        """
        Enables `ComplexNDArray + NDArray`.
        """
        var real: NDArray[dtype] = math.add[dtype](self._re, other)
        var imag: NDArray[dtype] = math.add[dtype](self._im, other)
        return Self(real, imag)

    fn __radd__(
        mut self, other: ComplexSIMD[cdtype, dtype=dtype]
    ) raises -> Self:
        """
        Enables `ComplexSIMD + ComplexNDArray`.
        """
        var real: NDArray[dtype] = math.add[dtype](self._re, other.re)
        var imag: NDArray[dtype] = math.add[dtype](self._im, other.im)
        return Self(real, imag)

    fn __radd__(mut self, other: Scalar[dtype]) raises -> Self:
        """
        Enables `Scalar + ComplexNDArray`.
        """
        var real: NDArray[dtype] = math.add[dtype](
            self._re, other.cast[dtype]()
        )
        var imag: NDArray[dtype] = math.add[dtype](
            self._im, other.cast[dtype]()
        )
        return Self(real, imag)

    fn __radd__(mut self, other: NDArray[dtype]) raises -> Self:
        """
        Enables `NDArray + ComplexNDArray`.
        """
        var real: NDArray[dtype] = math.add[dtype](self._re, other)
        var imag: NDArray[dtype] = math.add[dtype](self._im, other)
        return Self(real, imag)

    fn __iadd__(mut self, other: ComplexSIMD[cdtype, dtype=dtype]) raises:
        """
        Enables `ComplexNDArray += ComplexSIMD`.
        """
        self._re += other.re
        self._im += other.im

    fn __iadd__(mut self, other: Scalar[dtype]) raises:
        """
        Enables `ComplexNDArray += Scalar`.
        """
        self._re += other
        self._im += other

    fn __iadd__(mut self, other: Self) raises:
        """
        Enables `ComplexNDArray += ComplexNDArray`.
        """
        self._re += other._re
        self._im += other._im

    fn __iadd__(mut self, other: NDArray[dtype]) raises:
        """
        Enables `ComplexNDArray += NDArray`.
        """
        self._re += other
        self._im += other

    fn __sub__(self, other: ComplexSIMD[cdtype, dtype=dtype]) raises -> Self:
        """
        Enables `ComplexNDArray - ComplexSIMD`.
        """
        var real: NDArray[dtype] = math.sub[dtype](self._re, other.re)
        var imag: NDArray[dtype] = math.sub[dtype](self._im, other.im)
        return Self(real, imag)

    fn __sub__(self, other: Scalar[dtype]) raises -> Self:
        """
        Enables `ComplexNDArray - Scalar`.
        """
        var real: NDArray[dtype] = math.sub[dtype](
            self._re, other.cast[dtype]()
        )
        var imag: NDArray[dtype] = math.sub[dtype](
            self._im, other.cast[dtype]()
        )
        return Self(real, imag)

    fn __sub__(self, other: Self) raises -> Self:
        """
        Enables `ComplexNDArray - ComplexNDArray`.
        """
        var real: NDArray[dtype] = math.sub[dtype](self._re, other._re)
        var imag: NDArray[dtype] = math.sub[dtype](self._im, other._im)
        return Self(real, imag)

    fn __sub__(self, other: NDArray[dtype]) raises -> Self:
        """
        Enables `ComplexNDArray - NDArray`.
        """
        var real: NDArray[dtype] = math.sub[dtype](self._re, other)
        var imag: NDArray[dtype] = math.sub[dtype](self._im, other)
        return Self(real, imag)

    fn __rsub__(
        mut self, other: ComplexSIMD[cdtype, dtype=dtype]
    ) raises -> Self:
        """
        Enables `ComplexSIMD - ComplexNDArray`.
        """
        var real: NDArray[dtype] = math.sub[dtype](other.re, self._re)
        var imag: NDArray[dtype] = math.sub[dtype](other.im, self._im)
        return Self(real, imag)

    fn __rsub__(mut self, other: Scalar[dtype]) raises -> Self:
        """
        Enables `Scalar - ComplexNDArray`.
        """
        var real: NDArray[dtype] = math.sub[dtype](other, self._re)
        var imag: NDArray[dtype] = math.sub[dtype](other, self._im)
        return Self(real, imag)

    fn __rsub__(mut self, other: NDArray[dtype]) raises -> Self:
        """
        Enables `NDArray - ComplexNDArray`.
        """
        var real: NDArray[dtype] = math.sub[dtype](other, self._re)
        var imag: NDArray[dtype] = math.sub[dtype](other, self._im)
        return Self(real, imag)

    fn __isub__(mut self, other: ComplexSIMD[cdtype, dtype=dtype]) raises:
        """
        Enables `ComplexNDArray -= ComplexSIMD`.
        """
        self._re -= other.re
        self._im -= other.im

    fn __isub__(mut self, other: Scalar[dtype]) raises:
        """
        Enables `ComplexNDArray -= Scalar`.
        """
        self._re -= other
        self._im -= other

    fn __isub__(mut self, other: Self) raises:
        """
        Enables `ComplexNDArray -= ComplexNDArray`.
        """
        self._re -= other._re
        self._im -= other._im

    fn __isub__(mut self, other: NDArray[dtype]) raises:
        """
        Enables `ComplexNDArray -= NDArray`.
        """
        self._re -= other
        self._im -= other

    fn __matmul__(self, other: Self) raises -> Self:
        var re_re: NDArray[dtype] = linalg.matmul[dtype](self._re, other._re)
        var im_im: NDArray[dtype] = linalg.matmul[dtype](self._im, other._im)
        var re_im: NDArray[dtype] = linalg.matmul[dtype](self._re, other._im)
        var im_re: NDArray[dtype] = linalg.matmul[dtype](self._im, other._re)
        return Self(re_re - im_im, re_im + im_re)

    fn __mul__(self, other: ComplexSIMD[cdtype, dtype=dtype]) raises -> Self:
        """
        Enables `ComplexNDArray * ComplexSIMD`.
        """
        var re_re: NDArray[dtype] = math.mul[dtype](self._re, other.re)
        var im_im: NDArray[dtype] = math.mul[dtype](self._im, other.re)
        var re_im: NDArray[dtype] = math.mul[dtype](self._re, other.im)
        var im_re: NDArray[dtype] = math.mul[dtype](self._im, other.im)
        return Self(re_re - im_im, re_im + im_re)

    fn __mul__(self, other: Scalar[dtype]) raises -> Self:
        """
        Enables `ComplexNDArray * Scalar`.
        """
        var real: NDArray[dtype] = math.mul[dtype](self._re, other)
        var imag: NDArray[dtype] = math.mul[dtype](self._im, other)
        return Self(real, imag)

    fn __mul__(self, other: Self) raises -> Self:
        """
        Enables `ComplexNDArray * ComplexNDArray`.
        """
        var re_re: NDArray[dtype] = math.mul[dtype](self._re, other._re)
        var im_im: NDArray[dtype] = math.mul[dtype](self._im, other._im)
        var re_im: NDArray[dtype] = math.mul[dtype](self._re, other._im)
        var im_re: NDArray[dtype] = math.mul[dtype](self._im, other._re)
        return Self(re_re - im_im, re_im + im_re)

    fn __mul__(self, other: NDArray[dtype]) raises -> Self:
        """
        Enables `ComplexNDArray * NDArray`.
        """
        var real: NDArray[dtype] = math.mul[dtype](self._re, other)
        var imag: NDArray[dtype] = math.mul[dtype](self._im, other)
        return Self(real, imag)

    fn __rmul__(self, other: ComplexSIMD[cdtype, dtype=dtype]) raises -> Self:
        """
        Enables `ComplexSIMD * ComplexNDArray`.
        """
        var real: NDArray[dtype] = math.mul[dtype](self._re, other.re)
        var imag: NDArray[dtype] = math.mul[dtype](self._im, other.re)
        return Self(real, imag)

    fn __rmul__(self, other: Scalar[dtype]) raises -> Self:
        """
        Enables `Scalar * ComplexNDArray`.
        """
        var real: NDArray[dtype] = math.mul[dtype](self._re, other)
        var imag: NDArray[dtype] = math.mul[dtype](self._im, other)
        return Self(real, imag)

    fn __rmul__(self, other: NDArray[dtype]) raises -> Self:
        """
        Enables `NDArray * ComplexNDArray`.
        """
        var real: NDArray[dtype] = math.mul[dtype](self._re, other)
        var imag: NDArray[dtype] = math.mul[dtype](self._im, other)
        return Self(real, imag)

    fn __imul__(mut self, other: ComplexSIMD[cdtype, dtype=dtype]) raises:
        """
        Enables `ComplexNDArray *= ComplexSIMD`.
        """
        self._re *= other.re
        self._im *= other.im

    fn __imul__(mut self, other: Scalar[dtype]) raises:
        """
        Enables `ComplexNDArray *= Scalar`.
        """
        self._re *= other
        self._im *= other

    fn __imul__(mut self, other: Self) raises:
        """
        Enables `ComplexNDArray *= ComplexNDArray`.
        """
        self._re *= other._re
        self._im *= other._im

    fn __imul__(mut self, other: NDArray[dtype]) raises:
        """
        Enables `ComplexNDArray *= NDArray`.
        """
        self._re *= other
        self._im *= other

    fn __truediv__(
        self, other: ComplexSIMD[cdtype, dtype=dtype]
    ) raises -> Self:
        """
        Enables `ComplexNDArray / ComplexSIMD`.
        """
        var other_square = other * other.conj()
        var result = self * other.conj() * (1.0 / other_square.re)
        return result^

    fn __truediv__(self, other: Scalar[dtype]) raises -> Self:
        """
        Enables `ComplexNDArray / ComplexSIMD`.
        """
        var real: NDArray[dtype] = math.div[dtype](self._re, other)
        var imag: NDArray[dtype] = math.div[dtype](self._im, other)
        return Self(real, imag)

    fn __truediv__(
        self, other: ComplexNDArray[cdtype, dtype=dtype]
    ) raises -> Self:
        """
        Enables `ComplexNDArray / ComplexNDArray`.
        """
        var denom = other * other.conj()
        var numer = self * other.conj()
        var real = numer._re / denom._re
        var imag = numer._im / denom._re
        return Self(real, imag)

    fn __truediv__(self, other: NDArray[dtype]) raises -> Self:
        """
        Enables `ComplexNDArray / NDArray`.
        """
        var real: NDArray[dtype] = math.div[dtype](self._re, other)
        var imag: NDArray[dtype] = math.div[dtype](self._im, other)
        return Self(real, imag)

    fn __rtruediv__(
        mut self, other: ComplexSIMD[cdtype, dtype=dtype]
    ) raises -> Self:
        """
        Enables `ComplexSIMD / ComplexNDArray`.
        """
        var denom = other * other.conj()
        var numer = self * other.conj()
        var real = numer._re / denom.re
        var imag = numer._im / denom.re
        return Self(real, imag)

    fn __rtruediv__(mut self, other: Scalar[dtype]) raises -> Self:
        """
        Enables `Scalar / ComplexNDArray`.
        """
        var denom = self * self.conj()
        var numer = self.conj() * other
        var real = numer._re / denom._re
        var imag = numer._im / denom._re
        return Self(real, imag)

    fn __rtruediv__(mut self, other: NDArray[dtype]) raises -> Self:
        """
        Enables `NDArray / ComplexNDArray`.
        """
        var denom = self * self.conj()
        var numer = self.conj() * other
        var real = numer._re / denom._re
        var imag = numer._im / denom._re
        return Self(real, imag)

    fn __itruediv__(mut self, other: ComplexSIMD[cdtype, dtype=dtype]) raises:
        """
        Enables `ComplexNDArray /= ComplexSIMD`.
        """
        self._re /= other.re
        self._im /= other.im

    fn __itruediv__(mut self, other: Scalar[dtype]) raises:
        """
        Enables `ComplexNDArray /= Scalar`.
        """
        self._re /= other
        self._im /= other

    fn __itruediv__(mut self, other: Self) raises:
        """
        Enables `ComplexNDArray /= ComplexNDArray`.
        """
        self._re /= other._re
        self._im /= other._im

    fn __itruediv__(mut self, other: NDArray[dtype]) raises:
        """
        Enables `ComplexNDArray /= NDArray`.
        """
        self._re /= other
        self._im /= other

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#
    fn __str__(self) -> String:
        """
        Enables String(array).
        """
        var res: String
        try:
            res = self._array_to_string(0, 0, GLOBAL_PRINT_OPTIONS)
        except e:
            res = String("Cannot convert array to string") + String(e)

        return res

    fn write_to[W: Writer](self, mut writer: W):
        try:
            writer.write(
                self._array_to_string(0, 0, GLOBAL_PRINT_OPTIONS)
                + "\n"
                + String(self.ndim)
                + "D-array  Shape"
                + String(self.shape)
                + "  Strides"
                + String(self.strides)
                + "  DType: "
                + _concise_dtype_str(self.dtype)
                + "  C-cont: "
                + String(self.flags["C_CONTIGUOUS"])
                + "  F-cont: "
                + String(self.flags["F_CONTIGUOUS"])
                + "  own data: "
                + String(self.flags["OWNDATA"])
            )
        except e:
            writer.write("Cannot convert array to string" + String(e))

    fn __repr__(self) -> String:
        """
        Compute the "official" string representation of ComplexNDArray.
        An example is:
        ```
        fn main() raises:
            var A = ComplexNDArray[cf32](List[ComplexSIMD[cf32]](14,97,-59,-4,112,), shape=List[Int](5,))
            print(repr(A))
        ```
        It prints what can be used to construct the array itself:
        ```console
            ComplexNDArray[cf32](List[ComplexSIMD[cf32]](14,97,-59,-4,112,), shape=List[Int](5,))
        ```.
        """
        try:
            var result: String = String("ComplexNDArray[CDType.") + String(
                self.cdtype
            ) + String("](List[ComplexSIMD[CDType.c") + String(
                self._re.dtype
            ) + String(
                "]]("
            )
            if self._re.size > 6:
                for i in range(6):
                    result = result + String(self.item(i)) + String(",")
                result = result + " ... "
            else:
                for i in range(self._re.size):
                    result = result + String(self.item(i)) + String(",")
            result = result + String("), shape=List[Int](")
            for i in range(self._re.shape.ndim):
                result = result + String(self._re.shape._buf[i]) + ","
            result = result + String("))")
            return result
        except e:
            print("Cannot convert array to string", e)
            return ""

    fn _array_to_string(
        self,
        dimension: Int,
        offset: Int,
        print_options: PrintOptions,
    ) raises -> String:
        """
        Convert the array to a string.

        Args:
            dimension: The current dimension.
            offset: The offset of the current dimension.
            print_options: The print options.
        """
        var seperator = print_options.separator
        var padding = print_options.padding
        var edge_items = print_options.edge_items

        if self.ndim == 0:
            return String(self.item(0))
        if dimension == self.ndim - 1:
            var result: String = String("[") + padding
            var number_of_items = self.shape[dimension]
            if number_of_items <= edge_items:  # Print all items
                for i in range(number_of_items):
                    var value = self.load[width=1](
                        offset + i * self.strides[dimension]
                    )
                    var formatted_value = format_value(value, print_options)
                    result = result + formatted_value
                    if i < (number_of_items - 1):
                        result = result + seperator
                result = result + padding
            else:  # Print first 3 and last 3 items
                for i in range(edge_items):
                    var value = self.load[width=1](
                        offset + i * self.strides[dimension]
                    )
                    var formatted_value = format_value(value, print_options)
                    result = result + formatted_value
                    if i < (edge_items - 1):
                        result = result + seperator
                result = result + seperator + "..." + seperator
                for i in range(number_of_items - edge_items, number_of_items):
                    var value = self.load[width=1](
                        offset + i * self.strides[dimension]
                    )
                    var formatted_value = format_value(value, print_options)
                    result = result + formatted_value
                    if i < (number_of_items - 1):
                        result = result + seperator
                result = result + padding
            result = result + "]"
            return result
        else:
            var result: String = String("[")
            var number_of_items = self.shape[dimension]
            if number_of_items <= edge_items:  # Print all items
                for i in range(number_of_items):
                    if i == 0:
                        result = result + self._array_to_string(
                            dimension + 1,
                            offset + i * self.strides[dimension].__int__(),
                            print_options,
                        )
                    if i > 0:
                        result = (
                            result
                            + String(" ") * (dimension + 1)
                            + self._array_to_string(
                                dimension + 1,
                                offset + i * self.strides[dimension].__int__(),
                                print_options,
                            )
                        )
                    if i < (number_of_items - 1):
                        result = result + "\n"
            else:  # Print first 3 and last 3 items
                for i in range(edge_items):
                    if i == 0:
                        result = result + self._array_to_string(
                            dimension + 1,
                            offset + i * self.strides[dimension].__int__(),
                            print_options,
                        )
                    if i > 0:
                        result = (
                            result
                            + String(" ") * (dimension + 1)
                            + self._array_to_string(
                                dimension + 1,
                                offset + i * self.strides[dimension].__int__(),
                                print_options,
                            )
                        )
                    if i < (number_of_items - 1):
                        result += "\n"
                result = result + "...\n"
                for i in range(number_of_items - edge_items, number_of_items):
                    result = (
                        result
                        + String(" ") * (dimension + 1)
                        + self._array_to_string(
                            dimension + 1,
                            offset + i * self.strides[dimension].__int__(),
                            print_options,
                        )
                    )
                    if i < (number_of_items - 1):
                        result = result + "\n"
            result = result + "]"
            return result

    fn __len__(self) -> Int:
        return Int(self._re.size)

    fn load[
        width: Int = 1
    ](self, index: Int) raises -> ComplexSIMD[cdtype, dtype=dtype]:
        """
        Safely loads a SIMD element of size `width` at `index`
        from the underlying buffer.

        To bypass boundary checks, use `self._buf.ptr.load` directly.

        Raises:
            Index out of boundary.
        """

        if (index < 0) or (index >= self.size):
            raise Error(
                String("Invalid index: index out of bound [0, {}).").format(
                    self.size
                )
            )

        return ComplexSIMD[cdtype, dtype=dtype](
            re=self._re._buf.ptr.load[width=1](index),
            im=self._im._buf.ptr.load[width=1](index),
        )

    fn store[
        width: Int = 1
    ](mut self, index: Int, val: ComplexSIMD[cdtype, dtype=dtype]) raises:
        """
        Safely stores SIMD element of size `width` at `index`
        of the underlying buffer.

        To bypass boundary checks, use `self._buf.ptr.store` directly.

        Raises:
            Index out of boundary.
        """

        if (index < 0) or (index >= self.size):
            raise Error(
                String("Invalid index: index out of bound [0, {}).").format(
                    self.size
                )
            )

        self._re._buf.ptr.store(index, val.re)
        self._im._buf.ptr.store(index, val.im)

    fn load[
        width: Int = 1
    ](self, *indices: Int) raises -> ComplexSIMD[cdtype, dtype=dtype]:
        """
        Safely loads a SIMD element of size `width` at given variadic indices
        from the underlying buffer.

        To bypass boundary checks, use `self._buf.ptr.load` directly.

        Raises:
            Index out of boundary.
        """

        if len(indices) != self.ndim:
            raise (
                String(
                    "Length of indices {} does not match ndim {}".format(
                        len(indices), self.ndim
                    )
                )
            )

        for i in range(self.ndim):
            if (indices[i] < 0) or (indices[i] >= self.shape[i]):
                raise Error(
                    String(
                        "Invalid index at {}-th dim: "
                        "index out of bound [0, {})."
                    ).format(i, self.shape[i])
                )

        var idx: Int = _get_offset(indices, self.strides)
        return ComplexSIMD[cdtype, dtype=dtype](
            re=self._re._buf.ptr.load[width=1](idx),
            im=self._im._buf.ptr.load[width=1](idx),
        )

    fn store[
        width: Int = 1
    ](mut self, *indices: Int, val: ComplexSIMD[cdtype, dtype=dtype]) raises:
        """
        Safely stores SIMD element of size `width` at given variadic indices
        of the underlying buffer.

        To bypass boundary checks, use `self._buf.ptr.store` directly.

        Raises:
            Index out of boundary.
        """

        if len(indices) != self.ndim:
            raise (
                String(
                    "Length of indices {} does not match ndim {}".format(
                        len(indices), self.ndim
                    )
                )
            )

        for i in range(self.ndim):
            if (indices[i] < 0) or (indices[i] >= self.shape[i]):
                raise Error(
                    String(
                        "Invalid index at {}-th dim: "
                        "index out of bound [0, {})."
                    ).format(i, self.shape[i])
                )

        var idx: Int = _get_offset(indices, self.strides)
        self._re._buf.ptr.store(idx, val.re)
        self._im._buf.ptr.store(idx, val.im)

    # fn __iter__(self) raises -> _ComplexNDArrayIter[__origin_of(self._re), __origin_of(self._im), cdtype, dtype]:
    #     """Iterate over elements of the NDArray, returning copied value.

    #     Returns:
    #         An iterator of NDArray elements.

    #     Notes:
    #         Need to add lifetimes after the new release.
    #     """

    #     return _ComplexNDArrayIter[__origin_of(self._re), __origin_of(self._im), cdtype, dtype](
    #         array=self,
    #         length=self.shape[0],
    #     )

    # fn __reversed__(
    #     self,
    # ) raises -> _ComplexNDArrayIter[__origin_of(self._re), __origin_of(self._im), cdtype, dtype, forward=False]:
    #     """Iterate backwards over elements of the NDArray, returning
    #     copied value.

    #     Returns:
    #         A reversed iterator of NDArray elements.
    #     """

    # #     return _ComplexNDArrayIter[__origin_of(self._re), __origin_of(self._im), cdtype, dtype, forward=False](
    # #         array=self,
    # #         length=self.shape[0],
    # #     )

    fn item(self, owned index: Int) raises -> ComplexSIMD[cdtype, dtype=dtype]:
        """
        Return the scalar at the coordinates.

        If one index is given, get the i-th item of the ComplexNDArray (not buffer).
        It first scans over the first row, even it is a colume-major array.

        If more than one index is given, the length of the indices must match
        the number of dimensions of the array.

        Args:
            index: Index of item, counted in row-major way.

        Returns:
            A scalar matching the dtype of the array.

        Raises:
            Index is equal or larger than array size.
        """

        if index < 0:
            index += self.size

        if (index < 0) or (index >= self.size):
            raise Error(
                String("`index` exceeds array size ({})").format(self.size)
            )

        if self.flags["F_CONTIGUOUS"]:
            var c_stride = NDArrayStrides(shape=self.shape)
            var c_coordinates = List[Int]()
            var idx: Int = index
            for i in range(c_stride.ndim):
                var coordinate = idx // c_stride[i]
                idx = idx - c_stride[i] * coordinate
                c_coordinates.append(coordinate)

            # Get the value by coordinates and the strides
            return ComplexSIMD[cdtype, dtype=dtype](
                re=self._re._buf.ptr[_get_offset(c_coordinates, self.strides)],
                im=self._im._buf.ptr[_get_offset(c_coordinates, self.strides)],
            )

        else:
            return ComplexSIMD[cdtype, dtype=dtype](
                re=self._re._buf.ptr[index], im=self._im._buf.ptr[index]
            )

    fn item(self, *index: Int) raises -> ComplexSIMD[cdtype, dtype=dtype]:
        """
        Return the scalar at the coordinates.

        If one index is given, get the i-th item of the ComplexNDArray (not buffer).
        It first scans over the first row, even it is a colume-major array.

        If more than one index is given, the length of the indices must match
        the number of dimensions of the array.

        Args:
            index: The coordinates of the item.

        Returns:
            A scalar matching the dtype of the array.

        Raises:
            Index is equal or larger than size of dimension.
        """

        if len(index) != self.ndim:
            raise Error(
                String("Number of indices ({}) do not match ndim ({})").format(
                    len(index), self.ndim
                )
            )
        var list_index = List[Int]()
        for i in range(len(index)):
            if index[i] < 0:
                list_index.append(index[i] + self.shape[i])
            else:
                list_index.append(index[i])
            if (list_index[i] < 0) or (list_index[i] >= self.shape[i]):
                raise Error(
                    String("{}-th index exceeds shape size {}").format(
                        i, self.shape[i]
                    )
                )
        return ComplexSIMD[cdtype, dtype=dtype](
            re=self._re._buf.ptr[_get_offset(index, self.strides)],
            im=self._im._buf.ptr[_get_offset(index, self.strides)],
        )

    fn itemset(
        mut self,
        index: Variant[Int, List[Int]],
        item: ComplexSIMD[cdtype, dtype=dtype],
    ) raises:
        """Set the scalar at the coordinates.

        Args:
            index: The coordinates of the item.
                Can either be `Int` or `List[Int]`.
                If `Int` is passed, it is the index of i-th item of the whole array.
                If `List[Int]` is passed, it is the coordinate of the item.
            item: The scalar to be set.
        """

        # If one index is given
        if index.isa[Int]():
            var idx = index._get_ptr[Int]()[]
            if idx < self.size:
                if self.flags[
                    "F_CONTIGUOUS"
                ]:  # column-major should be converted to row-major
                    # The following code can be taken out as a function that
                    # convert any index to coordinates according to the order
                    var c_stride = NDArrayStrides(shape=self.shape)
                    var c_coordinates = List[Int]()
                    for i in range(c_stride.ndim):
                        var coordinate = idx // c_stride[i]
                        idx = idx - c_stride[i] * coordinate
                        c_coordinates.append(coordinate)
                    self._re._buf.ptr.store(
                        _get_offset(c_coordinates, self.strides), item.re
                    )
                    self._im._buf.ptr.store(
                        _get_offset(c_coordinates, self.strides), item.im
                    )
                else:
                    self._re._buf.ptr.store(idx, item.re)
                    self._im._buf.ptr.store(idx, item.im)
            else:
                raise Error(
                    String(
                        "Error: Elements of `index` ({}) \n"
                        "exceed the array size ({})."
                    ).format(idx, self.size)
                )

        else:
            var indices = index._get_ptr[List[Int]]()[]
            if indices.__len__() != self.ndim:
                raise Error("Error: Length of Indices do not match the shape")
            for i in range(indices.__len__()):
                if indices[i] >= self.shape[i]:
                    raise Error(
                        "Error: Elements of `index` exceed the array shape"
                    )
            self._re._buf.ptr.store(_get_offset(indices, self.strides), item.re)
            self._im._buf.ptr.store(_get_offset(indices, self.strides), item.im)

    fn conj(self) raises -> Self:
        """
        Return the complex conjugate of the ComplexNDArray.
        """
        return Self(self._re, -self._im)

    fn to_ndarray(self, type: String = "re") raises -> NDArray[dtype=dtype]:
        if type == "re":
            var result: NDArray[dtype=dtype] = NDArray[dtype=dtype](self.shape)
            memcpy(result._buf.ptr, self._re._buf.ptr, self.size)
            return result^
        elif type == "im":
            var result: NDArray[dtype=dtype] = NDArray[dtype=dtype](self.shape)
            memcpy(result._buf.ptr, self._im._buf.ptr, self.size)
            return result^
        else:
            raise Error("Invalid type: " + type + ", must be 're' or 'im'")


# @value
# struct _ComplexNDArrayIter[
#     is_mutable: Bool, //,
#     origin: Origin[is_mutable],
#     cdtype: CDType,
#     dtype: DType,
#     forward: Bool = True,
# ]:
#     """
#     Iterator for NDArray.

#     Parameters:
#         is_mutable: Whether the iterator is mutable.
#         origin: The lifetime of the underlying NDArray data.
#         cdtype: The complex data type of the item.
#         dtype: The data type of the item.
#         forward: The iteration direction. `False` is backwards.
#     """

#     var index: Int
#     var array: ComplexNDArray[cdtype, dtype=dtype]
#     var length: Int

#     fn __init__(
#         mut self,
#         array: ComplexNDArray[cdtype, dtype=dtype],
#         length: Int,
#     ):
#         self.index = 0 if forward else length
#         self.length = length
#         self.array = array

#     fn __iter__(self) -> Self:
#         return self

#     fn __next__(mut self) raises -> ComplexNDArray[cdtype, dtype=dtype]:
#         @parameter
#         if forward:
#             var current_index = self.index
#             self.index += 1
#             return self.array.__getitem__(current_index)
#         else:
#             var current_index = self.index
#             self.index -= 1
#             return self.array.__getitem__(current_index)

#     @always_inline
#     fn __has_next__(self) -> Bool:
#         @parameter
#         if forward:
#             return self.index < self.length
#         else:
#             return self.index > 0

#     fn __len__(self) -> Int:
#         @parameter
#         if forward:
#             return self.length - self.index
#         else:
#             return self.index
