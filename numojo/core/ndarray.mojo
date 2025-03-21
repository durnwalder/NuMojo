# ===----------------------------------------------------------------------=== #
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
Implements basic object methods for working with N-Dimensional Array.
"""
# ===----------------------------------------------------------------------===#
# SECTIONS OF THE FILE:
#
# `NDArray` type
# 1. Life cycle methods.
# 2. Indexing and slicing (get and set dunders and relevant methods).
# 3. Operator dunders.
# 4. IO, trait, and iterator dunders.
# 5. Other methods (Sorted alphabetically).
#
# Iterators of `NDArray`:
# 1. `_NDArrayIter` type
# 2. `_NDAxisIter` type
# 3. `_NDIter` type
#
# ===----------------------------------------------------------------------===#
# FORMAT FOR DOCSTRING (See "Mojo docstring style guide" for more information)
# 1. Description *
# 2. Parameters *
# 3. Args *
# 4. Constraints *
# 4) Returns *
# 5) Raises *
# 6) SEE ALSO
# 7) NOTES
# 8) REFERENCES
# 9) Examples *
# (Items marked with * are flavored in "Mojo docstring style guide")
#
# ===----------------------------------------------------------------------===#
# TODO: Consider whether we should add vectorization for _get_offset.
# TODO: Create NDArrayView that points to the buffer of the raw array.
#       This requires enhancement of functionalities of traits from Mojo's side.
#       The data buffer can implement an ArrayData trait (RawData or RefData)
#       RawData type is just a wrapper of `UnsafePointer`.
#       RefData type has an extra property `indices`: getitem(i) -> A[I[i]].
# TODO: Rename some variables or methods that should not be exposed to users.
# TODO: Remove some methods, `mdot()`, `rdot()`, `row()`, `col()`, etc,
#       that does not belong to the NDArray type.
# TODO: Special checks for 0d array (numojo scalar).
# ===----------------------------------------------------------------------===#

from algorithm import parallelize, vectorize
import builtin.math as builtin_math
import builtin.bool as builtin_bool
from builtin.type_aliases import Origin
from collections.optional import Optional
from memory import UnsafePointer, memset_zero, memcpy
from math import log10
from python import PythonObject
from sys import simdwidthof
from tensor import Tensor
from utils import Variant

import numojo.core._array_funcs as _af
from numojo.core._math_funcs import Vectorized
from numojo.core.datatypes import TypeCoercion, _concise_dtype_str
from numojo.core.flags import Flags
from numojo.core.item import Item
from numojo.core.ndshape import NDArrayShape
from numojo.core.ndstrides import NDArrayStrides
from numojo.core.own_data import OwnData
from numojo.core.utility import (
    _get_offset,
    _transfer_offset,
    _traverse_iterative,
    _traverse_iterative_setter,
    to_numpy,
    to_tensor,
    bool_to_numeric,
)
import numojo.routines.bitwise as bitwise
import numojo.routines.creation as creation
from numojo.routines.io.formatting import (
    format_value,
    PrintOptions,
    GLOBAL_PRINT_OPTIONS,
)
import numojo.routines.logic.comparison as comparison
import numojo.routines.math.arithmetic as arithmetic
import numojo.routines.math.rounding as rounding


struct NDArray[dtype: DType = DType.float64](
    Stringable, Representable, CollectionElement, Sized, Writable
):
    # TODO: NDArray[dtype: DType = DType.float64,
    #               Buffer: Bufferable[dtype] = OwnData[dtype]]
    """The N-dimensional array (NDArray).

    Parameters:
        dtype: Type of item in NDArray. Default type is DType.float64.

    The array can be uniquely defined by the following:
        1. The data buffer of all items.
        2. The shape of the array.
        3. The strides (Length of item to travel to next dimension).
        4. The datatype of the elements.

    The following attributes are also helpful:
        - The number of dimensions
        - Size of the array (number of items)
        - The order of the array: Row vs Columns major
    """

    alias width: Int = simdwidthof[dtype]()
    """Vector size of the data type."""

    var _buf: OwnData[dtype]
    """Data buffer of the items in the NDArray."""
    var ndim: Int
    """Number of Dimensions."""
    var shape: NDArrayShape
    """Size and shape of NDArray."""
    var size: Int
    """Size of NDArray."""
    var strides: NDArrayStrides
    """Contains offset, strides."""
    var flags: Flags
    """Information about the memory layout of the array."""

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    # default constructor

    @always_inline("nodebug")
    fn __init__(
        out self,
        shape: NDArrayShape,
        order: String = "C",
    ) raises:
        """
        Initializes an NDArray with given shape.
        The memory is not filled with values.

        Args:
            shape: Variadic shape.
            order: Memory order C or F.
        """

        self.ndim = shape.ndim
        self.shape = NDArrayShape(shape)
        self.size = self.shape.size_of_array()
        self.strides = NDArrayStrides(shape, order=order)
        self._buf = OwnData[dtype](self.size)
        self.flags = Flags(
            self.shape, self.strides, owndata=True, writeable=True
        )

    @always_inline("nodebug")
    fn __init__(
        out self,
        shape: List[Int],
        order: String = "C",
    ) raises:
        """
        (Overload) Initializes an NDArray with given shape (list of integers).

        Args:
            shape: List of shape.
            order: Memory order C or F.
        """

        self = Self(Shape(shape), order)

    @always_inline("nodebug")
    fn __init__(
        out self,
        shape: VariadicList[Int],
        order: String = "C",
    ) raises:
        """
        (Overload) Initializes an NDArray with given shape (variadic list of integers).

        Args:
            shape: Variadic List of shape.
            order: Memory order C or F.
        """

        self = Self(Shape(shape), order)

    fn __init__(
        mut self,
        shape: List[Int],
        offset: Int,
        strides: List[Int],
    ) raises:
        """
        Extremely specific NDArray initializer.

        Args:
            shape: List of shape.
            offset: Offset value.
            strides: List of strides.
        """
        self.shape = NDArrayShape(shape)
        self.ndim = self.shape.ndim
        self.size = self.shape.size_of_array()
        self.strides = NDArrayStrides(strides=strides)
        self._buf = OwnData[dtype](self.size)
        memset_zero(self._buf.ptr, self.size)
        self.flags = Flags(
            self.shape, self.strides, owndata=True, writeable=True
        )

    fn __init__(
        out self,
        shape: NDArrayShape,
        strides: NDArrayStrides,
        ndim: Int,
        size: Int,
        flags: Flags,
    ):
        """
        Constructs an extremely specific array, with value uninitialized.
        The properties do not need to be compatible and are not checked.
        For example, it can construct a 0-D array (numojo scalar).

        Args:
            shape: Shape of array.
            strides: Strides of array.
            ndim: Number of dimensions.
            size: Size of array.
            flags: Flags of array.
        """

        self.shape = shape
        self.strides = strides
        self.ndim = ndim
        self.size = size
        self.flags = flags
        self._buf = OwnData[dtype](self.size)

    # for creating views (unsafe!)
    fn __init__(
        out self,
        shape: NDArrayShape,
        ref buffer: UnsafePointer[Scalar[dtype]],
        offset: Int,
        strides: NDArrayStrides,
    ) raises:
        """
        Initialize an NDArray view with given shape, buffer, offset, and strides.
        ***Unsafe!*** This function is currently unsafe. Only for internal use.

        Args:
            shape: Shape of the array.
            buffer: Unsafe pointer to the buffer.
            offset: Offset value.
            strides: Strides of the array.
        """
        self.shape = shape
        self.strides = strides
        self.ndim = self.shape.ndim
        self.size = self.shape.size_of_array()
        self._buf = OwnData(ptr=buffer.offset(offset))
        self.flags = Flags(
            self.shape, self.strides, owndata=False, writeable=False
        )

    @always_inline("nodebug")
    fn __copyinit__(mut self, other: Self):
        """
        Copy other into self.
        It is a deep copy. So the new array owns the data.

        Args:
            other: The NDArray to copy from.
        """
        self.ndim = other.ndim
        self.shape = other.shape
        self.size = other.size
        self.strides = other.strides
        self._buf = OwnData[dtype](self.size)
        memcpy(self._buf.ptr, other._buf.ptr, other.size)
        self.flags = Flags(
            c_contiguous=other.flags.C_CONTIGUOUS,
            f_contiguous=other.flags.F_CONTIGUOUS,
            owndata=True,
            writeable=True,
        )

    @always_inline("nodebug")
    fn __moveinit__(mut self, owned existing: Self):
        """
        Move other into self.

        Args:
            existing: The NDArray to move from.
        """
        self.ndim = existing.ndim
        self.shape = existing.shape
        self.size = existing.size
        self.strides = existing.strides
        self.flags = existing.flags^
        self._buf = existing._buf^

    @always_inline("nodebug")
    fn __del__(owned self):
        """
        Destroys all elements in the list and free its memory.
        """
        if self.flags.OWNDATA:
            self._buf.ptr.free()

    # ===-------------------------------------------------------------------===#
    # Indexing and slicing
    # Getter and setter dunders and other methods
    # ===-------------------------------------------------------------------===#

    # ===-------------------------------------------------------------------===#
    # Getter dunders and other getter methods
    #
    # 1. Basic Indexing Operations
    # fn _getitem(self, *indices: Int) -> Scalar[dtype]                         # Direct unsafe getter
    # fn __getitem__(self) raises -> SIMD[dtype, 1]                             # Get 0d array value
    # fn __getitem__(self, index: Item) raises -> SIMD[dtype, 1]                # Get by coordinate list
    #
    # 2. Single Index Slicing
    # fn __getitem__(self, idx: Int) raises -> Self                             # Get by single index
    #
    # 3. Multi-dimensional Slicing
    # fn __getitem__(self, *slices: Slice) raises -> Self                       # Get by variable slices
    # fn __getitem__(self, slice_list: List[Slice]) raises -> Self              # Get by list of slices
    # fn __getitem__(self, *slices: Variant[Slice, Int]) raises -> Self         # Get by mix of slices/ints
    #
    # 4. Advanced Indexing
    # fn __getitem__(self, indices: NDArray[DType.index]) raises -> Self        # Get by index array
    # fn __getitem__(self, indices: List[Int]) raises -> Self                   # Get by list of indices
    # fn __getitem__(self, mask: NDArray[DType.bool]) raises -> Self            # Get by boolean mask
    # fn __getitem__(self, mask: List[Bool]) raises -> Self                     # Get by boolean list
    #
    # 5. Low-level Access
    # fn item(self, owned index: Int) raises -> Scalar[dtype]                   # Get item by linear index
    # fn item(self, *index: Int) raises -> Scalar[dtype]                        # Get item by coordinates
    # fn load(self, owned index: Int) raises -> Scalar[dtype]                   # Load with bounds check
    # fn load[width: Int](self, index: Int) raises -> SIMD[dtype, width]        # Load SIMD value
    # fn load[width: Int](self, *indices: Int) raises -> SIMD[dtype, width]     # Load SIMD at coordinates
    # ===-------------------------------------------------------------------===#

    fn _getitem(self, *indices: Int) -> Scalar[dtype]:
        """
        Get item at indices and bypass all boundary checks.
        ***UNSAFE!*** No boundary checks made, for internal use only.

        Args:
            indices: Indices to get the value.

        Returns:
            The element of the array at the indices.

        Notes:
            This function is unsafe and should be used only on internal use.

        Examples:

        ```mojo
        import numojo
        var A = numojo.ones(numojo.Shape(2,3,4))
        print(A._getitem(1,2,3))
        ```
        """
        var index_of_buffer: Int = 0
        for i in range(self.ndim):
            index_of_buffer += indices[i] * self.strides._buf[i]
        return self._buf.ptr[index_of_buffer]

    fn __getitem__(self) raises -> SIMD[dtype, 1]:
        """
        Gets the value of the 0-D array.

        Returns:
            The value of the 0-D array.

        Raises:
            Error: If the array is not 0-d.

        Examples:

        ```console
        >>>import numojo
        >>>var a = numojo.arange(3)[0]
        >>>print(a[]) # gets values of the 0-D array.
        ```.
        """
        if self.ndim != 0:
            raise Error(
                "\nError in `numojo.NDArray.__getitem__()`: "
                "Cannot get value without index."
            )
        return self._buf.ptr[]

    fn __getitem__(self, index: Item) raises -> SIMD[dtype, 1]:
        """
        Get the value at the index list.

        Args:
            index: Index list.

        Returns:
            The value at the index list.

        Raises:
            Error: If the length of `index` does not match the number of dimensions.
            Error: If any of the index elements exceeds the size of the dimension of the array.

        Examples:

        ```console
        >>>import numojo
        >>>var a = numojo.arange(0, 10, 1).reshape(numojo.Shape(2, 5))
        >>>print(a[Item(1, 2)]) # gets values of the element at (1, 2).
        ```.
        """
        if index.__len__() != self.ndim:
            raise Error(
                String(
                    "\nError in `numojo.NDArray.__getitem__(index: Item)`: "
                    "Length of index ({}) does not match the number of"
                    "dimensions ({})."
                ).format(index.__len__(), self.ndim)
            )

        for i in range(index.__len__()):
            if index[i] >= self.shape[i]:
                raise Error(
                    String(
                        "\nError in `numojo.NDArray.__getitem__(index: Item)`:"
                        " Index out of bounds for dimension {} with index {} "
                        " and dimension size {}."
                    ).format(i, index[i], self.shape[i])
                )

        var idx: Int = _get_offset(index, self.strides)
        return self._buf.ptr.load[width=1](idx)

    fn __getitem__(self, idx: Int) raises -> Self:
        """
        Retrieve a slice of the array corresponding to the index at the first dimension.

        Args:
            idx: Index to get the slice.

        Returns:
            A slice of the array.

        Raises:
            Error: If the array is 0-d.

        Examples:

        ```console
        >>>import numojo
        >>>var a = numojo.arange(0, 10, 1).reshape(numojo.Shape(2, 5))
        >>>print(a[1]) # returns the second row of the array.
        ```.
        """

        var slice_list = List[Slice]()
        slice_list.append(Slice(idx, idx + 1, 1))

        # If the ndim is 0, then it is a numojo scalar (0-D array).
        if self.ndim == 0:
            raise Error(
                "\nError in `numojo.NDArray.__getitem__(self, idx: Int)`: "
                "Cannot slice a 0-d array."
            )

        var narr: Self

        # If the ndim is 1
        if self.ndim == 1:
            narr = creation._0darray[dtype](self._buf.ptr[idx])

        else:
            for i in range(1, self.ndim):
                var size_at_dim: Int = self.shape[i]
                slice_list.append(Slice(0, size_at_dim, 1))

            narr = self.__getitem__(slice_list)

        return narr

    fn __getitem__(self, owned *slices: Slice) raises -> Self:
        """
        Retrieve slices of an array from variadic slices.

        Args:
            slices: Variadic slices.

        Returns:
            A slice of the array.

        Examples:

        ```console
        >>>import numojo
        >>>var a = numojo.arange(10).reshape(numojo.shape(2, 5))
        >>>var b = a[:, 2:4]
        >>>print(b) # `arr[:, 2:4]` returns the corresponding sliced array (2 x 2).
        ```.
        """

        var n_slices: Int = slices.__len__()
        var slice_list: List[Slice] = List[Slice]()
        for i in range(len(slices)):
            slice_list.append(slices[i])

        if n_slices < self.ndim:
            for i in range(n_slices, self.ndim):
                slice_list.append(Slice(0, self.shape[i], 1))

        var narr: Self = self[slice_list]
        return narr

    fn __getitem__(self, owned slice_list: List[Slice]) raises -> Self:
        """
        Retrieve slices of an array from a list of slices.

        Args:
            slice_list: List of slices.

        Returns:
            A slice of the array.

        Raises:
            Error: If the slice list is empty.

        Examples:

        ```console
        >>>import numojo
        >>>var a = numojo.arange(10).reshape(numojo.shape(2, 5))
        >>>var b = a[List[Slice](Slice(0, 2, 1), Slice(2, 4, 1))] # `arr[:, 2:4]` returns the corresponding sliced array (2 x 2).
        >>>print(b)
        ```.
        """
        # Check error cases
        if slice_list.__len__() == 0:
            raise Error(
                "\nError in `numojo.NDArray.__getitem__(slice_list:"
                " List[Slice])`:\nEmpty slice list provided!"
            )

        if slice_list.__len__() < self.ndim:
            for i in range(slice_list.__len__(), self.ndim):
                slice_list.append(Slice(0, self.shape[i], 1))

        # Adjust slice
        var slices = self._adjust_slice(slice_list)
        var spec = List[Int]()
        var ndims = 0

        # Calculate output shape and validate slices in one pass
        for i in range(self.ndim):
            var start: Int = slices[i].start.value()
            var end: Int = slices[i].end.value()
            var step: Int = slices[i].step.or_else(1)

            var slice_len: Int = len(range(start, end, step))
            spec.append(slice_len)
            if slice_len != 1:
                ndims += 1

        ndims = 1 if ndims == 0 else ndims

        # Calculate new slices array shape, coefficients, and offset
        var nshape = List[Int]()
        var ncoefficients = List[Int]()
        var noffset = 0
        var nnum_elements = 1

        for i in range(self.ndim):
            if spec[i] != 1:
                nshape.append(spec[i])
                nnum_elements *= spec[i]
                ncoefficients.append(self.strides[i] * slices[i].step.value())
            noffset += slices[i].start.value() * self.strides[i]

        if nshape.__len__() == 0:
            nshape.append(1)
            nnum_elements = 1
            ncoefficients.append(1)

        # Calculate strides based on memory layout: only C & F order are supported
        var nstrides = List[Int]()
        if self.flags.C_CONTIGUOUS:
            var temp_stride = 1
            for i in range(nshape.__len__() - 1, -1, -1):
                nstrides.insert(0, temp_stride)
                temp_stride *= nshape[i]
        else:  # F_CONTIGUOUS
            var temp_stride = 1
            for i in range(nshape.__len__()):
                nstrides.append(temp_stride)
                temp_stride *= nshape[i]

        # Create and iteratively set values in the new array
        var narr = Self(offset=noffset, shape=nshape, strides=nstrides)
        var index = List[Int]()
        for _ in range(ndims):
            index.append(0)

        _traverse_iterative[dtype](
            self, narr, nshape, ncoefficients, nstrides, noffset, index, 0
        )

        return narr

    fn __getitem__(self, owned *slices: Variant[Slice, Int]) raises -> Self:
        """
        Get items by a series of either slices or integers.

        Args:
            slices: A series of either Slice or Int.

        Returns:
            A slice of the ndarray with a smaller or equal dimension of the original one.

        Raises:
            Error: If the number of slices is greater than the number of dimensions of the array.

        Notes:
            A decrease of dimensions may or may not happen when `__getitem__` is
            called on an ndarray. An ndarray of X-D array can become Y-D array after
            `__getitem__` where `Y <= X`.

            Whether the dimension decreases or not depends on:
            1. What types of arguments are passed into `__getitem__`.
            2. The number of arguments that are passed in `__getitem__`.

            PRINCIPAL: The number of dimensions to be decreased is determined by
            the number of `Int` passed in `__getitem__`.

            For example, `A` is a 10x10x10 ndarray (3-D). Then,

            - `A[1, 2, 3]` leads to a 0-D array (scalar), since there are 3 integers.
            - `A[1, 2]` leads to a 1-D array (vector), since there are 2 integers,
            so the dimension decreases by 2.
            - `A[1]` leads to a 2-D array (matrix), since there is 1 integer, so the
            dimension decreases by 1.

            The number of dimensions will not decrease when Slice is passed in
            `__getitem__` or no argument is passed in for a certain dimension
            (it is an implicit slide and a slide of all items will be used).

            Take the same example `A` with 10x10x10 in shape. Then,

            - `A[1:4, 2:5, 3:6]`, leads to a 3-D array (no decrease in dimension),
            since there are 3 slices.
            - `A[2:8]`, leads to a 3-D array (no decrease in dimension), since
            there are 1 explicit slice and 2 implicit slices.

            When there is a mixture of int and slices passed into `__getitem__`,
            the number of integers will be the number of dimensions to be decreased.
            Example,

            - `A[1:4, 2, 2]`, leads to a 1-D array (vector), since there are 2
            integers, so the dimension decreases by 2.

            Note that, even though a slice contains one row, it does not reduce
            the dimensions. Example,

            - `A[1:2, 2:3, 3:4]`, leads to a 3-D array (no decrease in
            dimension), since there are 3 slices.

            Note that, when the number of integers equals to the number of
            dimensions, the final outcome is an 0-D array instead of a number.
            The user has to upack the 0-D array with the method`A.item(0)` to
            get the corresponding number.
            This behavior is different from numpy where the latter returns a
            number.

            More examples for 1-D, 2-D, and 3-D arrays.

        Examples:

        ```console
        A is a matrix
        [[      -128    -95     65      -11     ]
         [      8       -72     -116    45      ]
         [      45      111     -30     4       ]
         [      84      -120    -115    7       ]]
        2-D array  Shape: [4, 4]  DType: int8

        A[0]
        [       -128    -95     65      -11     ]
        1-D array  Shape: [4]  DType: int8

        A[0, 1]
        -95
        0-D array  Shape: [0]  DType: int8

        A[Slice(1,3)]
        [[      8       -72     -116    45      ]
         [      45      111     -30     4       ]]
        2-D array  Shape: [2, 4]  DType: int8

        A[1, Slice(2,4)]
        [       -116    45      ]
        1-D array  Shape: [2]  DType: int8

        A[Slice(1,3), Slice(1,3)]
        [[      -72     -116    ]
         [      111     -30     ]]
        2-D array  Shape: [2, 2]  DType: int8

        A.item(0,1) as Scalar
        -95

        ==============================
        A is a vector
        [       43      -127    -30     -111    ]
        1-D array  Shape: [4]  DType: int8

        A[0]
        43
        0-D array  Shape: [0]  DType: int8

        A[Slice(1,3)]
        [       -127    -30     ]
        1-D array  Shape: [2]  DType: int8

        A.item(0) as Scalar
        43

        ==============================
        A is a 3darray
        [[[     -22     47      22      110     ]
          [     88      6       -105    39      ]
          [     -22     51      105     67      ]
          [     -61     -116    60      -44     ]]
         [[     33      65      125     -35     ]
          [     -65     123     57      64      ]
          [     38      -110    33      98      ]
          [     -59     -17     68      -6      ]]
         [[     -68     -58     -37     -86     ]
          [     -4      101     104     -113    ]
          [     103     1       4       -47     ]
          [     124     -2      -60     -105    ]]
        [[     114     -110    0       -30     ]
          [     -58     105     7       -10     ]
          [     112     -116    66      69      ]
          [     83      -96     -124    48      ]]]
        3-D array  Shape: [4, 4, 4]  DType: int8

        A[0]
        [[      -22     47      22      110     ]
         [      88      6       -105    39      ]
         [      -22     51      105     67      ]
         [      -61     -116    60      -44     ]]
        2-D array  Shape: [4, 4]  DType: int8

        A[0, 1]
        [       88      6       -105    39      ]
        1-D array  Shape: [4]  DType: int8

        A[0, 1, 2]
        -105
        0-D array  Shape: [0]  DType: int8

        A[Slice(1,3)]
        [[[     33      65      125     -35     ]
          [     -65     123     57      64      ]
          [     38      -110    33      98      ]
          [     -59     -17     68      -6      ]]
         [[     -68     -58     -37     -86     ]
          [     -4      101     104     -113    ]
          [     103     1       4       -47     ]
          [     124     -2      -60     -105    ]]]
        3-D array  Shape: [2, 4, 4]  DType: int8

        A[1, Slice(2,4)]
        [[      38      -110    33      98      ]
         [      -59     -17     68      -6      ]]
        2-D array  Shape: [2, 4]  DType: int8

        A[Slice(1,3), Slice(1,3), 2]
        [[      57      33      ]
         [      104     4       ]]
        2-D array  Shape: [2, 2]  DType: int8

        A.item(0,1,2) as Scalar
        -105
        ```.
        """
        var n_slices: Int = slices.__len__()
        if n_slices > self.ndim:
            raise Error(
                String(
                    "\nError in `numojo.NDArray.__getitem__(slices:"
                    " Variant[Slice, Int])`:\nNumber of slices {} is greater"
                    " than number of dimension of array {}!"
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
                slice_list.append(Slice(int, int + 1, 1))

        if n_slices < self.ndim:
            for i in range(n_slices, self.ndim):
                var size_at_dim: Int = self.shape[i]
                slice_list.append(Slice(0, size_at_dim, 1))

        var narr: Self = self.__getitem__(slice_list)

        # Number of ints equals to nidm, it returns a 0-D array.
        if count_int == self.ndim:
            narr = creation._0darray[dtype](narr._buf.ptr[])

        return narr

    fn __getitem__(self, indices: NDArray[DType.index]) raises -> Self:
        """
        Get items from 0-th dimension of an ndarray of indices.
        If the original array is of shape (i,j,k) and
        the indices array is of shape (l, m, n), then the output array
        will be of shape (l,m,n,j,k).

        Args:
            indices: Array of indices.

        Returns:
            NDArray with items from the array of indices.

        Raises:
            Error: If the elements of indices are greater than size of the corresponding dimension of the array.

        Examples:

        ```console
        >>>var a = nm.arange[i8](6)
        >>>print(a)
        [       0       1       2       3       4       5       ]
        1-D array  Shape: [6]  DType: int8  C-cont: True  F-cont: True  own data: True
        >>>print(a[nm.array[isize]("[4, 2, 5, 1, 0, 2]")])
        [       4       2       5       1       0       2       ]
        1-D array  Shape: [6]  DType: int8  C-cont: True  F-cont: True  own data: True

        var b = nm.arange[i8](12).reshape(Shape(2, 2, 3))
        print(b)
        [[[     0       1       2       ]
          [     3       4       5       ]]
         [[     6       7       8       ]
          [     9       10      11      ]]]
        3-D array  Shape: [2, 2, 3]  DType: int8  C-cont: True  F-cont: False  own data: True
        print(b[nm.array[isize]("[2, 0, 1]")])
        [[[     0       0       0       ]
          [     0       67      95      ]]
         [[     0       1       2       ]
          [     3       4       5       ]]
         [[     6       7       8       ]
          [     9       10      11      ]]]
        3-D array  Shape: [3, 2, 3]  DType: int8  C-cont: True  F-cont: False  own data: True
        ```.
        """

        # Get the shape of resulted array
        var shape = indices.shape.join(self.shape._pop(0))

        var result = NDArray[dtype](shape)
        var size_per_item = self.size // self.shape[0]

        # Fill in the values
        for i in range(indices.size):
            if indices.item(i) >= self.shape[0]:
                raise Error(
                    String(
                        "\nError in `numojo.NDArray.__getitem__(indices:"
                        " NDArray[DType.index])`:\nindex {} with value {} is"
                        " out of boundary [0, {})"
                    ).format(i, indices.item(i), self.shape[0])
                )
            memcpy(
                result._buf.ptr + i * size_per_item,
                self._buf.ptr + indices.item(i) * size_per_item,
                size_per_item,
            )

        return result

    fn __getitem__(self, indices: List[Int]) raises -> Self:
        # TODO: Use trait IntLike when it is supported by Mojo.
        """
        Get items from 0-th dimension of an array. It is an overload of
        `__getitem__(self, indices: NDArray[DType.index]) raises -> Self`.

        Args:
            indices: A list of Int.

        Returns:
            NDArray with items from the list of indices.

        Raises:
            Error: If the elements of indices are greater than size of the corresponding dimension of the array.

        Examples:

        ```console
        >>>var a = nm.arange[i8](6)
        >>>print(a)
        [       0       1       2       3       4       5       ]
        1-D array  Shape: [6]  DType: int8  C-cont: True  F-cont: True  own data: True
        >>>print(a[List[Int](4, 2, 5, 1, 0, 2)])
        [       4       2       5       1       0       2       ]
        1-D array  Shape: [6]  DType: int8  C-cont: True  F-cont: True  own data: True

        var b = nm.arange[i8](12).reshape(Shape(2, 2, 3))
        print(b)
        [[[     0       1       2       ]
        [     3       4       5       ]]
        [[     6       7       8       ]
        [     9       10      11      ]]]
        3-D array  Shape: [2, 2, 3]  DType: int8  C-cont: True  F-cont: False  own data: True
        print(b[List[Int](2, 0, 1)])
        [[[     0       0       0       ]
        [     0       67      95      ]]
        [[     0       1       2       ]
        [     3       4       5       ]]
        [[     6       7       8       ]
        [     9       10      11      ]]]
        3-D array  Shape: [3, 2, 3]  DType: int8  C-cont: True  F-cont: False  own data: True
        ```.
        """

        var indices_array = NDArray[DType.index](shape=Shape(len(indices)))
        for i in range(len(indices)):
            (indices_array._buf.ptr + i).init_pointee_copy(indices[i])

        return self[indices_array]

    fn __getitem__(self, mask: NDArray[DType.bool]) raises -> Self:
        # TODO: Extend the mask into multiple dimensions.
        """
        Get item from an array according to a mask array.
        If array shape is equal to mask shape, it returns a flattened array of
        the values where mask is True.
        If array shape is not equal to mask shape, it returns items from the
        0-th dimension of the array where mask is True.

        Args:
            mask: NDArray with Dtype.bool.

        Returns:
            NDArray with items from the mask.

        Raises:
            Error: If the mask is not a 1-D array (Currently we only support 1-d mask array).

        Examples:

        ```console
        >>>var a = nm.arange[i8](6)
        >>>print(a)
        [       0       1       2       3       4       5       ]
        1-D array  Shape: [6]  DType: int8  C-cont: True  F-cont: True  own data: True
        >>>print(a[nm.array[boolean]("[1,0,1,1,0,1]")])
        [       0       2       3       5       ]
        1-D array  Shape: [4]  DType: int8  C-cont: True  F-cont: True  own data: True

        var b = nm.arange[i8](12).reshape(Shape(2, 2, 3))
        print(b)
        [[[     0       1       2       ]
        [     3       4       5       ]]
        [[     6       7       8       ]
        [     9       10      11      ]]]
        3-D array  Shape: [2, 2, 3]  DType: int8  C-cont: True  F-cont: False  own data: True
        >>>print(b[nm.array[boolean]("[0,1]")])
        [[[     6       7       8       ]
        [     9       10      11      ]]]
        3-D array  Shape: [1, 2, 3]  DType: int8  C-cont: True  F-cont: True  own data: True
        ```.
        """

        # CASE 1:
        # if array shape is equal to mask shape,
        # return a flattened array of the values where mask is True
        if mask.shape == self.shape:
            var len_of_result = 0

            # Count number of True
            for i in range(mask.size):
                if mask.item(i):
                    len_of_result += 1

            # Change the first number of the ndshape
            var result = NDArray[dtype](shape=NDArrayShape(len_of_result))

            # Fill in the values
            var offset = 0
            for i in range(mask.size):
                if mask.item(i):
                    (result._buf.ptr + offset).init_pointee_copy(
                        self._buf.ptr[i]
                    )
                    offset += 1

            return result

        # CASE 2:
        # if array shape is not equal to mask shape,
        # return items from the 0-th dimension of the array where mask is True
        if mask.ndim > 1:
            raise Error(
                String(
                    "\nError in `numojo.NDArray.__getitem__(mask:"
                    " NDArray[DType.bool])`:\nCurrently we only support 1-d"
                    " mask array."
                )
            )

        if mask.shape[0] != self.shape[0]:
            raise Error(
                String(
                    "\nError in `numojo.NDArray.__getitem__(mask:"
                    " NDArray[DType.bool])`:\nShape 0 of mask ({}) does not"
                    " match that of array ({})."
                ).format(mask.shape[0], self.shape[0])
            )

        var len_of_result = 0

        # Count number of True
        for i in range(mask.size):
            if mask.item(i):
                len_of_result += 1

        # Change the first number of the ndshape
        var shape = self.shape
        shape._buf[0] = len_of_result

        var result = NDArray[dtype](shape)
        var size_per_item = self.size // self.shape[0]

        # Fill in the values
        var offset = 0
        for i in range(mask.size):
            if mask.item(i):
                memcpy(
                    result._buf.ptr + offset * size_per_item,
                    self._buf.ptr + i * size_per_item,
                    size_per_item,
                )
                offset += 1

        return result

    fn __getitem__(self, mask: List[Bool]) raises -> Self:
        """
        Get items from 0-th dimension of an array according to mask.
        __getitem__(self, mask: NDArray[DType.bool]) raises -> Self.

        Args:
            mask: A list of boolean values.

        Returns:
            NDArray with items from the mask.

        Raises:
            Error: If the mask is not a 1-D array (Currently we only support 1-d mask array).

        Examples:

        ```console
        >>>var a = nm.arange[i8](6)
        >>>print(a)
        [       0       1       2       3       4       5       ]
        1-D array  Shape: [6]  DType: int8  C-cont: True  F-cont: True  own data: True
        >>>print(a[List[Bool](True, False, True, True, False, True)])
        [       0       2       3       5       ]
        1-D array  Shape: [4]  DType: int8  C-cont: True  F-cont: True  own data: True

        var b = nm.arange[i8](12).reshape(Shape(2, 2, 3))
        print(b)
        [[[     0       1       2       ]
        [     3       4       5       ]]
        [[     6       7       8       ]
        [     9       10      11      ]]]
        3-D array  Shape: [2, 2, 3]  DType: int8  C-cont: True  F-cont: False  own data: True
        >>>print(b[List[Bool](False, True)])
        [[[     6       7       8       ]
        [     9       10      11      ]]]
        3-D array  Shape: [1, 2, 3]  DType: int8  C-cont: True  F-cont: True  own data: True
        ```.
        """

        var mask_array = NDArray[DType.bool](shape=Shape(len(mask)))
        for i in range(len(mask)):
            (mask_array._buf.ptr + i).init_pointee_copy(mask[i])

        return self[mask_array]

    fn item(
        self, owned index: Int
    ) raises -> ref [self._buf.ptr.origin, self._buf.ptr.address_space] Scalar[
        dtype
    ]:
        """
        Return the scalar at the coordinates.
        If one index is given, get the i-th item of the array (not buffer).
        It first scans over the first row, even it is a colume-major array.
        If more than one index is given, the length of the indices must match
        the number of dimensions of the array.
        If the ndim is 0 (0-D array), get the value as a mojo scalar.

        Args:
            index: Index of item, counted in row-major way.

        Returns:
            A scalar matching the dtype of the array.

        Raises:
            Error if array is 0-D array (numojo scalar).
            Error if index is equal or larger than array size.

        Examples:

        ```console
        >>> var A = nm.random.randn[nm.f16](2, 2, 2)
        >>> A = A.reshape(A.shape, order="F")
        >>> print(A)
        [[[     0.2446289       0.5419922       ]
        [     0.09643555      -0.90722656     ]]
        [[     1.1806641       0.24389648      ]
        [     0.5234375       1.0390625       ]]]
        3-D array  Shape: [2, 2, 2]  DType: float16  order: F
        >>> for i in range(A.size):
        ...     print(A.item(i))
        0.2446289
        0.5419922
        0.09643555
        -0.90722656
        1.1806641
        0.24389648
        0.5234375
        1.0390625
        >>> print(A.item(0, 1, 1))
        -0.90722656
        ```.
        """
        # For 0-D array, raise error
        if self.ndim == 0:
            raise Error(
                String(
                    "\nError in `numojo.NDArray.item(index: Int)`: "
                    "Cannot index a 0-D array (numojo scalar). "
                    "Use `a.item()` without arguments."
                )
            )

        if index < 0:
            index += self.size

        if (index < 0) or (index >= self.size):
            raise Error(
                String(
                    "\nError in `numojo.NDArray.item(index: Int)`:"
                    "`index` exceeds array size ({})"
                ).format(self.size)
            )

        if self.flags.F_CONTIGUOUS:
            return (self._buf.ptr + _transfer_offset(index, self.strides))[]

        else:
            return (self._buf.ptr + index)[]

    fn item(
        self, *index: Int
    ) raises -> ref [self._buf.ptr.origin, self._buf.ptr.address_space] Scalar[
        dtype
    ]:
        """
        Return the scalar at the coordinates.
        If one index is given, get the i-th item of the array (not buffer).
        It first scans over the first row, even it is a colume-major array.
        If more than one index is given, the length of the indices must match
        the number of dimensions of the array.
        For 0-D array (numojo scalar), return the scalar value.

        Args:
            index: The coordinates of the item.

        Returns:
            A scalar matching the dtype of the array.

        Raises:
            Index is equal or larger than size of dimension.

        Examples:

        ```console
        >>> var A = nm.random.randn[nm.f16](2, 2, 2)
        >>> A = A.reshape(A.shape, order="F")
        >>> print(A)
        [[[     0.2446289       0.5419922       ]
        [     0.09643555      -0.90722656     ]]
        [[     1.1806641       0.24389648      ]
        [     0.5234375       1.0390625       ]]]
        3-D array  Shape: [2, 2, 2]  DType: float16  order: F
        >>> print(A.item(0, 1, 1))
        -0.90722656
        ```.
        """

        if len(index) != self.ndim:
            raise Error(
                String(
                    "\nError in `numojo.NDArray.item(*index: Int)`:"
                    "Number of indices ({}) do not match ndim ({})"
                ).format(len(index), self.ndim)
            )

        # For 0-D array, return the scalar value.
        if self.ndim == 0:
            return self._buf.ptr[]

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
        return (self._buf.ptr + _get_offset(index, self.strides))[]

    fn load(self, owned index: Int) raises -> Scalar[dtype]:
        """
        Safely retrieve i-th item from the underlying buffer.

        `A.load(i)` differs from `A._buf.ptr[i]` due to boundary check.

        Args:
            index: Index of the item.

        Returns:
            The value at the index.

        Raises:
            Index out of bounds.

        Examples:

        ```console
        > array.load(15)
        ```
        returns the item of index 15 from the array's data buffer.

        Note that it does not checked against C-order or F-order.
        ```console
        > # A is a 3x3 matrix, F-order (column-major)
        > A.load(3)  # Row 0, Col 1
        > A.item(3)  # Row 1, Col 0
        ```.
        """

        if index < 0:
            index += self.size

        if (index >= self.size) or (index < 0):
            raise Error(
                String(
                    "\nError in `numojo.NDArray.load(index: Int)`: "
                    "Invalid index: index out of bound [0, {})."
                ).format(self.size)
            )

        return self._buf.ptr[index]

    fn load[width: Int = 1](self, index: Int) raises -> SIMD[dtype, width]:
        """
        Safely loads a SIMD element of size `width` at `index`
        from the underlying buffer.

        To bypass boundary checks, use `self._buf.ptr.load` directly.

        Args:
            index: Index of the item.

        Returns:
            The SIMD element at the index.

        Raises:
            Index out of boundary.
        """

        if (index < 0) or (index >= self.size):
            raise Error(
                String(
                    "\nError in `numojo.NDArray.load[width: Int = 1](index:"
                    " Int)`:\nInvalid index: index out of bound [0, {})."
                ).format(self.size)
            )

        return self._buf.ptr.load[width=width](index)

    fn load[width: Int = 1](self, *indices: Int) raises -> SIMD[dtype, width]:
        """
        Safely loads SIMD element of size `width` at given variadic indices
        from the underlying buffer.

        To bypass boundary checks, use `self._buf.ptr.load` directly.

        Args:
            indices: Variadic indices.

        Returns:
            The SIMD element at the indices.

        Raises:
            Error: If the length of indices does not match the number of dimensions.
            Error: If any of the indices is out of bound.

        Examples:

        ```console
        >>> import numojo
        >>> var A = numojo.random.randn[numojo.f16](2, 2, 2)
        >>> print(A.load(0, 1, 1))
        ```.
        """

        if len(indices) != self.ndim:
            raise (
                String(
                    "\nError in `numojo.NDArray.load[width: Int = 1](*indices:"
                    " Int)`:\nLength of indices ({}) does not match ndim ({})."
                ).format(len(indices), self.ndim)
            )

        for i in range(self.ndim):
            if (indices[i] < 0) or (indices[i] >= self.shape[i]):
                raise Error(
                    String(
                        "\nError in `numojo.NDArray.load[width: Int ="
                        " 1](*indices: Int)`:\nInvalid index at {}-th dim:"
                        " index out of bound [0, {})."
                    ).format(i, self.shape[i])
                )

        var idx: Int = _get_offset(indices, self.strides)
        return self._buf.ptr.load[width=width](idx)

    # ===-------------------------------------------------------------------===#
    # Setter dunders and other setter methods
    #
    # Basic Setter Methods
    # fn _setitem(self, *indices: Int, val: Scalar[dtype])                      # Direct unsafe setter
    # fn __setitem__(mut self, idx: Int, val: Self) raises                      # Set by single index
    # fn __setitem__(mut self, index: Item, val: Scalar[dtype]) raises          # Set by coordinate list
    # fn __setitem__(mut self, mask: NDArray[DType.bool], value: Scalar[dtype]) # Set by boolean mask

    # Slice-based Setters
    # fn __setitem__(mut self, *slices: Slice, val: Self) raises                # Set by variable slices
    # fn __setitem__(mut self, slices: List[Slice], val: Self) raises           # Set by list of slices
    # fn __setitem__(mut self, *slices: Variant[Slice, Int], val: Self) raises  # Set by mix of slices/ints

    # Index-based Setters
    # fn __setitem__(self, indices: NDArray[DType.index], val: NDArray) raises  # Set by index array
    # fn __setitem__(mut self, mask: NDArray[DType.bool], val: NDArray[dtype])  # Set by boolean mask array

    # Helper Methods
    # fn itemset(mut self, index: Variant[Int, List[Int]], item: Scalar[dtype]) # Set single item
    # fn store(self, owned index: Int, val: Scalar[dtype]) raises               # Store with bounds checking
    # fn store[width: Int](mut self, index: Int, val: SIMD[dtype, width])       # Store SIMD value
    # fn store[width: Int = 1](mut self, *indices: Int, val: SIMD[dtype, width])# Store SIMD at coordinates
    # ===-------------------------------------------------------------------===#

    fn _setitem(self, *indices: Int, val: Scalar[dtype]):
        """
        (UNSAFE! for internal use only.)
        Set item at indices and bypass all boundary checks.

        Args:
            indices: Indices to set the value.
            val: Value to set.

        Notes:
            This function is unsafe and for internal use only.

        Examples:

        ```mojo
        import numojo
        var A = numojo.ones(numojo.Shape(2,3,4))
        A._setitem(1,2,3, val=10)
        ```
        """
        var index_of_buffer: Int = 0
        for i in range(self.ndim):
            index_of_buffer += indices[i] * self.strides._buf[i]
        self._buf.ptr[index_of_buffer] = val

    fn __setitem__(mut self, idx: Int, val: Self) raises:
        """
        Set a slice of array with given array.

        Args:
            idx: Index to set.
            val: Value to set.

        Raises:
            Error: If the index is out of bounds.
            Error: If the value is a 0-D array.

        Examples:

        ```console
        >>>import numojo as nm
        >>>var A = nm.random.rand[nm.i16](3, 2)
        >>>var B = nm.random.rand[nm.i16](3)
        >>>A[1:4] = B
        ```.
        """

        var normalized_index = idx
        if normalized_index < 0:
            normalized_index = self.shape[0] + idx
        if normalized_index >= self.shape[0]:
            raise Error(
                String(
                    "\nError in `numojo.NDArray.__setitem__(idx: Int, val:"
                    " Self)`:\nIndex out of bounds: index ({}) is out of bounds"
                    " [0, {})."
                ).format(idx, self.shape[0])
            )

        # If the ndim is 0, then it is a numojo scalar (0-D array).
        # Not allow to set value to 0-D array.
        if self.ndim == 0 or val.ndim == 0:
            raise Error(
                String(
                    "\nError in `numojo.NDArray.__setitem__(idx: Int, val:"
                    " Self)`:\nCannot set values to a 0-d array."
                )
            )

        var slice_list = List[Slice]()
        if idx >= self.shape[0]:
            raise Error(
                String(
                    "\nError in `numojo.NDArray.__setitem__(idx: Int, val:"
                    " Self)`:\nSlice value exceeds the array shape!\nThe {}-th"
                    " dimension is of size {}.\nThe slice goes from {} to {}"
                ).format(
                    0,
                    self.shape[0],
                    idx,
                    idx + 1,
                )
            )
        slice_list.append(Slice(idx, idx + 1, 1))
        if self.ndim > 1:
            for i in range(1, self.ndim):
                var size_at_dim: Int = self.shape[i]
                slice_list.append(Slice(0, size_at_dim, 1))

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
                raise Error(
                    String(
                        "\nError in `numojo.NDArray.__setitem__(idx: Int, val:"
                        " Self)`: Shape mismatch! Cannot set the array values"
                        " with given array. The {}-th dimension of the array"
                        " is of shape {}. The {}-th dimension of the value is"
                        " of shape {}."
                    ).format(nshape[i], val.shape[i])
                )

        var noffset: Int = 0
        if self.flags.C_CONTIGUOUS:
            noffset = 0
            for i in range(ndims):
                var temp_stride: Int = 1
                for j in range(i + 1, ndims):
                    temp_stride *= nshape[j]
                nstrides.append(temp_stride)
            for i in range(slice_list.__len__()):
                noffset += slice_list[i].start.value() * self.strides[i]
        elif self.flags.F_CONTIGUOUS:
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
            val, self, nshape, ncoefficients, nstrides, noffset, index
        )

    fn __setitem__(mut self, index: Item, val: Scalar[dtype]) raises:
        """
        Sets the value at the index list.

        Args:
            index: Index list.
            val: Value to set.

        Raises:
            Error: If the length of index does not match the number of dimensions.
            Error: If any of the indices is out of bound.

        Examples:

        ```console
        >>> import numojo
        >>> var A = numojo.random.rand[numojo.i16](2, 2, 2)
        >>> A[numojo.Item(0, 1, 1)] = 10
        ```.
        """
        if index.__len__() != self.ndim:
            raise Error(
                String(
                    "\nError in `numojo.NDArray.__setitem__(index: Item, val:"
                    " Scalar[dtype])`: Length of `index` does not match the"
                    " number of dimensions! Length of indices is {}. The"
                    " array dimension is {}."
                ).format(index.__len__(), self.ndim)
            )
        for i in range(index.__len__()):
            if index[i] >= self.shape[i]:
                raise Error(
                    String(
                        "\nError in `numojo.NDArray.__setitem__(index: Item,"
                        " val: Scalar[dtype])`: `index` exceeds the size! For"
                        " {}-th dimension: The index value is {}. The size of"
                        " the corresponding dimension is {}"
                    ).format(i, index[i], self.shape[i])
                )
        var idx: Int = _get_offset(index, self.strides)
        self._buf.ptr.store(idx, val)

    # only works if array is called as array.__setitem__(), mojo compiler doesn't parse it implicitly
    fn __setitem__(
        mut self, mask: NDArray[DType.bool], value: Scalar[dtype]
    ) raises:
        """
        Sets the value of the array at the indices where the mask is true.

        Args:
            mask: Boolean mask array.
            value: Value to set.

        Raises:
            Error: If the mask and the array do not have the same shape.

        Examples:

        ```console
        >>> import numojo
        >>> var A = numojo.random.rand[numojo.i16](2, 2, 2)
        >>> var mask = A > 0.5
        >>> A[mask] = 10
        ```.
        """
        if (
            mask.shape != self.shape
        ):  # this behavious could be removed potentially
            raise Error(
                String(
                    "\nError in `numojo.NDArray.__setitem__(mask:"
                    " NDArray[DType.bool], value: Scalar[dtype])`:\nMask and"
                    " array must have the same shape.\nThe mask shape is"
                    " {}.\nThe array shape is {}."
                ).format(mask.shape, self.shape)
            )

        for i in range(mask.size):
            if mask._buf.ptr.load[width=1](i):
                self._buf.ptr.store(i, value)

    fn __setitem__(mut self, *slices: Slice, val: Self) raises:
        """
        Sets the elements of the array at the slices with given array.

        Args:
            slices: Variadic slices.
            val: A NDArray to set.

        Raises:
            Error: If the length of slices does not match the number of dimensions.
            Error: If any of the slices is out of bound.

        Examples:

        ```console
        >>> import numojo
        >>> var A = numojo.random.rand[numojo.i16](2, 2, 2)
        >>> A[1:3, 2:4] = numojo.random.rand[numojo.i16](2, 2)
        ```.
        """
        var slice_list: List[Slice] = List[Slice]()
        for i in range(slices.__len__()):
            slice_list.append(slices[i])
        self.__setitem__(slices=slice_list, val=val)

    fn __setitem__(mut self, slices: List[Slice], val: Self) raises:
        """
        Sets the slices of an array from list of slices and array.

        Args:
            slices: List of slices.
            val: Value to set.

        Raises:
            Error: If the length of slices does not match the number of dimensions.
            Error: If any of the slices is out of bound.

        Examples:

        ```console
        >>> var a = nm.arange[i8](16).reshape(Shape(4, 4))
        print(a)
        [[      0       1       2       3       ]
         [      4       5       6       7       ]
         [      8       9       10      11      ]
         [      12      13      14      15      ]]
        2-D array  Shape: [4, 4]  DType: int8  C-cont: True  F-cont: False  own data: True
        >>> a[2:4, 2:4] = a[0:2, 0:2]
        print(a)
        [[      0       1       2       3       ]
         [      4       5       6       7       ]
         [      8       9       0       1       ]
         [      12      13      4       5       ]]
        2-D array  Shape: [4, 4]  DType: int8  C-cont: True  F-cont: False  own data: True
        ```.
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
                raise Error(
                    String(
                        "\nError in `numojo.NDArray.__setitem__(slices:"
                        " List[Slice], val: Self)`: Slice value exceeds the"
                        " array shape! The {}-th dimension is of size {}. The"
                        " slice goes from {} to {}"
                    ).format(
                        i,
                        self.shape[i],
                        slice_list[i].start.value(),
                        slice_list[i].end.value(),
                    )
                )
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
                raise Error(
                    String(
                        "\nError in `numojo.NDArray.__setitem__(slices:"
                        " List[Slice], val: Self)`: Shape mismatch! For {}-th"
                        " dimension: The size of the array is {}. The size"
                        " of the input value is {}."
                    ).format(i, nshape[i], val.shape[i])
                )

        var noffset: Int = 0
        if self.flags.C_CONTIGUOUS:
            noffset = 0
            for i in range(ndims):
                var temp_stride: Int = 1
                for j in range(i + 1, ndims):  # temp
                    temp_stride *= nshape[j]
                nstrides.append(temp_stride)
            for i in range(slice_list.__len__()):
                noffset += slice_list[i].start.value() * self.strides[i]
        elif self.flags.F_CONTIGUOUS:
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
            val, self, nshape, ncoefficients, nstrides, noffset, index
        )

    fn __setitem__(mut self, *slices: Variant[Slice, Int], val: Self) raises:
        """
        Gets items by a series of either slices or integers.

        Args:
            slices: Variadic slices or integers.
            val: Value to set.

        Raises:
            Error: If the length of slices does not match the number of dimensions.
            Error: If any of the slices is out of bound.

        Examples:

        ```console
        >>> var a = nm.arange[i8](16).reshape(Shape(4, 4))
        print(a)
        [[      0       1       2       3       ]
         [      4       5       6       7       ]
         [      8       9       10      11      ]
         [      12      13      14      15      ]]
        2-D array  Shape: [4, 4]  DType: int8  C-cont: True  F-cont: False  own data: True
        >>> a[0, Slice(2, 4)] = a[3, Slice(0, 2)]
        print(a)
        [[      0       1       12      13      ]
         [      4       5       6       7       ]
         [      8       9       10      11      ]
         [      12      13      14      15      ]]
        2-D array  Shape: [4, 4]  DType: int8  C-cont: True  F-cont: False  own data: True
        ```.
        """
        var n_slices: Int = slices.__len__()
        if n_slices > self.ndim:
            raise Error(
                String(
                    "\nError in `numojo.NDArray.__setitem__(*slices:"
                    " Variant[Slice, Int], val: Self)`: No of slices greater"
                    " than rank of array. The number of slices is {}. The"
                    " rank of the array is {}."
                ).format(n_slices, self.ndim)
            )
        var slice_list: List[Slice] = List[Slice]()

        var count_int = 0
        for i in range(len(slices)):
            if slices[i].isa[Slice]():
                slice_list.append(slices[i]._get_ptr[Slice]()[0])
            elif slices[i].isa[Int]():
                count_int += 1
                var int: Int = slices[i]._get_ptr[Int]()[0]
                slice_list.append(Slice(int, int + 1, 1))

        if n_slices < self.ndim:
            for i in range(n_slices, self.ndim):
                var size_at_dim: Int = self.shape[i]
                slice_list.append(Slice(0, size_at_dim, 1))

        self.__setitem__(slices=slice_list, val=val)

    # TODO: fix this setter, add bound checks. Not sure about it's use case.
    fn __setitem__(self, index: NDArray[DType.index], val: NDArray) raises:
        """
        Returns the items of the array from an array of indices.

        Args:
            index: Array of indices.
            val: Value to set.

        Examples:

        ```console
        > var X = nm.NDArray[nm.i8](3,random=True)
        > print(X)
        [       32      21      53      ]
        1-D array  Shape: [3]  DType: int8
        > print(X.argsort())
        [       1       0       2       ]
        1-D array  Shape: [3]  DType: index
        > print(X[X.argsort()])
        [       21      32      53      ]
        1-D array  Shape: [3]  DType: int8
        ```.
        """

        for i in range(len(index)):
            self.store(Int(index.load(i)), rebind[Scalar[dtype]](val.load(i)))

    fn __setitem__(
        mut self, mask: NDArray[DType.bool], val: NDArray[dtype]
    ) raises:
        """
        Sets the value of the array at the indices where the mask is true.

        Args:
            mask: Boolean mask array.
            val: Value to set.

        Raises:
            Error: If the mask and the array do not have the same shape.

        Examples:

        ```console
        >>> import numojo
        >>> var A = numojo.random.rand[numojo.i16](2, 2, 2)
        >>> var mask = A > 0.5
        >>> A[mask] = 10
        ```.
        """
        if (
            mask.shape != self.shape
        ):  # this behavious could be removed potentially
            raise Error(
                String(
                    "\nShape of mask does not match the shape of array. "
                    "The mask shape is {}. "
                    "The array shape is {}."
                ).format(mask.shape, self.shape)
            )

        for i in range(mask.size):
            if mask._buf.ptr.load(i):
                self._buf.ptr.store(i, val._buf.ptr.load(i))

    fn itemset(
        mut self, index: Variant[Int, List[Int]], item: Scalar[dtype]
    ) raises:
        """Set the scalar at the coordinates.

        Args:
            index: The coordinates of the item.
                Can either be `Int` or `List[Int]`.
                If `Int` is passed, it is the index of i-th item of the whole array.
                If `List[Int]` is passed, it is the coordinate of the item.
            item: The scalar to be set.

        Raises:
            Error: If the index is out of bound.
            Error: If the length of index does not match the number of dimensions.

        Note:
            This is similar to `numpy.ndarray.itemset`.
            The difference is that we takes in `List[Int]`, but numpy takes in a tuple.

        Examples:

        ```
        import numojo as nm
        fn main() raises:
            var A = nm.zeros[nm.i16](3, 3)
            print(A)
            A.itemset(5, 256)
            print(A)
            A.itemset(List(1,1), 1024)
            print(A)
        ```
        ```console
        [[      0       0       0       ]
        [      0       0       0       ]
        [      0       0       0       ]]
        2-D array  Shape: [3, 3]  DType: int16
        [[      0       0       0       ]
        [      0       0       256     ]
        [      0       0       0       ]]
        2-D array  Shape: [3, 3]  DType: int16
        [[      0       0       0       ]
        [      0       1024    256     ]
        [      0       0       0       ]]
        2-D array  Shape: [3, 3]  DType: int16
        ```.
        """

        # If one index is given
        if index.isa[Int]():
            var idx = index._get_ptr[Int]()[]
            if idx < self.size:
                if self.flags.F_CONTIGUOUS:
                    # column-major should be converted to row-major
                    # The following code can be taken out as a function that
                    # convert any index to coordinates according to the order
                    var c_stride = NDArrayStrides(shape=self.shape)
                    var c_coordinates = List[Int]()
                    for i in range(c_stride.ndim):
                        var coordinate = idx // c_stride[i]
                        idx = idx - c_stride[i] * coordinate
                        c_coordinates.append(coordinate)
                    self._buf.ptr.store(
                        _get_offset(c_coordinates, self.strides), item
                    )

                self._buf.ptr.store(idx, item)
            else:
                raise Error(
                    String(
                        "\nError in `numojo.NDArray.itemset(index: Variant[Int,"
                        " List[Int]], item: Scalar[dtype])`:\nElements of"
                        " `index` ({}) \nexceed the array size ({})."
                    ).format(idx, self.size)
                )

        else:
            var indices = index._get_ptr[List[Int]]()[]
            # If more than one index is given
            if indices.__len__() != self.ndim:
                raise Error(
                    "\nError in `numojo.NDArray.itemset(index: Variant[Int,"
                    " List[Int]], item: Scalar[dtype])`:\nLength of Indices do"
                    " not match the shape"
                )
            for i in range(indices.__len__()):
                if indices[i] >= self.shape[i]:
                    raise Error(
                        "\nError in `numojo.NDArray.itemset(index: Variant[Int,"
                        " List[Int]], item: Scalar[dtype])`:\nElements of"
                        " `index` exceed the array shape"
                    )
            self._buf.ptr.store(_get_offset(indices, self.strides), item)

    fn store(self, owned index: Int, val: Scalar[dtype]) raises:
        """
        Safely store a scalar to i-th item of the underlying buffer.

        `A.store(i, a)` differs from `A._buf.ptr[i] = a` due to boundary check.

        Args:
            index: Index of the item.
            val: Value to store.

        Raises:
            Index out of boundary.

        Examples:

        ```console
        > array.store(15, val = 100)
        ```
        sets the item of index 15 of the array's data buffer to 100.
        Note that it does not checked against C-order or F-order.
        """

        if index < 0:
            index += self.size

        if (index >= self.size) or (index < 0):
            raise Error(
                String(
                    "\nError in `numojo.NDArray.store(index: Int, val:"
                    " Scalar[dtype])`:\nInvalid index: index out of bound [0,"
                    " {})."
                ).format(self.size)
            )

        self._buf.ptr[index] = val

    fn store[width: Int](mut self, index: Int, val: SIMD[dtype, width]) raises:
        """
        Safely stores SIMD element of size `width` at `index`
        of the underlying buffer.

        To bypass boundary checks, use `self._buf.ptr.store` directly.

        Args:
            index: Index of the item.
            val: Value to store.

        Raises:
            Index out of boundary.

        Examples:

        ```console
        > array.store(15, val = 100)
        ```
        sets the item of index 15 of the array's data buffer to 100.
        """

        if (index < 0) or (index >= self.size):
            raise Error(
                String(
                    "\nError in `numojo.NDArray.store[width: Int](index: Int,"
                    " val: SIMD[dtype, width])`:\nInvalid index: index out of"
                    " bound [0, {})."
                ).format(self.size)
            )

        self._buf.ptr.store(index, val)

    fn store[
        width: Int = 1
    ](mut self, *indices: Int, val: SIMD[dtype, width]) raises:
        """
        Safely stores SIMD element of size `width` at given variadic indices
        of the underlying buffer.

        To bypass boundary checks, use `self._buf.ptr.store` directly.

        Args:
            indices: Variadic indices.
            val: Value to store.

        Raises:
            Index out of boundary.

        Examples:

        ```console
        >>> import numojo
        >>> var A = numojo.random.rand[numojo.i16](2, 2, 2)
        >>> A.store(0, 1, 1, val=100)
        ```.
        """

        if len(indices) != self.ndim:
            raise (
                String(
                    "\nError in `numojo.NDArray.store[width: Int](*indices:"
                    " Int, val: SIMD[dtype, width])`:\nLength of indices {}"
                    " does not match ndim {}".format(len(indices), self.ndim)
                )
            )

        for i in range(self.ndim):
            if (indices[i] < 0) or (indices[i] >= self.shape[i]):
                raise Error(
                    String(
                        "\nError in `numojo.NDArray.store[width: Int](*indices:"
                        " Int, val: SIMD[dtype, width])`:\nInvalid index at"
                        " {}-th dim: index out of bound [0, {})."
                    ).format(i, self.shape[i])
                )

        var idx: Int = _get_offset(indices, self.strides)
        self._buf.ptr.store(idx, val)

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    # TODO: We should make a version that checks nonzero/not_nan
    fn __bool__(self) raises -> Bool:
        """
        If all true return true.

        Raises:
            Error: If the array is not 0-D or length-1.

        Examples:

        ```console
        >>> import numojo
        >>> var A = numojo.random.rand[numojo.i16](2, 2, 2)
        >>> print(bool(A))
        ```.
        """
        if (self.size == 1) or (self.ndim == 0):
            return Bool(self._buf.ptr[])

        else:
            raise Error(
                "\nError in `numojo.NDArray.__bool__(self)`: "
                "Only 0-D arrays (numojo scalar) or length-1 arrays "
                "can be converted to Bool."
                "The truth value of an array with more than one element is "
                "ambiguous. Use a.any() or a.all()."
            )

    fn __int__(self) raises -> Int:
        """
        Gets `Int` representation of the array.

        Only 0-D arrays or length-1 arrays can be converted to scalars.

        Returns:
            Int representation of the array.

        Raises:
            Error: If the array is not 0-D or length-1.

        Examples:

        ```console
        > var A = NDArray[dtype](6, random=True)
        > print(Int(A))

        Unhandled exception caught during execution: Only 0-D arrays or length-1 arrays can be converted to scalars
        mojo: error: execution exited with a non-zero result: 1

        > var B = NDArray[dtype](1, 1, random=True)
        > print(Int(B))
        14
        ```.
        """
        if (self.size == 1) or (self.ndim == 0):
            return Int(self._buf.ptr[])
        else:
            raise Error(
                "\nError in `numojo.NDArray.__int__(self)`: "
                "Only 0-D arrays (numojo scalar) or length-1 arrays "
                "can be converted to scalars."
            )

    fn __float__(self) raises -> Float64:
        """
        Gets `Float64` representation of the array.

        Only 0-D arrays or length-1 arrays can be converted to scalars.

        Raises:
            Error: If the array is not 0-D or length-1.

        Returns:
            Float representation of the array.
        """
        if (self.size == 1) or (self.ndim == 0):
            return Float64(self._buf.ptr[])
        else:
            raise Error(
                "\nError in `numojo.NDArray.__float__(self)`: "
                "Only 0-D arrays (numojo scalar) or length-1 arrays "
                "can be converted to scalars."
            )

    fn __pos__(self) raises -> Self:
        """
        Unary positve returns self unless boolean type.
        """
        if self.dtype is DType.bool:
            raise Error(
                "ndarray:NDArrray:__pos__: pos does not accept bool type arrays"
            )
        return self

    fn __neg__(self) raises -> Self:
        """
        Unary negative returns self unless boolean type.

        For bolean use `__invert__`(~)
        """
        if self.dtype is DType.bool:
            raise Error(
                "ndarray:NDArrray:__pos__: pos does not accept bool type arrays"
            )
        return self * Scalar[dtype](-1.0)

    # maybe they don't need conversion with astype.
    @always_inline("nodebug")
    fn __eq__[
        OtherDtype: DType,
        ResultDType: DType = TypeCoercion.result[dtype, OtherDtype](),
    ](self, other: NDArray[OtherDtype]) raises -> NDArray[DType.bool]:
        """
        Itemwise equivalence.

        Parameters:
            OtherDtype: The data type of the other array.
            ResultDType: The data type of the result array.

        Args:
            other: The other array to compare with.

        Returns:
            An array of boolean values.
        """
        return comparison.equal[ResultDType](
            self.astype[ResultDType](), other.astype[ResultDType]()
        )

    @always_inline("nodebug")
    fn __eq__[
        OtherDtype: DType,
        ResultDType: DType = TypeCoercion.result[dtype, OtherDtype](),
    ](self, other: Scalar[OtherDtype]) raises -> NDArray[DType.bool]:
        """
        Itemwise equivalence.

        Parameters:
            OtherDtype: The data type of the other Scalar.
            ResultDType: The data type of the result array.

        Args:
            other: The other Scalar to compare with.

        Returns:
            An array of boolean values.
        """
        return comparison.equal[ResultDType](
            self.astype[ResultDType](), other.cast[ResultDType]()
        )

    @always_inline("nodebug")
    fn __eq__(self, other: Self) raises -> NDArray[DType.bool]:
        """
        Itemwise equivalence.

        Args:
            other: The other array to compare with.

        Returns:
            An array of boolean values.
        """
        return comparison.equal[dtype](self, other)

    @always_inline("nodebug")
    fn __eq__(self, other: SIMD[dtype, 1]) raises -> NDArray[DType.bool]:
        """
        Itemwise equivalence between scalar and Array.

        Args:
            other: The other SIMD value to compare with.

        Returns:
            An array of boolean values.
        """
        return comparison.equal[dtype](self, other)

    @always_inline("nodebug")
    fn __ne__[
        OtherDtype: DType,
        ResultDType: DType = TypeCoercion.result[dtype, OtherDtype](),
    ](self, other: NDArray[OtherDtype]) raises -> NDArray[DType.bool]:
        """
        Itemwise nonequivelence.

        Parameters:
            OtherDtype: The data type of the other array.
            ResultDType: The data type of the result array.

        Args:
            other: The other array to compare with.

        Returns:
            An array of boolean values.
        """
        return comparison.not_equal[ResultDType](
            self.astype[ResultDType](), other.astype[ResultDType]()
        )

    @always_inline("nodebug")
    fn __ne__[
        OtherDtype: DType,
        ResultDType: DType = TypeCoercion.result[dtype, OtherDtype](),
    ](self, other: Scalar[OtherDtype]) raises -> NDArray[DType.bool]:
        """
        Itemwise nonequivelence between scalar and Array.

        Parameters:
            OtherDtype: The data type of the other Scalar.
            ResultDType: The data type of the result array.

        Args:
            other: The other Scalar to compare with.

        Returns:
            An array of boolean values.
        """
        return comparison.not_equal[ResultDType](
            self.astype[ResultDType](), other.cast[ResultDType]()
        )

    @always_inline("nodebug")
    fn __ne__(self, other: SIMD[dtype, 1]) raises -> NDArray[DType.bool]:
        """
        Itemwise nonequivelence.

        Args:
            other: The other SIMD value to compare with.

        Returns:
            An array of boolean values.
        """
        return comparison.not_equal[dtype](self, other)

    @always_inline("nodebug")
    fn __ne__(self, other: NDArray[dtype]) raises -> NDArray[DType.bool]:
        """
        Itemwise nonequivelence between scalar and Array.

        Args:
            other: The other array to compare with.

        Returns:
            An array of boolean values.
        """
        return comparison.not_equal[dtype](self, other)

    @always_inline("nodebug")
    fn __lt__[
        OtherDtype: DType,
        ResultDType: DType = TypeCoercion.result[dtype, OtherDtype](),
    ](self, other: Scalar[OtherDtype]) raises -> NDArray[DType.bool]:
        """
        Itemwise less than.

        Parameters:
            OtherDtype: The data type of the other Scalar.
            ResultDType: The data type of the result array.

        Args:
            other: The other Scalar to compare with.

        Returns:
            An array of boolean values.
        """
        return comparison.less[ResultDType](
            self.astype[ResultDType](), other.cast[ResultDType]()
        )

    @always_inline("nodebug")
    fn __lt__[
        OtherDtype: DType,
        ResultDType: DType = TypeCoercion.result[dtype, OtherDtype](),
    ](self, other: NDArray[OtherDtype]) raises -> NDArray[DType.bool]:
        """
        Itemwise less than between scalar and Array.

        Parameters:
            OtherDtype: The data type of the other array.
            ResultDType: The data type of the result array.

        Args:
            other: The other array to compare with.

        Returns:
            An array of boolean values.
        """
        return comparison.less[ResultDType](
            self.astype[ResultDType](), other.astype[ResultDType]()
        )

    @always_inline("nodebug")
    fn __lt__(self, other: SIMD[dtype, 1]) raises -> NDArray[DType.bool]:
        """
        Itemwise less than.

        Args:
            other: The other SIMD value to compare with.

        Returns:
            An array of boolean values.
        """
        return comparison.less[dtype](self, other)

    @always_inline("nodebug")
    fn __lt__(self, other: NDArray[dtype]) raises -> NDArray[DType.bool]:
        """
        Itemwise less than between scalar and Array.

        Args:
            other: The other array to compare with.

        Returns:
            An array of boolean values.
        """
        return comparison.less[dtype](self, other)

    @always_inline("nodebug")
    fn __le__[
        OtherDtype: DType,
        ResultDType: DType = TypeCoercion.result[dtype, OtherDtype](),
    ](self, other: Scalar[OtherDtype]) raises -> NDArray[DType.bool]:
        """
        Itemwise less than or equal to.

        Parameters:
            OtherDtype: The data type of the other Scalar.
            ResultDType: The data type of the result array.

        Args:
            other: The other Scalar to compare with.

        Returns:
            An array of boolean values.
        """
        return comparison.less_equal[ResultDType](
            self.astype[ResultDType](), other.cast[ResultDType]()
        )

    @always_inline("nodebug")
    fn __le__[
        OtherDtype: DType,
        ResultDType: DType = TypeCoercion.result[dtype, OtherDtype](),
    ](self, other: NDArray[OtherDtype]) raises -> NDArray[DType.bool]:
        """
        Itemwise less than or equal to between scalar and Array.

        Parameters:
            OtherDtype: The data type of the other array.
            ResultDType: The data type of the result array.

        Args:
            other: The other array to compare with.

        Returns:
            An array of boolean values.
        """
        return comparison.less_equal[ResultDType](
            self.astype[ResultDType](), other.astype[ResultDType]()
        )

    @always_inline("nodebug")
    fn __le__(self, other: SIMD[dtype, 1]) raises -> NDArray[DType.bool]:
        """
        Itemwise less than or equal to.

        Args:
            other: The other SIMD value to compare with.

        Returns:
            An array of boolean values.
        """
        return comparison.less_equal[dtype](self, other)

    @always_inline("nodebug")
    fn __le__(self, other: NDArray[dtype]) raises -> NDArray[DType.bool]:
        """
        Itemwise less than or equal to between scalar and Array.

        Args:
            other: The other array to compare with.

        Returns:
            An array of boolean values.
        """
        return comparison.less_equal[dtype](self, other)

    @always_inline("nodebug")
    fn __gt__[
        OtherDtype: DType,
        ResultDType: DType = TypeCoercion.result[dtype, OtherDtype](),
    ](self, other: Scalar[OtherDtype]) raises -> NDArray[DType.bool]:
        """
        Itemwise greater than.

        Parameters:
            OtherDtype: The data type of the other Scalar.
            ResultDType: The data type of the result array.

        Args:
            other: The other Scalar to compare with.

        Returns:
            An array of boolean values.
        """
        return comparison.greater[ResultDType](
            self.astype[ResultDType](), other.cast[ResultDType]()
        )

    @always_inline("nodebug")
    fn __gt__[
        OtherDtype: DType,
        ResultDType: DType = TypeCoercion.result[dtype, OtherDtype](),
    ](self, other: NDArray[OtherDtype]) raises -> NDArray[DType.bool]:
        """
        Itemwise greater than between scalar and Array.

        Parameters:
            OtherDtype: The data type of the other array.
            ResultDType: The data type of the result array.

        Args:
            other: The other array to compare with.

        Returns:
            An array of boolean values.
        """
        return comparison.greater[ResultDType](
            self.astype[ResultDType](), other.astype[ResultDType]()
        )

    @always_inline("nodebug")
    fn __gt__(self, other: SIMD[dtype, 1]) raises -> NDArray[DType.bool]:
        """
        Itemwise greater than.

        Args:
            other: The other SIMD value to compare with.

        Returns:
            An array of boolean values.
        """
        return comparison.greater[dtype](self, other)

    @always_inline("nodebug")
    fn __gt__(self, other: NDArray[dtype]) raises -> NDArray[DType.bool]:
        """
        Itemwise greater than between scalar and Array.

        Args:
            other: The other array to compare with.

        Returns:
            An array of boolean values.
        """
        return comparison.greater[dtype](self, other)

    @always_inline("nodebug")
    fn __ge__[
        OtherDtype: DType,
        ResultDType: DType = TypeCoercion.result[dtype, OtherDtype](),
    ](self, other: Scalar[OtherDtype]) raises -> NDArray[DType.bool]:
        """
        Itemwise greater than or equal to.

        Parameters:
            OtherDtype: The data type of the other Scalar.
            ResultDType: The data type of the result array.

        Args:
            other: The other Scalar to compare with.

        Returns:
            An array of boolean values.
        """
        return comparison.greater_equal[ResultDType](
            self.astype[ResultDType](), other.cast[ResultDType]()
        )

    @always_inline("nodebug")
    fn __ge__[
        OtherDtype: DType,
        ResultDType: DType = TypeCoercion.result[dtype, OtherDtype](),
    ](self, other: NDArray[OtherDtype]) raises -> NDArray[DType.bool]:
        """
        Itemwise greater than or equal to between scalar and Array.

        Parameters:
            OtherDtype: The data type of the other array.
            ResultDType: The data type of the result array.

        Args:
            other: The other array to compare with.

        Returns:
            An array of boolean values.
        """
        return comparison.greater_equal[ResultDType](
            self.astype[ResultDType](), other.astype[ResultDType]()
        )

    @always_inline("nodebug")
    fn __ge__(self, other: SIMD[dtype, 1]) raises -> NDArray[DType.bool]:
        """
        Itemwise greater than or equal to.

        Args:
            other: The other SIMD value to compare with.

        Returns:
            An array of boolean values.
        """
        return comparison.greater_equal[dtype](self, other)

    @always_inline("nodebug")
    fn __ge__(self, other: NDArray[dtype]) raises -> NDArray[DType.bool]:
        """
        Itemwise greater than or equal to between Array and Array.

        Args:
            other: The other array to compare with.

        Returns:
            An array of boolean values.
        """
        return comparison.greater_equal[dtype](self, other)

    fn __add__[
        OtherDType: DType,
        ResultDType: DType = TypeCoercion.result[dtype, OtherDType](),
    ](self, other: Scalar[OtherDType]) raises -> NDArray[ResultDType]:
        """
        Enables `array + scalar`.

        Parameters:
            OtherDType: The data type of the other Scalar.
            ResultDType: The data type of the result array.

        Args:
            other: The other Scalar to compare with.

        Returns:
            An array of the result of the addition.
        """
        return math.add[ResultDType](
            self.astype[ResultDType](), other.cast[ResultDType]()
        )

    """ ARITHMETIC OPERATORS """

    fn __add__[
        OtherDType: DType,
        ResultDType: DType = TypeCoercion.result[dtype, OtherDType](),
    ](self, other: NDArray[OtherDType]) raises -> NDArray[ResultDType]:
        """
        Enables `array + array`.
        """
        return math.add[ResultDType](
            self.astype[ResultDType](), other.astype[ResultDType]()
        )

    fn __add__(self, other: Scalar[dtype]) raises -> Self:
        """
        Enables `array + scalar`.
        """
        return math.add[dtype](self, other)

    fn __add__(self, other: Self) raises -> Self:
        """
        Enables `array + array`.
        """
        return math.add[dtype](self, other)

    fn __radd__[
        OtherDType: DType,
        ResultDType: DType = TypeCoercion.result[dtype, OtherDType](),
    ](self, other: Scalar[OtherDType]) raises -> NDArray[ResultDType]:
        """
        Enables `scalar + array`.
        """
        return math.add[ResultDType](
            self.astype[ResultDType](), other.cast[ResultDType]()
        )

    fn __radd__(mut self, other: SIMD[dtype, 1]) raises -> Self:
        """
        Enables `scalar + array`.
        """
        return math.add[dtype](self, other)

    # TODO make an inplace version of arithmetic functions for the i dunders
    # Cannot do type coercion for iadd
    fn __iadd__(mut self, other: SIMD[dtype, 1]) raises:
        """
        Enables `array += scalar`.
        """
        self = _af.math_func_one_array_one_SIMD_in_one_array_out[
            dtype, SIMD.__add__
        ](self, other)

    fn __iadd__(mut self, other: Self) raises:
        """
        Enables `array *= array`.
        """
        self = _af.math_func_2_array_in_one_array_out[dtype, SIMD.__add__](
            self, other
        )

    fn __sub__[
        OtherDType: DType,
        ResultDType: DType = TypeCoercion.result[dtype, OtherDType](),
    ](self, other: Scalar[OtherDType]) raises -> NDArray[ResultDType]:
        """
        Enables `array - scalar`.
        """
        return math.sub[ResultDType](
            self.astype[ResultDType](), other.cast[ResultDType]()
        )

    fn __sub__[
        OtherDType: DType,
        ResultDType: DType = TypeCoercion.result[dtype, OtherDType](),
    ](self, other: NDArray[OtherDType]) raises -> NDArray[ResultDType]:
        """
        Enables `array - array`.
        """
        return math.sub[ResultDType](
            self.astype[ResultDType](), other.astype[ResultDType]()
        )

    fn __sub__(self, other: Scalar[dtype]) raises -> Self:
        """
        Enables `array - scalar`.
        """
        return math.sub[dtype](self, other)

    fn __sub__(self, other: Self) raises -> Self:
        """
        Enables `array - array`.
        """
        return math.sub[dtype](self, other)

    fn __rsub__[
        OtherDType: DType,
        ResultDType: DType = TypeCoercion.result[dtype, OtherDType](),
    ](self, other: Scalar[OtherDType]) raises -> NDArray[ResultDType]:
        """
        Enables `scalar - array`.
        """
        return math.sub[ResultDType](
            other.cast[ResultDType](), self.astype[ResultDType]()
        )

    fn __rsub__(mut self, other: SIMD[dtype, 1]) raises -> Self:
        """
        Enables `scalar - array`.
        """
        return math.sub[dtype](other, self)

    fn __isub__(mut self, other: SIMD[dtype, 1]) raises:
        """
        Enables `array -= scalar`.
        """
        self = self - other

    fn __isub__(mut self, other: Self) raises:
        """
        Enables `array -= array`.
        """
        self = self - other

    fn __matmul__(self, other: Self) raises -> Self:
        return numojo.linalg.matmul(self, other)

    fn __mul__[
        OtherDType: DType,
        ResultDType: DType = TypeCoercion.result[dtype, OtherDType](),
    ](self, other: Scalar[OtherDType]) raises -> NDArray[ResultDType]:
        """
        Enables `array * scalar`.
        """
        return math.mul[ResultDType](
            self.astype[ResultDType](), other.cast[ResultDType]()
        )

    fn __mul__[
        OtherDType: DType,
        ResultDType: DType = TypeCoercion.result[dtype, OtherDType](),
    ](self, other: NDArray[OtherDType]) raises -> NDArray[ResultDType]:
        """
        Enables `array * array`.
        """
        return math.mul[ResultDType](
            self.astype[ResultDType](), other.astype[ResultDType]()
        )

    fn __mul__(self, other: Scalar[dtype]) raises -> Self:
        """
        Enables `array * scalar`.
        """
        return math.mul[dtype](self, other)

    fn __mul__(self, other: Self) raises -> Self:
        """
        Enables `array * array`.
        """
        return math.mul[dtype](self, other)

    fn __rmul__[
        OtherDType: DType,
        ResultDType: DType = TypeCoercion.result[dtype, OtherDType](),
    ](self, other: Scalar[OtherDType]) raises -> NDArray[ResultDType]:
        """
        Enables `scalar * array`.
        """
        return math.mul[ResultDType](
            self.astype[ResultDType](), other.cast[ResultDType]()
        )

    fn __rmul__(mut self, other: SIMD[dtype, 1]) raises -> Self:
        """
        Enables `scalar * array`.
        """
        return math.mul[dtype](self, other)

    fn __imul__(mut self, other: SIMD[dtype, 1]) raises:
        """
        Enables `array *= scalar`.
        """
        self = self * other

    fn __imul__(mut self, other: Self) raises:
        """
        Enables `array *= array`.
        """
        self = self * other

    fn __abs__(self) -> Self:
        return abs(self)

    fn __invert__(self) raises -> Self:
        """
        Element-wise inverse (~ or not), only for bools and integral types.
        """
        return bitwise.invert[dtype](self)

    fn __pow__(self, p: Int) -> Self:
        return self._elementwise_pow(p)

    fn __pow__(self, rhs: Scalar[dtype]) raises -> Self:
        """Power of items."""
        var res = self
        for i in range(self.size):
            res._buf.ptr[i] = self._buf.ptr[i].__pow__(rhs)
        return res^

    fn __pow__(self, p: Self) raises -> Self:
        if self.size != p.size:
            raise Error(
                String(
                    "\nError in `numojo.NDArray.__pow__(self, p)`: "
                    "Both arrays must have same number of elements! "
                    "Self array has {} elements. "
                    "Other array has {} elements"
                ).format(self.size, p.size)
            )

        var result = Self(self.shape)

        @parameter
        fn vectorized_pow[simd_width: Int](index: Int) -> None:
            result._buf.ptr.store(
                index,
                self._buf.ptr.load[width=simd_width](index)
                ** p._buf.ptr.load[width=simd_width](index),
            )

        vectorize[vectorized_pow, self.width](self.size)
        return result

    fn __ipow__(mut self, p: Int):
        self = self.__pow__(p)

    fn _elementwise_pow(self, p: Int) -> Self:
        var new_vec = self

        @parameter
        fn array_scalar_vectorize[simd_width: Int](index: Int) -> None:
            new_vec._buf.ptr.store(
                index,
                builtin_math.pow(
                    self._buf.ptr.load[width=simd_width](index), p
                ),
            )

        vectorize[array_scalar_vectorize, self.width](self.size)
        return new_vec

    fn __truediv__[
        OtherDType: DType,
        ResultDType: DType = TypeCoercion.result[dtype, OtherDType](),
    ](self, other: Scalar[OtherDType]) raises -> NDArray[ResultDType]:
        """
        Enables `array / scalar`.
        """
        return math.div[ResultDType](
            self.astype[ResultDType](), other.cast[ResultDType]()
        )

    fn __truediv__[
        OtherDType: DType,
        ResultDType: DType = TypeCoercion.result[dtype, OtherDType](),
    ](self, other: NDArray[OtherDType]) raises -> NDArray[ResultDType]:
        """
        Enables `array / array`.
        """
        return math.div[ResultDType](
            self.astype[ResultDType](), other.astype[ResultDType]()
        )

    fn __truediv__(self, other: SIMD[dtype, 1]) raises -> Self:
        """
        Enables `array / scalar`.
        """
        return math.div[dtype](self, other)

    fn __truediv__(self, other: Self) raises -> Self:
        """
        Enables `array / array`.
        """
        return math.div[dtype](self, other)

    fn __itruediv__(mut self, s: SIMD[dtype, 1]) raises:
        """
        Enables `array /= scalar`.
        """
        self = self.__truediv__(s)

    fn __itruediv__(mut self, other: Self) raises:
        """
        Enables `array /= array`.
        """
        self = self.__truediv__(other)

    fn __rtruediv__[
        OtherDType: DType,
        ResultDType: DType = TypeCoercion.result[dtype, OtherDType](),
    ](self, s: Scalar[OtherDType]) raises -> NDArray[ResultDType]:
        """
        Enables `scalar / array`.
        """
        return math.div[ResultDType](
            s.cast[ResultDType](), self.astype[ResultDType]()
        )

    fn __rtruediv__(self, s: SIMD[dtype, 1]) raises -> Self:
        """
        Enables `scalar / array`.
        """
        return math.div[dtype](s, self)

    fn __floordiv__[
        OtherDType: DType,
        ResultDType: DType = TypeCoercion.result[dtype, OtherDType](),
    ](self, other: Scalar[OtherDType]) raises -> NDArray[ResultDType]:
        """
        Enables `array // scalar`.
        """
        return math.floor_div[ResultDType](
            self.astype[ResultDType](), other.cast[ResultDType]()
        )

    fn __floordiv__[
        OtherDType: DType,
        ResultDType: DType = TypeCoercion.result[dtype, OtherDType](),
    ](self, other: NDArray[OtherDType]) raises -> NDArray[ResultDType]:
        """
        Enables `array // array`.
        """
        return math.floor_div[ResultDType](
            self.astype[ResultDType](), other.astype[ResultDType]()
        )

    fn __floordiv__(self, other: SIMD[dtype, 1]) raises -> Self:
        """
        Enables `array // scalar`.
        """
        return math.floor_div[dtype](self, other)

    fn __floordiv__(self, other: Self) raises -> Self:
        """
        Enables `array // array`.
        """
        return math.floor_div[dtype](self, other)

    fn __ifloordiv__(mut self, s: SIMD[dtype, 1]) raises:
        """
        Enables `array //= scalar`.
        """
        self = self.__floordiv__(s)

    fn __ifloordiv__(mut self, other: Self) raises:
        """
        Enables `array //= array`.
        """
        self = self.__floordiv__(other)

    fn __rfloordiv__[
        OtherDType: DType,
        ResultDType: DType = TypeCoercion.result[dtype, OtherDType](),
    ](self, other: Scalar[OtherDType]) raises -> NDArray[ResultDType]:
        """
        Enables `scalar // array`.
        """
        return math.floor_div[ResultDType](
            other.cast[ResultDType](), self.astype[ResultDType]()
        )

    fn __rfloordiv__(self, other: SIMD[dtype, 1]) raises -> Self:
        """
        Enables `scalar // array`.
        """
        return math.floor_div[dtype](other, self)

    fn __mod__[
        OtherDType: DType,
        ResultDType: DType = TypeCoercion.result[dtype, OtherDType](),
    ](self, other: Scalar[OtherDType]) raises -> NDArray[ResultDType]:
        """
        Enables `array % scalar`.
        """
        return math.mod[ResultDType](
            self.astype[ResultDType](), other.cast[ResultDType]()
        )

    fn __mod__[
        OtherDType: DType,
        ResultDType: DType = TypeCoercion.result[dtype, OtherDType](),
    ](self, other: NDArray[OtherDType]) raises -> NDArray[ResultDType]:
        """
        Enables `array % array`.
        """
        return math.mod[ResultDType](
            self.astype[ResultDType](), other.astype[ResultDType]()
        )

    fn __mod__(mut self, other: SIMD[dtype, 1]) raises -> Self:
        """
        Enables `array % scalar`.
        """
        return math.mod[dtype](self, other)

    fn __mod__(mut self, other: NDArray[dtype]) raises -> Self:
        """
        Enables `array % array`.
        """
        return math.mod[dtype](self, other)

    fn __imod__(mut self, other: SIMD[dtype, 1]) raises:
        """
        Enables `array %= scalar`.
        """
        self = math.mod[dtype](self, other)

    fn __imod__(mut self, other: NDArray[dtype]) raises:
        """
        Enables `array %= array`.
        """
        self = math.mod[dtype](self, other)

    fn __rmod__(mut self, other: SIMD[dtype, 1]) raises -> Self:
        """
        Enables `scalar % array`.
        """
        return math.mod[dtype](other, self)

    fn __rmod__[
        OtherDType: DType,
        ResultDType: DType = TypeCoercion.result[dtype, OtherDType](),
    ](self, other: Scalar[OtherDType]) raises -> NDArray[ResultDType]:
        """
        Enables `scalar % array`.
        """
        return math.mod[ResultDType](
            other.cast[ResultDType](), self.astype[ResultDType]()
        )

    # ===-------------------------------------------------------------------===#
    # IO dunders and relevant methods
    # Trait implementations
    # ===-------------------------------------------------------------------===#
    fn __str__(self) -> String:
        """
        Enables String(array).

        Returns:
            A string representation of the array.
        """
        var res: String
        try:
            res = self._array_to_string(0, 0, GLOBAL_PRINT_OPTIONS)
        except e:
            res = String("Cannot convert array to string.\n") + String(e)

        return res

    fn write_to[W: Writer](self, mut writer: W):
        """
        Writes the array to a writer.

        Args:
            writer: The writer to write the array to.
        """
        if self.ndim == 0:
            # For 0-D array (numojo scalar), we can directly write the value
            writer.write(
                String(self._buf.ptr[])
                + String(
                    "  (0darray["
                    + _concise_dtype_str(self.dtype)
                    + "], use `[]` or `.item()` to unpack)"
                )
            )
        else:
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
                    + String(self.flags.C_CONTIGUOUS)
                    + "  F-cont: "
                    + String(self.flags.F_CONTIGUOUS)
                    + "  own data: "
                    + String(self.flags.OWNDATA)
                )
            except e:
                writer.write("Cannot convert array to string.\n" + String(e))

    fn __repr__(self) -> String:
        """
        Computes the "official" string representation of NDArray.
        You can construct the array using this representation.

        Returns:
            A string representation of the array.

        Examples:

        ```console
        >>>import numojo as nm
        >>>var b = nm.arange[nm.f32](20).reshape(Shape(4, 5))
        >>>print(repr(b))
        numojo.array[f32](
        '''
        [[0.0, 1.0, 2.0, 3.0, 4.0]
        [5.0, 6.0, 7.0, 8.0, 9.0]
        [10.0, 11.0, 12.0, 13.0, 14.0]
        [15.0, 16.0, 17.0, 18.0, 19.0]]
        '''
        )
        ```.
        """
        var result: String

        try:
            result = (
                String("numojo.array[")
                + _concise_dtype_str(self.dtype)
                + String('](\n"""\n')
                + self._array_to_string(0, 0, GLOBAL_PRINT_OPTIONS)
                + '\n"""\n)'
            )
        except e:
            result = "Cannot convert array to string.\n" + String(e)

        return result

    # ===-------------------------------------------------------------------===#
    # Trait dunders and iterator dunders
    # ===-------------------------------------------------------------------===#

    fn __len__(self) -> Int:
        """
        Returns length of 0-th dimension.
        """
        return self.shape._buf[0]

    fn __iter__(
        self,
    ) raises -> _NDArrayIter[__origin_of(self), dtype]:
        """
        Iterates over elements of the NDArray and return sub-arrays as view.

        Returns:
            An iterator of NDArray elements.

        Examples:

        ```
        >>> var a = nm.random.arange[nm.i8](2, 3, 4).reshape(nm.Shape(2, 3, 4))
        >>> for i in a:
        ...     print(i)
        [[      0       1       2       3       ]
        [      4       5       6       7       ]
        [      8       9       10      11      ]]
        2-D array  Shape: [3, 4]  DType: int8  C-cont: True  F-cont: False  own data: False
        [[      12      13      14      15      ]
        [      16      17      18      19      ]
        [      20      21      22      23      ]]
        2-D array  Shape: [3, 4]  DType: int8  C-cont: True  F-cont: False  own data: False
        ```.
        """

        return _NDArrayIter[__origin_of(self), dtype](
            self,
            dimension=0,
        )

    fn __reversed__(
        self,
    ) raises -> _NDArrayIter[__origin_of(self), dtype, forward=False]:
        """
        Iterates backwards over elements of the NDArray, returning
        copied value.

        Returns:
            A reversed iterator of NDArray elements.
        """

        return _NDArrayIter[__origin_of(self), dtype, forward=False](
            self,
            dimension=0,
        )

    fn _adjust_slice(self, slice_list: List[Slice]) raises -> List[Slice]:
        """
        Adjusts the slice values to lie within 0 and dim.

        Args:
            slice_list: List of slices.

        Returns:
            Adjusted list of slices.

        Raises:
            Error: If the slice step is zero.
            Error: If the slice start or end is negative.
            Error: If the slice start is greater than or equal to the slice end.
        """
        var n_slices: Int = slice_list.__len__()
        var slices = List[Slice]()
        for i in range(n_slices):
            # Get initial values with defaults
            var start = slice_list[i].start.or_else(0)
            var end = slice_list[i].end.or_else(self.shape[i])
            var step = slice_list[i].step.or_else(1)

            # Validate step
            if step == 0:
                raise Error("\nError: Slice step cannot be zero")

            # Check for negative indices
            if start < 0 or end < 0:
                raise Error(
                    String(
                        "\nError: Negative indexing not supported in"
                        " dimension {}"
                    ).format(i)
                )
                # Future implementation:
                # start = self.shape[i] + start if start < 0 else start
                # end = self.shape[i] + end if end < 0 else end

            if start >= self.shape[i]:
                raise Error(
                    String(
                        "\nError: Start index {} exceeds dimension {} size {}"
                    ).format(start, i, self.shape[i])
                )
            if end > self.shape[i]:
                raise Error(
                    String(
                        "\nError: End index {} exceeds dimension {} size {}"
                    ).format(end, i, self.shape[i])
                )
            if start >= end:
                raise Error(
                    String(
                        "\nError: Start index {} must be less than end index {}"
                        " in dimension {}"
                    ).format(start, end, i)
                )

            slices.append(
                Slice(
                    start=Optional(start),
                    end=Optional(end),
                    step=Optional(step),
                )
            )

        return slices^

    fn _array_to_string(
        self,
        dimension: Int,
        offset: Int,
        owned print_options: PrintOptions,
    ) raises -> String:
        """
        Convert the array to a string.

        Args:
            dimension: The current dimension.
            offset: The offset of the current dimension.
            print_options: The print options.

        Returns:
            String representation of the array.
        """

        if self.ndim == 0:
            # For 0-D array (numojo scalar), return the scalar value.
            return String(self._buf.ptr[0])

        var seperator = print_options.separator
        var padding = print_options.padding
        var edge_items = print_options.edge_items

        # The following code get the max value and the min value of
        # the pritable region to determine the digits before decimals and
        # the negative sign and then determine the formatted width.
        if dimension == 0:
            var negative_sign: Bool = False  # whether there should be a negative sign
            var number_of_digits: Int  # number of digits before or after decimal point
            var number_of_digits_small_values: Int  # number of digits after decimal point for small values
            var formatted_width: Int  # formatted width based on precision and digits before decimal points
            var max_value: Scalar[dtype] = abs(
                self._buf.ptr[]
            )  # maximum absolute value of the items
            var min_value: Scalar[dtype] = abs(
                self._buf.ptr[]
            )  # minimum absolute value of the items
            var indices = Item(
                ndim=self.ndim, initialized=True
            )  # Temporarily store the indices

            self._find_max_and_min_in_printable_region(
                self.shape,
                self.strides,
                edge_items,
                indices,
                negative_sign,
                max_value,
                min_value,
                0,
            )

            number_of_digits = Int(log10(Float64(max_value))) + 1
            number_of_digits_small_values = (
                abs(Int(log10(Float64(min_value)))) + 1
            )

            if dtype.is_floating_point():
                formatted_width = (
                    print_options.precision
                    + 1
                    + number_of_digits
                    + Int(negative_sign)
                )
                # If the number is not too wide,
                # or digits after decimal point is not many
                # format it as a floating point.
                if (formatted_width <= 14) and (
                    number_of_digits_small_values <= 2
                ):
                    print_options.formatted_width = formatted_width
                # Otherwise, format it as a scientific number.
                else:
                    print_options.float_format = "scientific"
                    print_options.formatted_width = 7 + print_options.precision
            else:  # type is integral
                print_options.formatted_width = number_of_digits + Int(
                    negative_sign
                )

        if dimension == self.ndim - 1:
            var result: String = String("[") + padding
            var number_of_items = self.shape[dimension]
            if number_of_items <= edge_items * 2:  # Print all items
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
            if number_of_items <= edge_items * 2:  # Print all items
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

    fn _find_max_and_min_in_printable_region(
        self,
        shape: NDArrayShape,
        strides: NDArrayStrides,
        edge_items: Int,
        mut indices: Item,
        mut negative_sign: Bool,  # whether there should be a negative sign
        mut max_value: Scalar[dtype],  # maximum absolute value of the items
        mut min_value: Scalar[dtype],  # minimum absolute value of the items
        current_axis: Int = 0,
    ) raises:
        """
        Travel through the printable region of the array to find maximum and minimum values.
        """
        var offsets = List[Int]()
        if shape[current_axis] > edge_items * 2:
            for i in range(0, edge_items):
                offsets.append(i)
                offsets.append(shape[current_axis] - 1 - i)
        else:
            for i in range(0, shape[current_axis]):
                offsets.append(i)

        for index_at_axis in offsets:
            indices._buf[current_axis] = index_at_axis[]
            if current_axis == shape.ndim - 1:
                var val = (self._buf.ptr + _get_offset(indices, strides))[]
                if val < 0:
                    negative_sign = True
                max_value = max(max_value, abs(val))
                min_value = min(min_value, abs(val))
            else:
                self._find_max_and_min_in_printable_region(
                    shape,
                    strides,
                    edge_items,
                    indices,
                    negative_sign,
                    max_value,
                    min_value,
                    current_axis + 1,
                )

    # ===-------------------------------------------------------------------===#
    # OTHER METHODS
    # (Sorted alphabetically)
    #
    # TODO: Implement axis parameter for all operations that are along an axis
    #
    # # not urgent: argpartition, byteswap, choose, conj, dump, getfield
    # # partition, put, repeat, searchsorted, setfield, squeeze, swapaxes, take,
    # # tobyets, tofile, view
    # ===-------------------------------------------------------------------===#

    fn all(self) raises -> Bool:
        """
        If all true return true.

        Returns:
            True if all elements are true, otherwise False.

        Raises:
            Error: If the array elements are not Boolean or Integer.
        """
        # make this a compile time check when they become more readable
        if not (self.dtype is DType.bool or self.dtype.is_integral()):
            raise Error(
                "\nError in `numojo.NDArray.all(self)`: "
                "Array elements must be Boolean or Integer."
            )
        # We might need to figure out how we want to handle truthyness before can do this
        var result: Bool = True

        @parameter
        fn vectorized_all[simd_width: Int](idx: Int) -> None:
            result = result and builtin_bool.all(
                (self._buf.ptr + idx).strided_load[width=simd_width](1)
            )

        vectorize[vectorized_all, self.width](self.size)
        return result

    fn any(self) raises -> Bool:
        """
        True if any true.

        Returns:
            True if any element is true, otherwise False.

        Raises:
            Error: If the array elements are not Boolean or Integer.
        """
        # make this a compile time check
        if not (self.dtype is DType.bool or self.dtype.is_integral()):
            raise Error(
                "\nError in `numojo.NDArray.any(self)`: "
                "Array elements must be Boolean or Integer."
            )
        var result: Bool = False

        @parameter
        fn vectorized_any[simd_width: Int](idx: Int) -> None:
            result = result or builtin_bool.any(
                (self._buf.ptr + idx).strided_load[width=simd_width](1)
            )

        vectorize[vectorized_any, self.width](self.size)
        return result

    fn argmax(self) raises -> Int:
        """
        Get location in pointer of max value.

        Returns:
            Index of the maximum value.
        """
        var result: Int = 0
        var max_val: SIMD[dtype, 1] = self.load[width=1](0)
        for i in range(1, self.size):
            var temp: SIMD[dtype, 1] = self.load[width=1](i)
            if temp > max_val:
                max_val = temp
                result = i
        return result

    fn argmin(self) raises -> Int:
        """
        Get location in pointer of min value.

        Returns:
            Index of the minimum value.
        """
        var result: Int = 0
        var min_val: SIMD[dtype, 1] = self.load[width=1](0)
        for i in range(1, self.size):
            var temp: SIMD[dtype, 1] = self.load[width=1](i)
            if temp < min_val:
                min_val = temp
                result = i
        return result

    fn argsort(self) raises -> NDArray[DType.index]:
        """
        Sort the NDArray and return the sorted indices.
        See `numojo.argsort()`.

        Returns:
            The indices of the sorted NDArray.
        """

        return numojo.sorting.argsort(self)

    fn argsort(self, axis: Int) raises -> NDArray[DType.index]:
        """
        Sort the NDArray and return the sorted indices.
        See `numojo.argsort()`.

        Returns:
            The indices of the sorted NDArray.
        """

        return numojo.sorting.argsort(self, axis=axis)

    fn astype[target: DType](self) raises -> NDArray[target]:
        """
        Convert type of array.

        Parameters:
            target: Target data type.

        Returns:
            NDArray with the target data type.
        """
        return creation.astype[target](self)

    fn clip(self, a_min: Scalar[dtype], a_max: Scalar[dtype]) -> Self:
        """
        Limit the values in an array between [a_min, a_max].
        If a_min is greater than a_max, the value is equal to a_max.
        See `numojo.clip()` for more details.

        Args:
            a_min: The minimum value.
            a_max: The maximum value.

        Returns:
            An array with the clipped values.
        """

        return numojo.clip(self, a_min, a_max)

    fn compress[
        dtype: DType
    ](self, condition: NDArray[DType.bool], axis: Int) raises -> Self:
        # TODO: @forFudan try using parallelization for this function
        """
        Return selected slices of an array along given axis.
        If no axis is provided, the array is flattened before use.

        Parameters:
            dtype: DType.

        Args:
            condition: 1-D array of booleans that selects which entries to return.
                If length of condition is less than the size of the array along the
                given axis, then output is filled to the length of the condition
                with False.
            axis: The axis along which to take slices.

        Returns:
            An array.

        Raises:
            Error: If the axis is out of bound for the given array.
            Error: If the condition is not 1-D array.
            Error: If the condition length is out of bound for the given axis.
            Error: If the condition contains no True values.
        """

        return numojo.compress(condition=condition, a=self, axis=axis)

    fn compress[
        dtype: DType
    ](self, condition: NDArray[DType.bool]) raises -> Self:
        """
        Return selected slices of an array along given axis.
        If no axis is provided, the array is flattened before use.
        This is a function ***OVERLOAD***.

        Parameters:
            dtype: DType.

        Args:
            condition: 1-D array of booleans that selects which entries to return.
                If length of condition is less than the size of the array along the
                given axis, then output is filled to the length of the condition
                with False.

        Returns:
            An array.

        Raises:
            Error: If the condition is not 1-D array.
            Error: If the condition length is out of bound for the given axis.
            Error: If the condition contains no True values.
        """

        return numojo.compress(condition=condition, a=self)

    # TODO: Remove this function, use slicing instead
    fn col(self, id: Int) raises -> Self:
        """Get the ith column of the matrix.

        Args:
            id: The column index.

        Returns:
            The ith column of the matrix.
        """

        if self.ndim > 2:
            raise Error(
                String(
                    "\nError in `numojo.NDArray.col(self, id)`: "
                    "The number of dimension is {}. It should be 2."
                ).format(self.ndim)
            )

        var width = self.shape[1]
        var height = self.shape[0]
        var buffer = Self(Shape(height))
        for i in range(height):
            buffer.store(i, self._buf.ptr.load[width=1](id + i * width))
        return buffer

    fn copy(self) raises -> Self:
        # TODO: Add logics for non-contiguous arrays when views are implemented.
        """
        Returns a copy of the array that owns the data.
        The returned array will be contiguous in memory.

        Returns:
            A copy of the array.
        """

        if (self.strides == NDArrayStrides(shape=self.shape)) or (
            self.strides == NDArrayStrides(shape=self.shape, order="F")
        ):
            # The strides and shape are matched.
            # It either owns the data or it is a contiguous view of another array.
            # The array is contiguous in memory. Nothing needs to be changed.
            var result = self
            return result
        else:
            # The strides and shape are not matched.
            # It is a view of another array with different shape and strides.
            if self.flags.C_CONTIGUOUS:
                # The array is C-contiguous in memory.
                # Can be copied by the last dimension.
                var result = self
                return result

            elif self.flags.F_CONTIGUOUS:
                # The array is F-contiguous in memory.
                # Can be copied by the first dimension.
                var result = self
                return result
            else:
                # The array is not contiguous in memory.
                # Can be copied by item.
                var result = self
                return result

    fn cumprod(self) raises -> NDArray[dtype]:
        """
        Returns cumprod of all items of an array.
        The array is flattened before cumprod.

        Returns:
            Cumprod of all items of an array.
        """
        return numojo.math.cumprod[dtype](self)

    fn cumprod(self, axis: Int) raises -> NDArray[dtype]:
        """
        Returns cumprod of array by axis.

        Args:
            axis: Axis.

        Returns:
            Cumprod of array by axis.
        """
        return numojo.math.cumprod[dtype](self, axis=axis)

    fn cumsum(self) raises -> NDArray[dtype]:
        """
        Returns cumsum of all items of an array.
        The array is flattened before cumsum.

        Returns:
            Cumsum of all items of an array.
        """
        return numojo.math.cumsum[dtype](self)

    fn cumsum(self, axis: Int) raises -> NDArray[dtype]:
        """
        Returns cumsum of array by axis.

        Args:
            axis: Axis.

        Returns:
            Cumsum of array by axis.
        """
        return numojo.math.cumsum[dtype](self, axis=axis)

    fn diagonal[dtype: DType](self, offset: Int = 0) raises -> Self:
        """
        Returns specific diagonals.
        Currently supports only 2D arrays.

        Raises:
            Error: If the array is not 2D.
            Error: If the offset is beyond the shape of the array.

        Parameters:
            dtype: Data type of the array.

        Args:
            offset: Offset of the diagonal from the main diagonal.

        Returns:
            The diagonal of the NDArray.
        """
        return numojo.linalg.diagonal(self, offset=offset)

    fn fill(mut self, val: Scalar[dtype]):
        """
        Fill all items of array with value.

        Args:
            val: Value to fill.
        """

        for i in range(self.size):
            self._buf.ptr[i] = val

    fn flatten(self, order: String = "C") raises -> Self:
        """
        Return a copy of the array collapsed into one dimension.

        Args:
            order: A NDArray.

        Returns:
            The 1 dimensional flattened NDArray.
        """
        return ravel(self, order=order)

    fn iter_along_axis[
        forward: Bool = True
    ](self, axis: Int, order: String = "C") raises -> _NDAxisIter[
        __origin_of(self), dtype, forward
    ]:
        """
        Returns an iterator yielding 1-d array slices along the given axis.

        Parameters:
            forward: If True, iterate from the beginning to the end.
                If False, iterate from the end to the beginning.

        Args:
            axis: The axis by which the iteration is performed.
            order: The order to traverse the array.

        Returns:
            An iterator yielding 1-d array slices along the given axis.

        Raises:
            Error: If the axis is out of bound for the given array.

        Examples:

        ```mojo
        from numojo.prelude import *
        var a = nm.arange[i8](24).reshape(Shape(2, 3, 4))
        print(a)
        for i in a.iter_along_axis(axis=0):
            print(String(i))
        ```

        This prints:

        ```console
        [[[ 0  1  2  3]
        [ 4  5  6  7]
        [ 8  9 10 11]]
        [[12 13 14 15]
        [16 17 18 19]
        [20 21 22 23]]]
        3D-array  Shape(2,3,4)  Strides(12,4,1)  DType: i8  C-cont: True  F-cont: False  own data: True
        [ 0 12]
        [ 1 13]
        [ 2 14]
        [ 3 15]
        [ 4 16]
        [ 5 17]
        [ 6 18]
        [ 7 19]
        [ 8 20]
        [ 9 21]
        [10 22]
        [11 23]
        ```

        Another example:

        ```mojo
        from numojo.prelude import *
        var a = nm.arange[i8](24).reshape(Shape(2, 3, 4))
        print(a)
        for i in a.iter_along_axis(axis=2):
            print(String(i))
        ```

        This prints:

        ```console
        [[[ 0  1  2  3]
        [ 4  5  6  7]
        [ 8  9 10 11]]
        [[12 13 14 15]
        [16 17 18 19]
        [20 21 22 23]]]
        3D-array  Shape(2,3,4)  Strides(12,4,1)  DType: i8  C-cont: True  F-cont: False  own data: True
        [0 1 2 3]
        [4 5 6 7]
        [ 8  9 10 11]
        [12 13 14 15]
        [16 17 18 19]
        [20 21 22 23]
        ```.
        """

        var normalized_axis: Int = axis
        if normalized_axis < 0:
            normalized_axis += self.ndim
        if (normalized_axis >= self.ndim) or (normalized_axis < 0):
            raise Error(
                String(
                    "\nError in `numojo.NDArray.iter_along_axis()`: "
                    "Axis ({}) is not in valid range [{}, {})."
                ).format(axis, -self.ndim, self.ndim)
            )

        return _NDAxisIter[__origin_of(self), dtype, forward](
            self,
            axis=normalized_axis,
            order=order,
        )

    fn iter_over_dimension[
        forward: Bool = True
    ](read self, dimension: Int) raises -> _NDArrayIter[
        __origin_of(self), dtype, forward
    ]:
        """
        Returns an iterator yielding `ndim-1` arrays over the given dimension.

        Parameters:
            forward: If True, iterate from the beginning to the end.
                If False, iterate from the end to the beginning.

        Args:
            dimension: The dimension by which the iteration is performed.

        Returns:
            An iterator yielding `ndim-1` arrays over the given dimension.

        Raises:
            Error: If the axis is out of bound for the given array.
        """

        var normalized_dim: Int = dimension
        if normalized_dim < 0:
            normalized_dim += self.ndim
        if (normalized_dim >= self.ndim) or (normalized_dim < 0):
            raise Error(
                String(
                    "\nError in `numojo.NDArray.iter_over_dimension()`: "
                    "Axis ({}) is not in valid range [{}, {})."
                ).format(dimension, -self.ndim, self.ndim)
            )

        return _NDArrayIter[__origin_of(self), dtype, forward](
            a=self,
            dimension=normalized_dim,
        )

    fn max(self) raises -> Scalar[dtype]:
        """
        Finds the max value of an array.
        When no axis is given, the array is flattened before sorting.

        Returns:
            The max value.
        """

        return numojo.math.max(self)

    fn max(self, axis: Int) raises -> Self:
        """
        Finds the max value of an array along the axis.
        The number of dimension will be reduced by 1.
        When no axis is given, the array is flattened before sorting.

        Args:
            axis: The axis along which the max is performed.

        Returns:
            An array with reduced number of dimensions.
        """

        return numojo.math.max(self, axis=axis)

    # TODO: Remove this methods
    fn mdot(self, other: Self) raises -> Self:
        """
        Dot product of two matrix.
        Matrix A: M * N.
        Matrix B: N * L.

        Args:
            other: The other matrix.

        Returns:
            The dot product of the two matrices.

        Raises:
            Error: If the arrays are not matrices.
            Error: If the second dimension of the self array does not match the
                first dimension of the other array.
        """

        if (self.ndim != 2) or (other.ndim != 2):
            raise Error(
                String(
                    "\nError in `numojo.NDArray.mdot(self, other)`: "
                    "The array should have only two dimensions (matrix).\n"
                    "The self array has {} dimensions.\n"
                    "The orther array has {} dimensions"
                ).format(self.ndim, other.ndim)
            )

        if self.shape[1] != other.shape[0]:
            raise Error(
                String(
                    "\nError in `numojo.NDArray.mdot(self, other)`: "
                    "Second dimension of A does not match first dimension of"
                    " B.\nA is {}x{}. \nB is {}x{}."
                ).format(
                    self.shape[0], self.shape[1], other.shape[0], other.shape[1]
                )
            )

        var new_matrix = Self(Shape(self.shape[0], other.shape[1]))
        for row in range(self.shape[0]):
            for col in range(other.shape[1]):
                new_matrix.__setitem__(
                    Item(row, col),
                    self[row : row + 1, :].vdot(other[:, col : col + 1]),
                )
        return new_matrix

    fn mean[
        returned_dtype: DType = DType.float64
    ](self) raises -> Scalar[returned_dtype]:
        """
        Mean of a array.

        Returns:
            The mean of the array.
        """
        return numojo.statistics.mean[returned_dtype](self)

    fn mean[
        returned_dtype: DType = DType.float64
    ](self: Self, axis: Int) raises -> NDArray[returned_dtype]:
        """
        Mean of array elements over a given axis.

        Args:
            axis: The axis along which the mean is performed.

        Returns:
            An NDArray.

        """
        return numojo.statistics.mean[returned_dtype](self, axis)

    fn median[
        returned_dtype: DType = DType.float64
    ](self) raises -> Scalar[returned_dtype]:
        """
        Median of a array.

        Returns:
            The median of the array.
        """
        return median[returned_dtype](self)

    fn median[
        returned_dtype: DType = DType.float64
    ](self: Self, axis: Int) raises -> NDArray[returned_dtype]:
        """
        Median of array elements over a given axis.

        Args:
            axis: The axis along which the median is performed.

        Returns:
            An NDArray.

        """
        return median[returned_dtype](self, axis)

    fn min(self) raises -> Scalar[dtype]:
        """
        Finds the min value of an array.
        When no axis is given, the array is flattened before sorting.

        Returns:
            The min value.
        """

        return numojo.math.min(self)

    fn min(self, axis: Int) raises -> Self:
        """
        Finds the min value of an array along the axis.
        The number of dimension will be reduced by 1.
        When no axis is given, the array is flattened before sorting.

        Args:
            axis: The axis along which the min is performed.

        Returns:
            An array with reduced number of dimensions.
        """

        return numojo.math.min(self, axis=axis)

    fn nditer(self) raises -> _NDIter[__origin_of(self), dtype]:
        """
        ***Overload*** Return an iterator yielding the array elements according
        to the memory layout of the array.

        Returns:
            An iterator yielding the array elements.

        Examples:

        ```console
        >>>var a = nm.random.rand[i8](2, 3, min=0, max=100)
        >>>print(a)
        [[      37      8       25      ]
        [      25      2       57      ]]
        2-D array  (2,3)  DType: int8  C-cont: True  F-cont: False  own data: True
        >>>for i in a.nditer():
        ...    print(i, end=" ")
        37 8 25 25 2 57
        ```.
        """

        var order: String

        if self.flags.F_CONTIGUOUS:
            order = "F"
        else:
            order = "C"

        return self.nditer(order=order)

    fn nditer(self, order: String) raises -> _NDIter[__origin_of(self), dtype]:
        """
        Return an iterator yielding the array elements according to the order.

        Args:
            order: Order of the array.

        Returns:
            An iterator yielding the array elements.

        Examples:

        ```console
        >>>var a = nm.random.rand[i8](2, 3, min=0, max=100)
        >>>print(a)
        [[      37      8       25      ]
        [      25      2       57      ]]
        2-D array  (2,3)  DType: int8  C-cont: True  F-cont: False  own data: True
        >>>for i in a.nditer():
        ...    print(i, end=" ")
        37 8 25 25 2 57
        ```.
        """

        if order not in List[String]("C", "F"):
            raise Error(
                String(
                    "\nError in `nditer()`: Invalid order: '{}'. "
                    "The order should be 'C' or 'F'."
                ).format(order)
            )

        var axis: Int

        if order == "C":
            axis = self.ndim - 1
        else:
            axis = 0

        return _NDIter[__origin_of(self), dtype](a=self, order=order, axis=axis)

    fn num_elements(self) -> Int:
        """
        Function to retreive size (compatability).

        Returns:
            The size of the array.
        """
        return self.size

    fn prod(self: Self) raises -> Scalar[dtype]:
        """
        Product of all array elements.

        Returns:
            Scalar.
        """
        return numojo.math.prod(self)

    fn prod(self: Self, axis: Int) raises -> Self:
        """
        Product of array elements over a given axis.

        Args:
            axis: The axis along which the product is performed.

        Returns:
            An NDArray.
        """

        return numojo.math.prod(self, axis=axis)

    # TODO: Remove this methods
    fn rdot(self, other: Self) raises -> Self:
        """
        Dot product of two matrix.
        Matrix A: M * N.
        Matrix B: N * L.

        Args:
            other: The other matrix.

        Returns:
            The dot product of the two matrices.

        Raises:
            Error: If the arrays are not matrices.
            Error: If the second dimension of the self array does not match the
                first dimension of the other array.
        """

        if (self.ndim != 2) or (other.ndim != 2):
            raise Error(
                String(
                    "\nError in `numojo.NDArray.rdot(self, other)`: "
                    "The array should have only two dimensions (matrix)."
                    "The self array is of {} dimensions.\n"
                    "The other array is of {} dimensions."
                ).format(self.ndim, other.ndim)
            )
        if self.shape[1] != other.shape[0]:
            raise Error(
                String(
                    "\nError in `numojo.NDArray.rdot(self, other)`: "
                    "Second dimension of A ({}) \n"
                    "does not match first dimension of B ({})."
                ).format(self.shape[1], other.shape[0])
            )

        var new_matrix = Self(Shape(self.shape[0], other.shape[1]))
        for row in range(self.shape[0]):
            for col in range(other.shape[1]):
                new_matrix.store(
                    col + row * other.shape[1],
                    self.row(row).vdot(other.col(col)),
                )
        return new_matrix

    fn reshape(self, shape: NDArrayShape, order: String = "C") raises -> Self:
        """
        Returns an array of the same data with a new shape.

        Args:
            shape: Shape of returned array.
            order: Order of the array - Row major `C` or Column major `F`.

        Returns:
            Array of the same data with a new shape.
        """
        return numojo.reshape(self, shape=shape, order=order)

    fn resize(mut self, shape: NDArrayShape) raises:
        """
        In-place change shape and size of array.

        Notes:
        To returns a new array, use `reshape`.

        Args:
            shape: Shape after resize.
        """

        var order = "C" if self.flags.C_CONTIGUOUS else "F"

        if shape.size_of_array() > self.size:
            var other = Self(shape=shape, order=order)
            memcpy(other._buf.ptr, self._buf.ptr, self.size)
            for i in range(self.size, other.size):
                (other._buf.ptr + i).init_pointee_copy(0)
            self = other^
        else:
            self.shape = shape
            self.ndim = shape.ndim
            self.size = shape.size_of_array()
            self.strides = NDArrayStrides(shape, order=order)

    fn round(self) raises -> Self:
        """
        Rounds the elements of the array to a whole number.

        Returns:
            An NDArray.
        """
        return rounding.tround[dtype](self)

    fn row(self, id: Int) raises -> Self:
        """Get the ith row of the matrix.

        Args:
            id: The row index.

        Returns:
            The ith row of the matrix.

        Raises:
            Error: If the ndim is greater than 2.
        """

        if self.ndim > 2:
            raise Error(
                String(
                    "\nError in `numojo.NDArray.row(self, id)`: "
                    "The number of dimension is {}.\nIt should be 2."
                ).format(self.ndim)
            )

        var width = self.shape[1]
        var buffer = Self(Shape(width))
        for i in range(width):
            buffer.store(i, self._buf.ptr.load[width=1](i + id * width))
        return buffer

    fn sort(mut self, axis: Int = -1) raises:
        """
        Sorts the array in-place along the given axis using quick sort method.
        The deault axis is -1.
        See `numojo.sorting.sort` for more information.

        Args:
            axis: The axis along which the array is sorted. Defaults to -1.

        Raises:
            Error: If the axis is out of bound for the given array.
        """

        var normalized_axis: Int = axis
        if normalized_axis < 0:
            normalized_axis += self.ndim
        if (normalized_axis >= self.ndim) or (normalized_axis < 0):
            raise Error(
                String(
                    "\nError in `numojo.NDArray.sort()`: "
                    "Axis ({}) is not in valid range [-{}, {})."
                ).format(axis, self.ndim, self.ndim)
            )

        numojo.sorting._sort_inplace(self, axis=normalized_axis)

    fn std[
        returned_dtype: DType = DType.float64
    ](self, ddof: Int = 0) raises -> Scalar[returned_dtype]:
        """
        Compute the standard deviation.
        See `numojo.std`.

        Parameters:
            returned_dtype: The returned data type, defaulting to float64.

        Args:
            ddof: Delta degree of freedom.
        """

        return std[returned_dtype](self, ddof=ddof)

    fn std[
        returned_dtype: DType = DType.float64
    ](self, axis: Int, ddof: Int = 0) raises -> NDArray[returned_dtype]:
        """
        Compute the standard deviation along the axis.
        See `numojo.std`.

        Parameters:
            returned_dtype: The returned data type, defaulting to float64.

        Args:
            axis: The axis along which the mean is performed.
            ddof: Delta degree of freedom.
        """

        return std[returned_dtype](self, axis=axis, ddof=ddof)

    fn sum(self: Self) raises -> Scalar[dtype]:
        """
        Returns sum of all array elements.

        Returns:
            Scalar.
        """
        return sum(self)

    fn sum(self: Self, axis: Int) raises -> Self:
        """
        Sum of array elements over a given axis.

        Args:
            axis: The axis along which the sum is performed.

        Returns:
            An NDArray.
        """
        return sum(self, axis=axis)

    fn T(self, axes: List[Int]) raises -> Self:
        """
        Transpose array of any number of dimensions according to
        arbitrary permutation of the axes.

        If `axes` is not given, it is equal to flipping the axes.

        Args:
            axes: List of axes.

        Returns:
            Transposed array.

        Defined in `numojo.routines.manipulation.transpose`.
        """
        return numojo.routines.manipulation.transpose(self, axes)

    fn T(self) raises -> Self:
        """
        ***Overload*** Transposes the array when `axes` is not given.
        If `axes` is not given, it is equal to flipping the axes.
        See docstring of `transpose`.

        Returns:
            Transposed array.

        Defined in `numojo.routines.manipulation.transpose`.
        """
        return numojo.routines.manipulation.transpose(self)

    fn tolist(self) -> List[Scalar[dtype]]:
        """
        Converts NDArray to a 1-D List.

        Returns:
            A 1-D List.
        """
        var result: List[Scalar[dtype]] = List[Scalar[dtype]]()
        for i in range(self.size):
            result.append(self._buf.ptr[i])
        return result

    fn to_numpy(self) raises -> PythonObject:
        """
        Convert to a numpy array.

        Returns:
            A numpy array.
        """
        return to_numpy(self)

    fn to_tensor(self) raises -> Tensor[dtype]:
        """
        Convert array to tensor of the same dtype.

        Returns:
            A tensor of the same dtype.

        Examples:

        ```mojo
        import numojo as nm
        from numojo.prelude import *

        fn main() raises:
            var a = nm.random.randn[f16](2, 3, 4)
            print(a)
            print(a.to_tensor())

            var b = nm.array[i8]("[[1, 2, 3], [4, 5, 6]]")
            print(b)
            print(b.to_tensor())

            var c = nm.array[boolean]("[[1,0], [0,1]]")
            print(c)
            print(c.to_tensor())
        ```
        .
        """

        return to_tensor(self)

    # TODO: add axis parameter
    fn trace(
        self, offset: Int = 0, axis1: Int = 0, axis2: Int = 1
    ) raises -> NDArray[dtype]:
        """
        Computes the trace of a ndarray.

        Args:
            offset: Offset of the diagonal from the main diagonal.
            axis1: First axis.
            axis2: Second axis.

        Returns:
            The trace of the ndarray.
        """
        return numojo.linalg.trace[dtype](self, offset, axis1, axis2)

    # TODO: Remove the underscore in the method name when view is supported.
    fn _transpose(self) raises -> Self:
        """
        Returns a view of transposed array.

        It is unsafe!

        Returns:
            A view of transposed array.
        """
        return Self(
            shape=self.shape._flip(),
            buffer=self._buf.ptr,
            offset=0,
            strides=self.strides._flip(),
        )

    fn unsafe_ptr(self) -> UnsafePointer[Scalar[dtype]]:
        """
        Retreive pointer without taking ownership.

        Returns:
            Unsafe pointer to the data buffer.

        """
        return self._buf.ptr

    fn variance[
        returned_dtype: DType = DType.float64
    ](self, ddof: Int = 0) raises -> Scalar[returned_dtype]:
        """
        Returns the variance of array.

        Parameters:
            returned_dtype: The returned data type, defaulting to float64.

        Args:
            ddof: Delta degree of freedom.

        Returns:
            The variance of the array.
        """
        return variance[returned_dtype](self, ddof=ddof)

    fn variance[
        returned_dtype: DType = DType.float64
    ](self, axis: Int, ddof: Int = 0) raises -> NDArray[returned_dtype]:
        """
        Returns the variance of array along the axis.
        See `numojo.variance`.

        Parameters:
            returned_dtype: The returned data type, defaulting to float64.

        Args:
            axis: The axis along which the mean is performed.
            ddof: Delta degree of freedom.

        Returns:
            The variance of the array along the axis.
        """
        return variance[returned_dtype](self, axis=axis, ddof=ddof)

    # TODO: Remove this methods, but add it into routines.
    fn vdot(self, other: Self) raises -> SIMD[dtype, 1]:
        """
        Inner product of two vectors.

        Args:
            other: The other vector.

        Returns:
            The inner product of the two vectors.
        """
        if self.size != other.size:
            raise Error("The lengths of two vectors do not match.")

        var sum = Scalar[dtype](0)
        for i in range(self.size):
            sum = sum + self.load(i) * other.load(i)
        return sum


# ===----------------------------------------------------------------------===#
# NDArrayIterator
# ===----------------------------------------------------------------------===#


@value
struct _NDArrayIter[
    is_mutable: Bool, //,
    origin: Origin[is_mutable],
    dtype: DType,
    forward: Bool = True,
]:
    # TODO:
    # Return a view instead of copy where possible
    # (when Bufferable is supported).
    """
    An iterator yielding `ndim-1` array slices over the given dimension.
    It is the default iterator of the `NDArray.__iter__() method and for loops.
    It can also be constructed using the `NDArray.iter_over_dimension()` method.
    It trys to create a view where possible.

    Parameters:
        is_mutable: Whether the iterator is mutable.
        origin: The lifetime of the underlying NDArray data.
        dtype: The data type of the item.
        forward: The iteration direction. `False` is backwards.
    """

    var index: Int
    var ptr: UnsafePointer[Scalar[dtype]]
    var dimension: Int
    var length: Int
    var shape: NDArrayShape
    var strides: NDArrayStrides
    """Strides of array or view. It is not necessarily compatible with shape."""
    var ndim: Int
    var size_of_item: Int

    fn __init__(out self, read a: NDArray[dtype], read dimension: Int) raises:
        """
        Initialize the iterator.

        Args:
            a: The array
            dimension: Dimension to iterate over.
        """

        if dimension < 0 or dimension >= a.ndim:
            raise Error("Axis must be in the range of [0, ndim).")

        self.ptr = a._buf.ptr
        self.dimension = dimension
        self.shape = a.shape
        self.strides = a.strides
        self.ndim = a.ndim
        self.length = a.shape[dimension]
        self.size_of_item = a.size // a.shape[dimension]
        # Status of the iterator
        self.index = 0 if forward else a.shape[dimension] - 1

    fn __iter__(self) -> Self:
        return self

    fn __next__(mut self) raises -> NDArray[dtype]:
        var res = NDArray[dtype](self.shape._pop(self.dimension))
        var current_index = self.index

        @parameter
        if forward:
            self.index += 1
        else:
            self.index -= 1

        for offset in range(self.size_of_item):
            var remainder = offset
            var item = Item(ndim=self.ndim, initialized=False)

            for i in range(self.ndim - 1, -1, -1):
                if i != self.dimension:
                    (item._buf + i).init_pointee_copy(remainder % self.shape[i])
                    remainder = remainder // self.shape[i]
                else:
                    (item._buf + self.dimension).init_pointee_copy(
                        current_index
                    )

            (res._buf.ptr + offset).init_pointee_copy(
                self.ptr[_get_offset(item, self.strides)]
            )
        return res

    @always_inline
    fn __has_next__(self) -> Bool:
        @parameter
        if forward:
            return self.index < self.length
        else:
            return self.index >= 0

    fn __len__(self) -> Int:
        @parameter
        if forward:
            return self.length - self.index
        else:
            return self.index

    fn ith(self, index: Int) raises -> NDArray[dtype]:
        """
        Gets the i-th array of the iterator.

        Args:
            index: The index of the item. It must be non-negative.

        Returns:
            The i-th `ndim-1`-D array of the iterator.
        """

        if (index >= self.length) or (index < 0):
            raise Error(
                String(
                    "\nError in `NDArrayIter.ith()`: "
                    "Index ({}) must be in the range of [0, {})"
                ).format(index, self.length)
            )

        if self.ndim > 1:
            var res = NDArray[dtype](self.shape._pop(self.dimension))

            for offset in range(self.size_of_item):
                var remainder = offset
                var item = Item(ndim=self.ndim, initialized=False)

                for i in range(self.ndim - 1, -1, -1):
                    if i != self.dimension:
                        (item._buf + i).init_pointee_copy(
                            remainder % self.shape[i]
                        )
                        remainder = remainder // self.shape[i]
                    else:
                        (item._buf + self.dimension).init_pointee_copy(index)

                (res._buf.ptr + offset).init_pointee_copy(
                    self.ptr[_get_offset(item, self.strides)]
                )
            return res

        else:  # 0-D array
            var res = numojo.creation._0darray[dtype](self.ptr[index])
            return res


@value
struct _NDAxisIter[
    is_mutable: Bool, //,
    origin: Origin[is_mutable],
    dtype: DType,
    forward: Bool = True,
]():
    # TODO:
    # Return a view instead of copy where possible
    # (when Bufferable is supported).
    """
    An iterator yielding 1-d array slices along the given axis.
    The yielded array slices are garanteed to be contiguous on memory.
    It trys to create a view where possible.
    It can be constructed by `NDArray.iter_along_axis()` method.
    The iterator is useful when applying functions along a certain axis.

    Parameters:
        is_mutable: Whether the iterator is mutable.
        origin: The lifetime of the underlying NDArray data.
        dtype: The data type of the item.
        forward: The iteration direction. `False` is backwards.

    Examples:

    ```
    [[[ 0,  1,  2,  3],
    [ 4,  5,  6,  7],
    [ 8,  9, 10, 11]],
    [[12, 13, 14, 15],
    [16, 17, 18, 19],
    [20, 21, 22, 23]]]
    ```
    The above array is of shape (2,3,3). Itering by `axis=0` returns:
    ```
    [0, 12], [1, 13], [2, 14], [3, 15],
    [4, 16], [5, 17], [6, 18], [7, 19],
    [8, 20], [9, 21], [10, 22], [11, 23]
    ```
    Itering by `axis=1` returns:
    ```
    [0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7, 11],
    [12, 16, 20], [13, 17, 21], [14, 18, 22], [15, 19, 23]
    ```
    """

    var ptr: UnsafePointer[Scalar[dtype]]
    var axis: Int
    var order: String
    var length: Int
    var size: Int
    var ndim: Int
    var shape: NDArrayShape
    var strides: NDArrayStrides
    """Strides of array or view. It is not necessarily compatible with shape."""
    var strides_compatible: NDArrayStrides
    """Strides according to shape of view and along the axis."""
    var index: Int
    """Status counter."""
    var size_of_item: Int
    """Size of the result 1-d array."""

    fn __init__(
        out self,
        read a: NDArray[dtype],
        axis: Int,
        order: String,
    ) raises:
        """
        Initialize the iterator.

        Args:
            a: the array.
            axis: Axis.
            order: Order to traverse the array.
        """
        if axis < 0 or axis >= a.ndim:
            raise Error("Axis must be in the range of [0, ndim).")

        self.size = a.size
        self.size_of_item = a.shape[axis]
        self.ptr = a._buf.ptr
        self.axis = axis
        self.order = order
        self.length = self.size // self.size_of_item
        self.ndim = a.ndim
        self.shape = a.shape
        self.strides = a.strides
        # Construct the compatible strides
        self.strides_compatible = NDArrayStrides(
            ndim=self.ndim, initialized=False
        )
        (self.strides_compatible._buf + axis).init_pointee_copy(1)
        temp = a.shape[axis]
        if order == "C":
            for i in range(self.ndim - 1, -1, -1):
                if i != axis:
                    (self.strides_compatible._buf + i).init_pointee_copy(temp)
                    temp *= a.shape[i]
        else:
            for i in range(self.ndim):
                if i != axis:
                    (self.strides_compatible._buf + i).init_pointee_copy(temp)
                    temp *= a.shape[i]

        # Status of the iterator
        self.index = 0 if forward else self.length - 1

    fn __has_next__(self) -> Bool:
        @parameter
        if forward:
            return self.index < self.length
        else:
            return self.index >= 0

    fn __iter__(self) -> Self:
        return self

    fn __len__(self) -> Int:
        @parameter
        if forward:
            return self.length - self.index
        else:
            return self.index

    fn __next__(mut self) raises -> NDArray[dtype]:
        var res = NDArray[dtype](Shape(self.size_of_item))
        var current_index = self.index

        @parameter
        if forward:
            self.index += 1
        else:
            self.index -= 1

        var remainder = current_index * self.size_of_item
        var item = Item(ndim=self.ndim, initialized=False)

        if self.order == "C":
            for i in range(self.ndim):
                if i != self.axis:
                    (item._buf + i).init_pointee_copy(
                        remainder // self.strides_compatible[i]
                    )
                    remainder %= self.strides_compatible[i]
                else:
                    (item._buf + i).init_pointee_copy(0)
        else:
            for i in range(self.ndim - 1, -1, -1):
                if i != self.axis:
                    (item._buf + i).init_pointee_copy(
                        remainder // self.strides_compatible[i]
                    )
                    remainder %= self.strides_compatible[i]
                else:
                    (item._buf + i).init_pointee_copy(0)

        if ((self.axis == self.ndim - 1) or (self.axis == 0)) & (
            (self.shape[self.axis] == 1) or (self.strides[self.axis] == 1)
        ):
            # The memory layout is C-contiguous or F-contiguous
            memcpy(
                res._buf.ptr,
                self.ptr + _get_offset(item, self.strides),
                self.size_of_item,
            )

        else:
            for j in range(self.size_of_item):
                (res._buf.ptr + j).init_pointee_copy(
                    self.ptr[_get_offset(item, self.strides)]
                )
                item._buf[self.axis] += 1

        return res^

    fn ith(self, index: Int) raises -> NDArray[dtype]:
        """
        Gets the i-th 1-d array of the iterator.

        Args:
            index: The index of the item. It must be non-negative.

        Returns:
            The i-th 1-d array of the iterator.
        """

        if (index >= self.length) or (index < 0):
            raise Error(
                String(
                    "\nError in `NDAxisIter.ith()`: "
                    "Index ({}) must be in the range of [0, {})"
                ).format(index, self.length)
            )

        var elements = NDArray[dtype](Shape(self.size_of_item))

        var remainder = index * self.size_of_item
        var item = Item(ndim=self.ndim, initialized=True)

        if self.order == "C":
            for i in range(self.ndim):
                if i != self.axis:
                    (item._buf + i).init_pointee_copy(
                        remainder // self.strides_compatible[i]
                    )
                    remainder %= self.strides_compatible[i]
                else:
                    (item._buf + i).init_pointee_copy(0)
        else:
            for i in range(self.ndim - 1, -1, -1):
                if i != self.axis:
                    (item._buf + i).init_pointee_copy(
                        remainder // self.strides_compatible[i]
                    )
                    remainder %= self.strides_compatible[i]
                else:
                    (item._buf + i).init_pointee_copy(0)

        if ((self.axis == self.ndim - 1) or (self.axis == 0)) & (
            (self.shape[self.axis] == 1) or (self.strides[self.axis] == 1)
        ):
            # The memory layout is C-contiguous or F-contiguous
            memcpy(
                elements._buf.ptr,
                self.ptr + _get_offset(item, self.strides),
                self.size_of_item,
            )
        else:
            for j in range(self.size_of_item):
                (elements._buf.ptr + j).init_pointee_copy(
                    self.ptr[_get_offset(item, self.strides)]
                )
                item._buf[self.axis] += 1

        return elements

    fn ith_with_offsets(
        self, index: Int
    ) raises -> Tuple[NDArray[DType.index], NDArray[dtype]]:
        """
        Gets the i-th 1-d array of the iterator and the offsets (in C-order)
        of its elements.

        Args:
            index: The index of the item. It must be non-negative.

        Returns:
            Offsets (in C-order) and elements of the i-th 1-d array of the
            iterator.
        """
        var offsets = NDArray[DType.index](Shape(self.size_of_item))
        var elements = NDArray[dtype](Shape(self.size_of_item))

        if (index >= self.length) or (index < 0):
            raise Error(
                String(
                    "\nError in `NDAxisIter.ith_with_offsets()`: "
                    "Index ({}) must be in the range of [0, {})"
                ).format(index, self.length)
            )

        var remainder = index * self.size_of_item
        var item = Item(ndim=self.ndim, initialized=True)
        for i in range(self.axis):
            item._buf[i] = remainder // self.strides_compatible[i]
            remainder %= self.strides_compatible[i]
        for i in range(self.axis + 1, self.ndim):
            item._buf[i] = remainder // self.strides_compatible[i]
            remainder %= self.strides_compatible[i]

        var new_strides = NDArrayStrides(self.shape, order="C")

        if (self.axis == self.ndim - 1) & (
            (self.shape[self.axis] == 1) or (self.strides[self.axis] == 1)
        ):
            # The memory layout is C-contiguous
            memcpy(
                elements._buf.ptr,
                self.ptr + _get_offset(item, self.strides),
                self.size_of_item,
            )
            var begin_offset = _get_offset(item, new_strides)
            for j in range(self.size_of_item):
                (offsets._buf.ptr + j).init_pointee_copy(begin_offset + j)

        elif (self.axis == 0) & (
            (self.shape[self.axis] == 1) or (self.strides[self.axis] == 1)
        ):
            # The memory layout is F-contiguous
            memcpy(
                elements._buf.ptr,
                self.ptr + _get_offset(item, self.strides),
                self.size_of_item,
            )
            for j in range(self.size_of_item):
                (offsets._buf.ptr + j).init_pointee_copy(
                    _get_offset(item, new_strides)
                )
                item._buf[self.axis] += 1

        else:
            for j in range(self.size_of_item):
                (offsets._buf.ptr + j).init_pointee_copy(
                    _get_offset(item, new_strides)
                )
                (elements._buf.ptr + j).init_pointee_copy(
                    self.ptr[_get_offset(item, self.strides)]
                )
                item._buf[self.axis] += 1

        return Tuple(offsets, elements)


@value
struct _NDIter[
    is_mutable: Bool, //, origin: Origin[is_mutable], dtype: DType
]():
    """
    An iterator yielding the array elements according to the order.
    It can be constructed by `NDArray.nditer()` method.
    """

    var ptr: UnsafePointer[Scalar[dtype]]
    var length: Int
    var ndim: Int
    var shape: NDArrayShape
    var strides: NDArrayStrides
    var strides_compatible: NDArrayStrides
    var index: Int
    var axis: Int
    """Axis along which the iterator travels."""
    var order: String
    """Order to traverse the array."""

    fn __init__(out self, a: NDArray[dtype], order: String, axis: Int) raises:
        self.length = a.size
        self.order = order
        self.axis = axis
        self.ptr = a._buf.ptr
        self.ndim = a.ndim
        self.shape = a.shape
        self.strides = a.strides
        # Construct the compatible strides
        self.strides_compatible = NDArrayStrides(
            ndim=self.ndim, initialized=False
        )
        (self.strides_compatible._buf + axis).init_pointee_copy(1)
        temp = a.shape[axis]
        if order == "C":
            for i in range(self.ndim - 1, -1, -1):
                if i != axis:
                    (self.strides_compatible._buf + i).init_pointee_copy(temp)
                    temp *= a.shape[i]
        else:
            for i in range(self.ndim):
                if i != axis:
                    (self.strides_compatible._buf + i).init_pointee_copy(temp)
                    temp *= a.shape[i]

        self.index = 0

    fn __iter__(self) -> Self:
        return self

    fn __has_next__(self) -> Bool:
        if self.index < self.length:
            return True
        else:
            return False

    fn __next__(mut self) raises -> Scalar[dtype]:
        var current_index = self.index
        self.index += 1

        var remainder = current_index
        var indices = Item(ndim=self.ndim, initialized=False)

        if self.order == "C":
            for i in range(self.ndim):
                if i != self.axis:
                    (indices._buf + i).init_pointee_copy(
                        remainder // self.strides_compatible._buf[i]
                    )
                    remainder %= self.strides_compatible._buf[i]
            (indices._buf + self.axis).init_pointee_copy(remainder)

        else:
            for i in range(self.ndim - 1, -1, -1):
                if i != self.axis:
                    (indices._buf + i).init_pointee_copy(
                        remainder // self.strides_compatible._buf[i]
                    )
                    remainder %= self.strides_compatible._buf[i]
            (indices._buf + self.axis).init_pointee_copy(remainder)

        return self.ptr[_get_offset(indices, self.strides)]

    fn ith(self, index: Int) raises -> Scalar[dtype]:
        """
        Gets the i-th element of the iterator.

        Args:
            index: The index of the item. It must be non-negative.

        Returns:
            The i-th element of the iterator.
        """

        if (index >= self.length) or (index < 0):
            raise Error(
                String(
                    "\nError in `NDIter.ith()`: "
                    "Index ({}) must be in the range of [0, {})"
                ).format(index, self.length)
            )

        var remainder = index
        var indices = Item(ndim=self.ndim, initialized=False)

        if self.order == "C":
            for i in range(self.ndim):
                if i != self.axis:
                    (indices._buf + i).init_pointee_copy(
                        remainder // self.strides_compatible._buf[i]
                    )
                    remainder %= self.strides_compatible._buf[i]
            (indices._buf + self.axis).init_pointee_copy(remainder)
        else:
            for i in range(self.ndim - 1, -1, -1):
                if i != self.axis:
                    (indices._buf + i).init_pointee_copy(
                        remainder // self.strides_compatible._buf[i]
                    )
                    remainder %= self.strides_compatible._buf[i]
            (indices._buf + self.axis).init_pointee_copy(remainder)

        return self.ptr[_get_offset(indices, self.strides)]
