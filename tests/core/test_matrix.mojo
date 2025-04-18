import numojo as nm
from numojo.prelude import *
from numojo.core.matrix import Matrix
from python import Python, PythonObject
from testing.testing import assert_raises, assert_true

# ===-----------------------------------------------------------------------===#
# Main functions
# ===-----------------------------------------------------------------------===#


fn check_matrices_equal[
    dtype: DType
](matrix: Matrix[dtype], np_sol: PythonObject, st: String) raises:
    var np = Python.import_module("numpy")
    assert_true(np.all(np.equal(np.matrix(matrix.to_numpy()), np_sol)), st)


fn check_matrices_close[
    dtype: DType
](matrix: Matrix[dtype], np_sol: PythonObject, st: String) raises:
    var np = Python.import_module("numpy")
    assert_true(
        np.all(np.isclose(np.matrix(matrix.to_numpy()), np_sol, atol=0.01)), st
    )


fn check_values_close[
    dtype: DType
](value: Scalar[dtype], np_sol: PythonObject, st: String) raises:
    var np = Python.import_module("numpy")
    assert_true(np.isclose(value, np_sol, atol=0.01), st)


# ===-----------------------------------------------------------------------===#
# Manipulation
# ===-----------------------------------------------------------------------===#

var memory_layouts = List[String]("C", "F")


def test_manipulation():
    for i in range(2):
        var np = Python.import_module("numpy")
        var A = Matrix.rand[f64]((10, 10), order=memory_layouts[i]) * 1000
        var Anp = np.matrix(A.to_numpy())
        check_matrices_equal(
            A.astype[nm.i32](),
            Anp.astype(np.int32),
            "`astype` is broken for memory layout " + memory_layouts[i],
        )

        check_matrices_equal(
            A.reshape((50, 2)),
            Anp.reshape((50, 2)),
            "Reshape is broken for memory layout " + memory_layouts[i],
        )

        _ = A.resize((1000, 100))
        _ = Anp.resize((1000, 100))
        check_matrices_equal(
            A,
            Anp,
            "Resize is broke nfor memory layout " + memory_layouts[i],
        )


# ===-----------------------------------------------------------------------===#
# Creation
# ===-----------------------------------------------------------------------===#


def test_full():
    for i in range(2):
        var np = Python.import_module("numpy")
        check_matrices_equal(
            Matrix.full[f64]((10, 10), 10, order=memory_layouts[i]),
            np.full((10, 10), 10, dtype=np.float64),
            "Full is broken for memory layout " + memory_layouts[i],
        )


def test_zeros():
    for i in range(2):
        var np = Python.import_module("numpy")
        check_matrices_equal(
            Matrix.zeros[f64](shape=(10, 10), order=memory_layouts[i]),
            np.zeros((10, 10), dtype=np.float64),
            "Zeros is broken for memory layout " + memory_layouts[i],
        )


# ===-----------------------------------------------------------------------===#
# Arithmetic
# ===-----------------------------------------------------------------------===#


def test_arithmetic():
    for i in range(2):
        var np = Python.import_module("numpy")
        var A = Matrix.rand[f64]((10, 10), order=memory_layouts[i])
        var B = Matrix.rand[f64]((10, 10), order=memory_layouts[i])
        var C = Matrix.rand[f64]((10, 1), order=memory_layouts[i])
        var Ap = A.to_numpy()
        var Bp = B.to_numpy()
        var Cp = C.to_numpy()
        check_matrices_close(
            A + B,
            Ap + Bp,
            "Add is broken for memory layout " + memory_layouts[i],
        )
        check_matrices_close(
            A - B,
            Ap - Bp,
            "Sub is broken for memory layout " + memory_layouts[i],
        )
        check_matrices_close(
            A * B,
            Ap * Bp,
            "Mul is broken for memory layout " + memory_layouts[i],
        )
        check_matrices_close(
            A @ B,
            np.matmul(Ap, Bp),
            "Matmul is broken for memory layout " + memory_layouts[i],
        )
        check_matrices_close(
            A + C,
            Ap + Cp,
            "Add (broadcast) is broken for memory layout " + memory_layouts[i],
        )
        check_matrices_close(
            A - C,
            Ap - Cp,
            "Sub (broadcast) is broken for memory layout " + memory_layouts[i],
        )
        check_matrices_close(
            A * C,
            Ap * Cp,
            "Mul (broadcast) is broken for memory layout " + memory_layouts[i],
        )
        check_matrices_close(
            A / C,
            Ap / Cp,
            "Div (broadcast) is broken for memory layout " + memory_layouts[i],
        )
        check_matrices_close(
            A + 1,
            Ap + 1,
            "Add (to int) is broken for memory layout " + memory_layouts[i],
        )
        check_matrices_close(
            A - 1,
            Ap - 1,
            "Sub (to int) is broken for memory layout " + memory_layouts[i],
        )
        check_matrices_close(
            A * 1,
            Ap * 1,
            "Mul (to int) is broken for memory layout " + memory_layouts[i],
        )
        check_matrices_close(
            A / 1,
            Ap / 1,
            "Div (to int) is broken for memory layout " + memory_layouts[i],
        )
        check_matrices_close(
            A**2,
            np.power(Ap, 2),
            "Pow (to int) is broken for memory layout " + memory_layouts[i],
        )
        check_matrices_close(
            A**0.5,
            np.power(Ap, 0.5),
            "Pow (to float) is broken for memory layout " + memory_layouts[i],
        )


def test_logic():
    for i in range(2):
        var np = Python.import_module("numpy")
        var A = Matrix.ones((5, 1), order=memory_layouts[i])
        var B = Matrix.ones((5, 1), order=memory_layouts[i])
        var L = Matrix.fromstring[i8](
            "[[0,0,0],[0,0,1],[1,1,1],[1,0,0]]",
            shape=(4, 3),
            order=memory_layouts[i],
        )
        var Anp = np.matrix(A.to_numpy())
        var Bnp = np.matrix(B.to_numpy())
        var Lnp = np.matrix(L.to_numpy())

        check_matrices_equal(
            A > B,
            Anp > Bnp,
            "gt is broken for memory layout " + memory_layouts[i],
        )
        check_matrices_equal(
            A < B,
            Anp < Bnp,
            "lt is broken for memory layout " + memory_layouts[i],
        )
        assert_true(
            np.equal(nm.all(L), np.all(Lnp)),
            "`all` is broken for memory layout " + memory_layouts[i],
        )
        for j in range(2):
            check_matrices_close(
                Matrix.all(L, axis=j),
                np.all(Lnp, axis=j),
                String(
                    "`all` by axis {} is broken for memory layout {}"
                ).format(j, memory_layouts[i]),
            )
        assert_true(
            np.equal(Matrix.any(L), np.any(Lnp)),
            "`any` is broken for memory layout " + memory_layouts[i],
        )
        for j in range(2):
            check_matrices_close(
                Matrix.any(L, axis=j),
                np.any(Lnp, axis=j),
                String(
                    "`any` by axis {} is broken for memory layout {}"
                ).format(j, memory_layouts[i]),
            )


# ===-----------------------------------------------------------------------===#
# Linear algebra
# ===-----------------------------------------------------------------------===#


def test_linalg():
    for i in range(2):
        var np = Python.import_module("numpy")
        var A = Matrix.rand[f64]((100, 100), order=memory_layouts[i])
        var B = Matrix.rand[f64]((100, 100), order=memory_layouts[i])
        var E = Matrix.fromstring(
            "[[1,2,3],[4,5,6],[7,8,9],[10,11,12]]",
            shape=(4, 3),
            order=memory_layouts[i],
        )
        var Y = Matrix.rand((100, 1), order=memory_layouts[i])
        var Anp = A.to_numpy()
        var Bnp = B.to_numpy()
        var Ynp = Y.to_numpy()
        var Enp = E.to_numpy()

        check_matrices_close(
            nm.linalg.solve(A, B),
            np.linalg.solve(Anp, Bnp),
            "Solve is broken for memory layout " + memory_layouts[i],
        )
        check_matrices_close(
            nm.linalg.inv(A),
            np.linalg.inv(Anp),
            "Inverse is broken for memory layout " + memory_layouts[i],
        )
        check_matrices_close(
            nm.linalg.lstsq(A, Y),
            np.linalg.lstsq(Anp, Ynp)[0],
            "Least square is broken for memory layout " + memory_layouts[i],
        )
        check_matrices_close(
            A.transpose(),
            Anp.transpose(),
            "Transpose is broken for memory layout " + memory_layouts[i],
        )
        check_matrices_close(
            Y.transpose(),
            Ynp.transpose(),
            "Transpose is broken for memory layout " + memory_layouts[i],
        )
        assert_true(
            np.all(np.isclose(nm.linalg.det(A), np.linalg.det(Anp), atol=0.1)),
            "Determinant is broken for memory layout " + memory_layouts[i],
        )
        for j in range(-10, 10):
            assert_true(
                np.all(
                    np.isclose(
                        nm.linalg.trace(E, offset=j),
                        np.trace(Enp, offset=j),
                        atol=0.1,
                    )
                ),
                "Trace is broken for memory layout " + memory_layouts[i],
            )


def test_qr_decomposition():
    for i in range(2):
        var A = Matrix.rand[f64]((20, 20), order=memory_layouts[i])
        var np = Python.import_module("numpy")

        Q, R = nm.linalg.qr(A)

        # Check if Q^T Q is close to the identity matrix, i.e Q is orthonormal
        var id = Q.transpose() @ Q
        assert_true(
            np.allclose(id.to_numpy(), np.eye(Q.shape[0]), atol=1e-14),
            "Q not orthonormal for memory layout " + memory_layouts[i],
        )

        # Check if R is upper triangular
        assert_true(
            np.allclose(R.to_numpy(), np.triu(R.to_numpy()), atol=1e-14),
            "R not upper triangular for memory layout " + memory_layouts[i],
        )

        # Check if A = QR
        var A_test = Q @ R
        assert_true(
            np.allclose(A_test.to_numpy(), A.to_numpy(), atol=1e-14),
            "QR decomposition incorrect for memory layout " + memory_layouts[i],
        )


def test_qr_decomposition_asym_reduced():
    for i in range(2):
        var np = Python.import_module("numpy")
        var A = Matrix.rand[f64]((12, 5), order=memory_layouts[i])
        Q, R = nm.linalg.qr(A, mode="reduced")

        assert_true(
            Q.shape[0] == 12 and Q.shape[1] == 5,
            "Q has unexpected shape for reduced with order "
            + memory_layouts[i],
        )
        assert_true(
            R.shape[0] == 5 and R.shape[1] == 5,
            "R has unexpected shape for reduced with order "
            + memory_layouts[i],
        )

        var id = Q.transpose() @ Q
        assert_true(
            np.allclose(id.to_numpy(), np.eye(Q.shape[1]), atol=1e-14),
            "Q not orthonormal for reduced with order " + memory_layouts[i],
        )
        assert_true(
            np.allclose(R.to_numpy(), np.triu(R.to_numpy()), atol=1e-14),
            "R not upper triangular for reduced with order "
            + memory_layouts[i],
        )

        var A_test = Q @ R
        assert_true(
            np.allclose(A_test.to_numpy(), A.to_numpy(), atol=1e-14),
            "QR reduced decomposition incorrect for memory layout "
            + memory_layouts[i],
        )


def test_qr_decomposition_asym_complete():
    for i in range(2):
        var np = Python.import_module("numpy")
        var A = Matrix.rand[f64]((12, 5), order=memory_layouts[i])
        Q, R = nm.linalg.qr(A, mode="complete")

        assert_true(
            Q.shape[0] == 12 and Q.shape[1] == 12,
            "Q has unexpected shape for complete with order "
            + memory_layouts[i],
        )
        assert_true(
            R.shape[0] == 12 and R.shape[1] == 5,
            "R has unexpected shape for complete with order "
            + memory_layouts[i],
        )

        var id = Q.transpose() @ Q
        assert_true(
            np.allclose(id.to_numpy(), np.eye(Q.shape[0]), atol=1e-14),
            "Q not orthonormal for complete with order " + memory_layouts[i],
        )
        assert_true(
            np.allclose(R.to_numpy(), np.triu(R.to_numpy()), atol=1e-14),
            "R not upper triangular for complete with order "
            + memory_layouts[i],
        )

        var A_test = Q @ R
        assert_true(
            np.allclose(A_test.to_numpy(), A.to_numpy(), atol=1e-14),
            "QR complete decomposition incorrect for memory layout "
            + memory_layouts[i],
        )


def test_qr_decomposition_asym_complete2():
    for i in range(2):
        var np = Python.import_module("numpy")
        var A = Matrix.rand[f64]((5, 12), order=memory_layouts[i])
        Q, R = nm.linalg.qr(A, mode="complete")

        assert_true(
            Q.shape[0] == 5 and Q.shape[1] == 5,
            "Q has unexpected shape for complete with order "
            + memory_layouts[i],
        )
        assert_true(
            R.shape[0] == 5 and R.shape[1] == 12,
            "R has unexpected shape for complete with order "
            + memory_layouts[i],
        )

        var id = Q.transpose() @ Q
        assert_true(
            np.allclose(id.to_numpy(), np.eye(Q.shape[0]), atol=1e-14),
            "Q not orthonormal for complete with order " + memory_layouts[i],
        )
        assert_true(
            np.allclose(R.to_numpy(), np.triu(R.to_numpy()), atol=1e-14),
            "R not upper triangular for complete with order "
            + memory_layouts[i],
        )

        var A_test = Q @ R
        assert_true(
            np.allclose(A_test.to_numpy(), A.to_numpy(), atol=1e-14),
            "QR complete2 decomposition incorrect for memory layout "
            + memory_layouts[i],
        )


# ===-----------------------------------------------------------------------===#
# Mathematics
# ===-----------------------------------------------------------------------===#


def test_math():
    for i in range(2):
        var np = Python.import_module("numpy")
        var A = Matrix.rand[f64]((100, 100), order=memory_layouts[i])
        var Anp = np.matrix(A.to_numpy())

        assert_true(
            np.all(np.isclose(nm.sum(A), np.sum(Anp), atol=0.1)),
            "`sum` is broken for memory layout " + memory_layouts[i],
        )
        for j in range(2):
            check_matrices_close(
                nm.sum(A, axis=j),
                np.sum(Anp, axis=j),
                String(
                    "`sum` by axis {} is broken for memory layout {}"
                ).format(j, memory_layouts[i]),
            )

        assert_true(
            np.all(np.isclose(nm.prod(A), np.prod(Anp), atol=0.1)),
            "`prod` is broken for memory layout " + memory_layouts[i],
        )
        for j in range(2):
            check_matrices_close(
                nm.prod(A, axis=j),
                np.prod(Anp, axis=j),
                String(
                    "`prod` by axis {} is broken for memory layout {}"
                ).format(j, memory_layouts[i]),
            )

        check_matrices_close(
            nm.cumsum(A),
            np.cumsum(Anp),
            "`cumsum` is broken for memory layout " + memory_layouts[i],
        )
        for j in range(2):
            check_matrices_close(
                nm.cumsum(A, axis=j),
                np.cumsum(Anp, axis=j),
                String(
                    "`cumsum` by axis {} is broken for memory layout {}"
                ).format(j, memory_layouts[i]),
            )

        check_matrices_close(
            nm.cumprod(A),
            np.cumprod(Anp),
            "`cumprod` is broken for memory layout " + memory_layouts[i],
        )
        for j in range(2):
            check_matrices_close(
                nm.cumprod(A, axis=j),
                np.cumprod(Anp, axis=j),
                String(
                    "`cumprod` by axis {} is broken for memory layout {}"
                ).format(j, memory_layouts[i]),
            )


def test_trigonometric():
    for i in range(2):
        var np = Python.import_module("numpy")
        var A = Matrix.rand[f64]((100, 100), order=memory_layouts[i])
        var Anp = np.matrix(A.to_numpy())
        check_matrices_close(
            nm.sin(A),
            np.sin(Anp),
            "sin is broken for memory layout " + memory_layouts[i],
        )
        check_matrices_close(
            nm.cos(A),
            np.cos(Anp),
            "cos is broken for memory layout " + memory_layouts[i],
        )
        check_matrices_close(
            nm.tan(A),
            np.tan(Anp),
            "tan is broken for memory layout " + memory_layouts[i],
        )
        check_matrices_close(
            nm.arcsin(A),
            np.arcsin(Anp),
            "arcsin is broken for memory layout " + memory_layouts[i],
        )
        check_matrices_close(
            nm.asin(A),
            np.arcsin(Anp),
            "asin is broken for memory layout " + memory_layouts[i],
        )
        check_matrices_close(
            nm.arccos(A),
            np.arccos(Anp),
            "arccos is broken for memory layout " + memory_layouts[i],
        )
        check_matrices_close(
            nm.acos(A),
            np.arccos(Anp),
            "acos is broken for memory layout " + memory_layouts[i],
        )
        check_matrices_close(
            nm.arctan(A),
            np.arctan(Anp),
            "arctan is broken for memory layout " + memory_layouts[i],
        )
        check_matrices_close(
            nm.atan(A),
            np.arctan(Anp),
            "atan is broken for memory layout " + memory_layouts[i],
        )


def test_hyperbolic():
    for i in range(2):
        var np = Python.import_module("numpy")
        var A = Matrix.fromstring(
            "[[1,2,3],[4,5,6],[7,8,9]]", shape=(3, 3), order=memory_layouts[i]
        )
        var B = A / 10
        var Anp = np.matrix(A.to_numpy())
        var Bnp = np.matrix(B.to_numpy())
        check_matrices_close(
            nm.sinh(A),
            np.sinh(Anp),
            "sinh is broken for memory layout " + memory_layouts[i],
        )
        check_matrices_close(
            nm.cosh(A),
            np.cosh(Anp),
            "cosh is broken for memory layout " + memory_layouts[i],
        )
        check_matrices_close(
            nm.tanh(A),
            np.tanh(Anp),
            "tanh is broken for memory layout " + memory_layouts[i],
        )
        check_matrices_close(
            nm.arcsinh(A),
            np.arcsinh(Anp),
            "arcsinh is broken for memory layout " + memory_layouts[i],
        )
        check_matrices_close(
            nm.asinh(A),
            np.arcsinh(Anp),
            "asinh is broken for memory layout " + memory_layouts[i],
        )
        check_matrices_close(
            nm.arccosh(A),
            np.arccosh(Anp),
            "arccosh is broken for memory layout " + memory_layouts[i],
        )
        check_matrices_close(
            nm.acosh(A),
            np.arccosh(Anp),
            "acosh is broken for memory layout " + memory_layouts[i],
        )
        check_matrices_close(
            nm.arctanh(B),
            np.arctanh(Bnp),
            "arctanh is broken for memory layout " + memory_layouts[i],
        )
        check_matrices_close(
            nm.atanh(B),
            np.arctanh(Bnp),
            "atanh is broken for memory layout " + memory_layouts[i],
        )


def test_sorting():
    for i in range(2):
        var np = Python.import_module("numpy")
        var A = Matrix.rand[f64]((10, 10), order=memory_layouts[i])
        var Anp = np.matrix(A.to_numpy())

        check_matrices_close(
            nm.sort(A),
            np.sort(Anp, axis=None),
            "Sort is broken for memory layout " + memory_layouts[i],
        )
        for j in range(2):
            check_matrices_close(
                nm.sort(A, axis=j),
                np.sort(Anp, axis=j),
                String("Sort by axis {} is broken for memory layout {}").format(
                    j, memory_layouts[i]
                ),
            )

        check_matrices_close(
            nm.argsort(A),
            np.argsort(Anp, axis=None),
            "Argsort is broken for memory layout " + memory_layouts[i],
        )
        for j in range(2):
            check_matrices_close(
                nm.argsort(A, axis=j),
                np.argsort(Anp, axis=j),
                String(
                    "Argsort by axis {} is broken for memory layout {}"
                ).format(j, memory_layouts[i]),
            )


def test_searching():
    for i in range(2):
        var np = Python.import_module("numpy")
        var A = Matrix.rand[f64]((10, 10), order=memory_layouts[i])
        var Anp = np.matrix(A.to_numpy())

        check_values_close(
            nm.max(A),
            np.max(Anp, axis=None),
            "`max` is broken for memory layout " + memory_layouts[i],
        )
        for j in range(2):
            check_matrices_close(
                nm.max(A, axis=j),
                np.max(Anp, axis=j),
                String(
                    "`max` by axis {} is broken for memory layout {}"
                ).format(j, memory_layouts[i]),
            )

        check_values_close(
            nm.argmax(A),
            np.argmax(Anp, axis=None),
            "`argmax` is broken for memory layout " + memory_layouts[i],
        )
        for j in range(2):
            check_matrices_close(
                nm.argmax(A, axis=j),
                np.argmax(Anp, axis=j),
                String(
                    "`argmax` by axis {} is broken for memory layout {}"
                ).format(j, memory_layouts[i]),
            )

        check_values_close(
            nm.min(A),
            np.min(Anp, axis=None),
            "`min` is broken for memory layout " + memory_layouts[i],
        )
        for j in range(2):
            check_matrices_close(
                nm.min(A, axis=j),
                np.min(Anp, axis=j),
                String(
                    "`min` by axis {} is broken for memory layout {}"
                ).format(j, memory_layouts[i]),
            )

        check_values_close(
            nm.argmin(A),
            np.argmin(Anp, axis=None),
            "`argmin` is broken for memory layout " + memory_layouts[i],
        )
        for j in range(2):
            check_matrices_close(
                nm.argmin(A, axis=j),
                np.argmin(Anp, axis=j),
                String(
                    "`argmin` by axis {} is broken for memory layout {}"
                ).format(j, memory_layouts[i]),
            )
