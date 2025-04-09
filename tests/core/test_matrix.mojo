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


def test_manipulation():
    var np = Python.import_module("numpy")
    var A = Matrix.rand[f64]((10, 10)) * 1000
    var Anp = np.matrix(A.to_numpy())
    check_matrices_equal(
        A.astype[nm.i32](),
        Anp.astype(np.int32),
        "`astype` is broken",
    )

    check_matrices_equal(
        A.reshape((50, 2)),
        Anp.reshape((50, 2)),
        "Reshape is broken",
    )

    _ = A.resize((1000, 100))
    _ = Anp.resize((1000, 100))
    check_matrices_equal(
        A,
        Anp,
        "Resize is broken",
    )


# ===-----------------------------------------------------------------------===#
# Creation
# ===-----------------------------------------------------------------------===#


def test_full():
    var np = Python.import_module("numpy")
    check_matrices_equal(
        Matrix.full[f64]((10, 10), 10),
        np.full((10, 10), 10, dtype=np.float64),
        "Full is broken",
    )


def test_zeros():
    var np = Python.import_module("numpy")
    check_matrices_equal(
        Matrix.zeros[f64](shape=(10, 10)),
        np.zeros((10, 10), dtype=np.float64),
        "Zeros is broken",
    )


# ===-----------------------------------------------------------------------===#
# Arithmetic
# ===-----------------------------------------------------------------------===#


def test_arithmetic():
    var np = Python.import_module("numpy")
    var A = Matrix.rand[f64]((10, 10))
    var B = Matrix.rand[f64]((10, 10))
    var C = Matrix.rand[f64]((10, 1))
    var Ap = A.to_numpy()
    var Bp = B.to_numpy()
    var Cp = C.to_numpy()
    check_matrices_close(A + B, Ap + Bp, "Add is broken")
    check_matrices_close(A - B, Ap - Bp, "Sub is broken")
    check_matrices_close(A * B, Ap * Bp, "Mul is broken")
    check_matrices_close(A @ B, np.matmul(Ap, Bp), "Matmul is broken")
    check_matrices_close(A + C, Ap + Cp, "Add (broadcast) is broken")
    check_matrices_close(A - C, Ap - Cp, "Sub (broadcast) is broken")
    check_matrices_close(A * C, Ap * Cp, "Mul (broadcast) is broken")
    check_matrices_close(A / C, Ap / Cp, "Div (broadcast) is broken")
    check_matrices_close(A + 1, Ap + 1, "Add (to int) is broken")
    check_matrices_close(A - 1, Ap - 1, "Sub (to int) is broken")
    check_matrices_close(A * 1, Ap * 1, "Mul (to int) is broken")
    check_matrices_close(A / 1, Ap / 1, "Div (to int) is broken")
    check_matrices_close(A**2, np.power(Ap, 2), "Pow (to int) is broken")
    check_matrices_close(A**0.5, np.power(Ap, 0.5), "Pow (to int) is broken")


def test_logic():
    var np = Python.import_module("numpy")
    var A = Matrix.ones((5, 1))
    var B = Matrix.ones((5, 1))
    var L = Matrix.fromstring[i8](
        "[[0,0,0],[0,0,1],[1,1,1],[1,0,0]]", shape=(4, 3)
    )
    var Anp = np.matrix(A.to_numpy())
    var Bnp = np.matrix(B.to_numpy())
    var Lnp = np.matrix(L.to_numpy())

    check_matrices_equal(A > B, Anp > Bnp, "gt is broken")
    check_matrices_equal(A < B, Anp < Bnp, "lt is broken")
    assert_true(
        np.equal(nm.all(L), np.all(Lnp)),
        "`all` is broken",
    )
    for i in range(2):
        check_matrices_close(
            Matrix.all(L, axis=i),
            np.all(Lnp, axis=i),
            String("`all` by axis {i} is broken"),
        )
    assert_true(
        np.equal(Matrix.any(L), np.any(Lnp)),
        "`any` is broken",
    )
    for i in range(2):
        check_matrices_close(
            Matrix.any(L, axis=i),
            np.any(Lnp, axis=i),
            String("`any` by axis {i} is broken"),
        )


# ===-----------------------------------------------------------------------===#
# Linear algebra
# ===-----------------------------------------------------------------------===#


def test_linalg():
    var np = Python.import_module("numpy")
    var A = Matrix.rand[f64]((100, 100))
    var B = Matrix.rand[f64]((100, 100))
    var E = Matrix.fromstring(
        "[[1,2,3],[4,5,6],[7,8,9],[10,11,12]]", shape=(4, 3)
    )
    var Y = Matrix.rand((100, 1))
    var Anp = A.to_numpy()
    var Bnp = B.to_numpy()
    var Ynp = Y.to_numpy()
    var Enp = E.to_numpy()
    check_matrices_close(
        nm.linalg.solve(A, B),
        np.linalg.solve(Anp, Bnp),
        "Solve is broken",
    )
    check_matrices_close(
        nm.linalg.inv(A),
        np.linalg.inv(Anp),
        "Inverse is broken",
    )
    check_matrices_close(
        nm.linalg.lstsq(A, Y),
        np.linalg.lstsq(Anp, Ynp)[0],
        "Least square is broken",
    )
    check_matrices_close(
        A.transpose(),
        Anp.transpose(),
        "Transpose is broken",
    )
    check_matrices_close(
        Y.transpose(),
        Ynp.transpose(),
        "Transpose is broken",
    )
    assert_true(
        np.all(np.isclose(nm.linalg.det(A), np.linalg.det(Anp), atol=0.1)),
        "Determinant is broken",
    )
    for i in range(-10, 10):
        assert_true(
            np.all(
                np.isclose(
                    nm.linalg.trace(E, offset=i),
                    np.trace(Enp, offset=i),
                    atol=0.1,
                )
            ),
            "Trace is broken",
        )


def test_qr_decomposition_non_quadratic():
    A = Matrix.rand[f64]((4, 13))

    var np = Python.import_module("numpy")

    Q, R = nm.linalg.qr(A)

    # Check if Q^T Q is close to the identity matrix, i.e Q is orthonormal
    var id = Q.transpose() @ Q
    assert_true(np.allclose(id.to_numpy(), np.eye(Q.shape[0]), atol=1e-14))

    # Check if R is upper triangular
    assert_true(np.allclose(R.to_numpy(), np.triu(R.to_numpy()), atol=1e-14))

    # Check if A = QR
    var A_test = Q @ R
    assert_true(np.allclose(A_test.to_numpy(), A.to_numpy(), atol=1e-14))


def test_qr_decomposition_quadratic():
    A = Matrix.rand[f64]((15, 15))

    var np = Python.import_module("numpy")

    Q, R = nm.linalg.qr(A)

    # Check if Q^T Q is close to the identity matrix, i.e Q is orthonormal
    var id = Q.transpose() @ Q
    assert_true(np.allclose(id.to_numpy(), np.eye(Q.shape[0]), atol=1e-14))

    # Check if R is upper triangular
    assert_true(np.allclose(R.to_numpy(), np.triu(R.to_numpy()), atol=1e-14))

    # Check if A = QR
    var A_test = Q @ R
    assert_true(np.allclose(A_test.to_numpy(), A.to_numpy(), atol=1e-14))


def test_qr_decomposition_quadratic_large():
    A = Matrix.fromstring(
        (
            "[[1, 1, 1, 1, 1, 1.000000, 1.000000, 1.000000, 1.000000, 1, 1, 1,"
            " 0.841471, 0.540302, 1, 1, 1, 0.909297, -0.416147, -0.989992],[1,"
            " 2, 4, 8, 16, 0.500000, 1.500000, 0.800000, -0.900000, -1, -2, -4,"
            " 0.909297, -0.416147, 2, 2, 19, -0.756802, -0.653644,"
            " 0.960170],[1, 3, 9, 27, 81, 0.333333, 2.250000, 0.640000,"
            " 0.810000, 1, 3, 9, 0.141120, -0.989992, 3, 3, 171, -0.279415,"
            " 0.960170, -0.911130],[1, 4, 16, 64, 256, 0.250000, 3.375000,"
            " 0.512000, -0.729000, -1, -4, -16, -0.756802, -0.653644, 1, 4,"
            " 969, 0.989358, -0.145500, 0.843854],[1, 5, 25, 125, 625,"
            " 0.200000, 5.062500, 0.409600, 0.656100, 1, 5, 25, -0.958924,"
            " 0.283662, 2, 1, 3876, -0.544021, -0.839072, -0.759688],[1, 6, 36,"
            " 216, 1296, 0.166667, 7.593750, 0.327680, -0.590490, -1, -6, -36,"
            " -0.279415, 0.960170, 3, 2, 11628, -0.536573, 0.843854,"
            " 0.660317],[1, 7, 49, 343, 2401, 0.142857, 11.390625, 0.262144,"
            " 0.531441, 1, 7, 49, 0.656987, 0.753902, 1, 3, 27132, 0.990607,"
            " 0.136737, -0.547729],[1, 8, 64, 512, 4096, 0.125000, 17.085938,"
            " 0.209715, -0.478297, -1, -8, -64, 0.989358, -0.145500, 2, 4,"
            " 50388, -0.287903, -0.957659, 0.424179],[1, 9, 81, 729, 6561,"
            " 0.111111, 25.628906, 0.167772, 0.430467, 1, 9, 81, 0.412118,"
            " -0.911130, 3, 1, 75582, -0.750987, 0.660317, -0.292139],[1, 10,"
            " 100, 1000, 10000, 0.100000, 38.443359, 0.134218, -0.387420, -1,"
            " -10, -100, -0.544021, -0.839072, 1, 2, 92378, 0.912945, 0.408082,"
            " 0.154251],[1, 11, 121, 1331, 14641, 0.090909, 57.665039,"
            " 0.107374, 0.348678, 1, 11, 121, -0.999990, 0.004426, 2, 3, 92378,"
            " -0.008851, -0.999961, -0.013277],[1, 12, 144, 1728, 20736,"
            " 0.083333, 86.497559, 0.085899, -0.313811, -1, -12, -144,"
            " -0.536573, 0.843854, 3, 4, 75582, -0.905578, 0.424179,"
            " -0.127964],[1, 13, 169, 2197, 28561, 0.076923, 129.746338,"
            " 0.068719, 0.282430, 1, 13, 169, 0.420167, 0.907447, 1, 1, 50388,"
            " 0.762558, 0.646919, 0.266643],[1, 14, 196, 2744, 38416, 0.071429,"
            " 194.619507, 0.054976, -0.254187, -1, -14, -196, 0.990607,"
            " 0.136737, 2, 2, 27132, 0.270906, -0.962606, -0.399985],[1, 15,"
            " 225, 3375, 50625, 0.066667, 291.929260, 0.043980, 0.228768, 1,"
            " 15, 225, 0.650288, -0.759688, 3, 3, 11628, -0.988032, 0.154251,"
            " 0.525322],[1, 16, 256, 4096, 65536, 0.062500, 437.893890,"
            " 0.035184, -0.205891, -1, -16, -256, -0.287903, -0.957659, 1, 4,"
            " 3876, 0.551427, 0.834223, -0.640144],[1, 17, 289, 4913, 83521,"
            " 0.058824, 656.840836, 0.028147, 0.185302, 1, 17, 289, -0.961397,"
            " -0.275163, 2, 1, 969, 0.529083, -0.848570, 0.742154],[1, 18, 324,"
            " 5832, 104976, 0.055556, 985.261253, 0.022518, -0.166772, -1, -18,"
            " -324, -0.750987, 0.660317, 3, 2, 171, -0.991779, -0.127964,"
            " -0.829310],[1, 19, 361, 6859, 130321, 0.052632, 1477.891880,"
            " 0.018014, 0.150095, 1, 19, 361, 0.149877, 0.988705, 1, 3, 19,"
            " 0.296369, 0.955074, 0.899867],[1, 20, 400, 8000, 160000,"
            " 0.050000, 2216.837820, 0.014412, -0.135085, -1, -20, -400,"
            " 0.912945, 0.408082, 2, 4, 1, 0.745113, -0.666938, -0.952413]]"
        ),
        shape=(20, 20),
        c_contigous=True,
    )

    var np = Python.import_module("numpy")

    Q, R = nm.linalg.qr(A)

    # Check if Q^T Q is close to the identity matrix, i.e Q is orthonormal
    var id = Q.transpose() @ Q
    assert_true(np.allclose(id.to_numpy(), np.eye(Q.shape[0]), atol=1e-09))

    # Check if R is upper triangular
    assert_true(np.allclose(R.to_numpy(), np.triu(R.to_numpy()), atol=1e-09))

    # Check if A = QR
    var A_test = Q @ R
    assert_true(np.allclose(A_test.to_numpy(), A.to_numpy(), atol=1e-09))


# ===-----------------------------------------------------------------------===#
# Mathematics
# ===-----------------------------------------------------------------------===#


def test_math():
    var np = Python.import_module("numpy")
    var A = Matrix.rand[f64]((100, 100))
    var Anp = np.matrix(A.to_numpy())

    assert_true(
        np.all(np.isclose(nm.sum(A), np.sum(Anp), atol=0.1)),
        "`sum` is broken",
    )
    for i in range(2):
        check_matrices_close(
            nm.sum(A, axis=i),
            np.sum(Anp, axis=i),
            String("`sum` by axis {i} is broken"),
        )

    assert_true(
        np.all(np.isclose(nm.prod(A), np.prod(Anp), atol=0.1)),
        "`prod` is broken",
    )
    for i in range(2):
        check_matrices_close(
            nm.prod(A, axis=i),
            np.prod(Anp, axis=i),
            String("`prod` by axis {i} is broken"),
        )

    check_matrices_close(
        nm.cumsum(A),
        np.cumsum(Anp),
        "`cumsum` is broken",
    )
    for i in range(2):
        check_matrices_close(
            nm.cumsum(A, axis=i),
            np.cumsum(Anp, axis=i),
            String("`cumsum` by axis {i} is broken"),
        )

    check_matrices_close(
        nm.cumprod(A),
        np.cumprod(Anp),
        "`cumprod` is broken",
    )
    for i in range(2):
        check_matrices_close(
            nm.cumprod(A, axis=i),
            np.cumprod(Anp, axis=i),
            String("`cumprod` by axis {i} is broken"),
        )


def test_trigonometric():
    var np = Python.import_module("numpy")
    var A = Matrix.rand[f64]((100, 100))
    var Anp = np.matrix(A.to_numpy())
    check_matrices_close(nm.sin(A), np.sin(Anp), "sin is broken")
    check_matrices_close(nm.cos(A), np.cos(Anp), "cos is broken")
    check_matrices_close(nm.tan(A), np.tan(Anp), "tan is broken")
    check_matrices_close(nm.arcsin(A), np.arcsin(Anp), "arcsin is broken")
    check_matrices_close(nm.asin(A), np.arcsin(Anp), "asin is broken")
    check_matrices_close(nm.arccos(A), np.arccos(Anp), "arccos is broken")
    check_matrices_close(nm.acos(A), np.arccos(Anp), "acos is broken")
    check_matrices_close(nm.arctan(A), np.arctan(Anp), "arctan is broken")
    check_matrices_close(nm.atan(A), np.arctan(Anp), "atan is broken")


def test_hyperbolic():
    var np = Python.import_module("numpy")
    var A = Matrix.fromstring("[[1,2,3],[4,5,6],[7,8,9]]", shape=(3, 3))
    var B = A / 10
    var Anp = np.matrix(A.to_numpy())
    var Bnp = np.matrix(B.to_numpy())
    check_matrices_close(nm.sinh(A), np.sinh(Anp), "sinh is broken")
    check_matrices_close(nm.cosh(A), np.cosh(Anp), "cosh is broken")
    check_matrices_close(nm.tanh(A), np.tanh(Anp), "tanh is broken")
    check_matrices_close(nm.arcsinh(A), np.arcsinh(Anp), "arcsinh is broken")
    check_matrices_close(nm.asinh(A), np.arcsinh(Anp), "asinh is broken")
    check_matrices_close(nm.arccosh(A), np.arccosh(Anp), "arccosh is broken")
    check_matrices_close(nm.acosh(A), np.arccosh(Anp), "acosh is broken")
    check_matrices_close(nm.arctanh(B), np.arctanh(Bnp), "arctanh is broken")
    check_matrices_close(nm.atanh(B), np.arctanh(Bnp), "atanh is broken")


def test_sorting():
    var np = Python.import_module("numpy")
    var A = Matrix.rand[f64]((10, 10))
    var Anp = np.matrix(A.to_numpy())

    check_matrices_close(
        nm.sort(A), np.sort(Anp, axis=None), String("Sort is broken")
    )
    for i in range(2):
        check_matrices_close(
            nm.sort(A, axis=i),
            np.sort(Anp, axis=i),
            String("Sort by axis {} is broken").format(i),
        )

    check_matrices_close(
        nm.argsort(A), np.argsort(Anp, axis=None), String("Argsort is broken")
    )
    for i in range(2):
        check_matrices_close(
            nm.argsort(A, axis=i),
            np.argsort(Anp, axis=i),
            String("Argsort by axis {} is broken").format(i),
        )


def test_searching():
    var np = Python.import_module("numpy")
    var A = Matrix.rand[f64]((10, 10))
    var Anp = np.matrix(A.to_numpy())

    check_values_close(
        nm.max(A), np.max(Anp, axis=None), String("`max` is broken")
    )
    for i in range(2):
        check_matrices_close(
            nm.max(A, axis=i),
            np.max(Anp, axis=i),
            String("`max` by axis {} is broken").format(i),
        )

    check_values_close(
        nm.argmax(A), np.argmax(Anp, axis=None), String("`argmax` is broken")
    )
    for i in range(2):
        check_matrices_close(
            nm.argmax(A, axis=i),
            np.argmax(Anp, axis=i),
            String("`argmax` by axis {} is broken").format(i),
        )

    check_values_close(
        nm.min(A), np.min(Anp, axis=None), String("`min` is broken.")
    )
    for i in range(2):
        check_matrices_close(
            nm.min(A, axis=i),
            np.min(Anp, axis=i),
            String("`min` by axis {} is broken").format(i),
        )

    check_values_close(
        nm.argmin(A), np.argmin(Anp, axis=None), String("`argmin` is broken.")
    )
    for i in range(2):
        check_matrices_close(
            nm.argmin(A, axis=i),
            np.argmin(Anp, axis=i),
            String("`argmin` by axis {} is broken").format(i),
        )
