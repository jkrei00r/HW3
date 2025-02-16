from DoolittleMethod import Doolittle  # Import Doolittle method
from Gauss_Seidel import GaussSeidel  # Import Gauss-Seidel method

def is_symmetric(matrix):
    """Check if a matrix is symmetric."""
    rows = len(matrix)
    cols = len(matrix[0]) if rows > 0 else 0
    if rows != cols:
        return False
    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] != matrix[j][i]:
                return False
    return True

def is_positive_definite(matrix):
    """Check if a matrix is positive definite using Cholesky decomposition."""
    try:
        cholesky_decomposition(matrix)
        return True
    except ValueError:
        return False

def cholesky_decomposition(matrix):
    """Perform Cholesky decomposition on a matrix."""
    n = len(matrix)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            sum_k = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                if matrix[i][i] - sum_k <= 0:
                    raise ValueError("Matrix is not positive definite")
                L[i][j] = (matrix[i][i] - sum_k) ** 0.5
            else:
                L[i][j] = (matrix[i][j] - sum_k) / L[j][j]
    return L

def forward_substitution(L, b):
    """Solve Ly = b using forward substitution."""
    n = len(L)
    y = [0.0] * n
    for i in range(n):
        y[i] = (b[i] - sum(L[i][j] * y[j] for j in range(i))) / L[i][i]
    return y

def backward_substitution(U, y):
    """Solve Ux = y using backward substitution."""
    n = len(U)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))) / U[i][i]
    return x

def cholesky_solve(A, b):
    """Solve Ax = b using Cholesky decomposition."""
    L = cholesky_decomposition(A)
    y = forward_substitution(L, b)
    x = backward_substitution([[L[j][i] for j in range(len(L))] for i in range(len(L))], y)
    return x

def solve_matrix_equation(A, b):
    """
    Solve Ax = b using Cholesky, Doolittle, or Gauss-Seidel method.
    :param A: the coefficient matrix (list of lists)
    :param b: the right-hand side vector (list)
    :return: the solution vector x
    """
    if is_symmetric(A) and is_positive_definite(A):
        print("Matrix is symmetric and positive definite. Using Cholesky method.")
        x = cholesky_solve(A, b)
    else:
        print("Matrix is not symmetric and positive definite. Using Doolittle method.")
        Aaug = [A[i] + [b[i]] for i in range(len(A))]  # Create augmented matrix
        x = Doolittle(Aaug)  # Call Doolittle method
    return x

def main():
    # Problem 1
    A1 = [[1, -1, 3, 2],
          [-1, 5, -5, -2],
          [3, -5, 19, 3],
          [2, -2, 3, 21]]
    b1 = [15, -35, 94, 1]

    print("Problem 1:")
    x1 = solve_matrix_equation(A1, b1)
    print(f"Solution vector: {x1}\n")

    # Problem 2
    A2 = [[4, 2, 4, 0],
          [2, 2, 3, 2],
          [4, 3, 6, 3],
          [0, 2, 3, 9]]
    b2 = [20, 36, 60, 122]

    print("Problem 2:")
    x2 = solve_matrix_equation(A2, b2)
    print(f"Solution vector: {x2}\n")

if __name__ == "__main__":
    main()
