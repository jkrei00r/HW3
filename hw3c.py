#hw3c.py
import numpy as np
from DoolittleMethod import Doolittle  # Import Doolittle method
from Gauss_Seidel import GaussSeidel  # Import Gauss-Seidel method

def is_symmetric(matrix):
    """Check if a matrix is symmetric."""
    matrix = np.array(matrix)  # Convert list of lists to numpy array
    return np.allclose(matrix, matrix.T)

def is_positive_definite(matrix):
    """Check if a matrix is positive definite using Cholesky decomposition."""
    matrix = np.array(matrix)  # Convert list of lists to numpy array
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False

def cholesky_solve(A, b):
    """Solve Ax = b using Cholesky decomposition."""
    A = np.array(A)  # Convert list of lists to numpy array
    b = np.array(b)  # Convert list to numpy array
    L = np.linalg.cholesky(A)
    y = np.linalg.solve(L, b)
    x = np.linalg.solve(L.T, y)
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
    A1 = [[1, -1, 3,2],
          [-1,5,-5,-2],
          [3,-5,19,3],
          [2,-2,3,21]]
    b1 = [15, -35, 94,1]

    print("Problem 1:")
    x1 = solve_matrix_equation(A1, b1)
    print(f"Solution vector: {x1}\n")

    # Problem 2
    A2 = [[4,2,4,0],
          [2,2,3,2],
          [4,3,6,3],
          [0,2,3,9]]
    b2 = [20,36,60,122]

    print("Problem 2:")
    x2 = solve_matrix_equation(A2, b2)
    print(f"Solution vector: {x2}\n")

if __name__ == "__main__":
    main()