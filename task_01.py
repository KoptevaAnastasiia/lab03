import numpy as np


def svd(A):

    AtA = np.dot(A.T, A)
    eigvals_V, eigvecs_V = np.linalg.eigh(AtA)

    # Сортуємо власні значення та відповідні власні вектори
    sorted_indices = np.argsort(eigvals_V)[::-1]
    eigvals_V = eigvals_V[sorted_indices]
    eigvecs_V = eigvecs_V[:, sorted_indices]


    V = eigvecs_V
    sigma = np.sqrt(eigvals_V)

    #U = A * V * Σ^-1
    sigma_inv = np.diag(1 / sigma)
    U = np.dot(A, np.dot(V, sigma_inv))

    Sigma = np.diag(sigma)

    return U, Sigma, V



A = np.array([[1, 1, 3], [2, 5, 6], [7, 2, 3]])
U, Sigma, V = svd(A)

# Відновлення початкової матриці
A_new = np.dot(U, np.dot(Sigma, V.T))

print("A:\n", A)
print("A_new:\n", A_new)
print("U:\n", U)
print("Sigma:\n", Sigma)
print("V:\n", V.T)
