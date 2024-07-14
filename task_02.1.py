import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds



# Зчитування CSV файлу
file_path = 'ratings.csv'
df = pd.read_csv(file_path)

ratings_matrix = df.pivot(index='userId', columns='movieId', values='rating')




ratings_matrix = ratings_matrix.dropna(thresh=200, axis=0)
ratings_matrix = ratings_matrix.dropna(thresh=100, axis=1)

ratings_matrix_filled = ratings_matrix.fillna(2.5)
R = ratings_matrix_filled.values
user_ratings_mean = np.mean(R, axis=1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)


U, sigma, Vt = svds(R_demeaned, k=3)


def plot_users(U):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(U[:, 0], U[:, 1], U[:, 2])
    plt.show()

plot_users(U)

V = Vt.T

def plot_movies(V):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(V[:, 0], V[:, 1], V[:, 2])

    plt.show()

plot_movies(V)


print("U:\n\n", U, "\n")
print("Sigma:\n\n", sigma, "\n")
print("Vt:\n\n", Vt, "\n")
print("V:\n\n", V)
