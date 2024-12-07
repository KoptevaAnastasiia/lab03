import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds


ratings_df = pd.read_csv('tables/ratings.csv')
movies_df = pd.read_csv('tables/movies.csv')


ratings_matrix = ratings_df.pivot(index='userId', columns='movieId', values='rating')
ratings_matrix = ratings_matrix.dropna(thresh=100, axis=0)
ratings_matrix = ratings_matrix.dropna(thresh=50, axis=1)

ratings_matrix_filled = ratings_matrix.fillna(2.5)



R = ratings_matrix_filled.values
user_ratings_mean = np.mean(R, axis=1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)


def svd_predict():
    U, sigma, Vt = svds(R_demeaned, k=10)
    sigma = np.diag(sigma)

    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
    preds_df = pd.DataFrame(all_user_predicted_ratings, columns=ratings_matrix.columns, index=ratings_matrix.index)
    return preds_df

predict_data = svd_predict()




print("\n\nДані до прогнозування:\n")
print(ratings_matrix.head())

print("\n\nДані після прогнозування:\n")
print(predict_data.head())

predict_only = predict_data.where(~ratings_matrix.notna(), np.nan)
print("\n\nТільки прогнозовані дані:\n")
print(predict_only.head())



original_ratings = R_demeaned + user_ratings_mean.reshape(-1, 1)




def recommend_movies(user_id, predict_only, movies_df, num_recommendations):
    user_row_number = user_id - 1
    sorted_user_predictions = predict_only.iloc[user_row_number].sort_values(ascending=False)

    recommendations = sorted_user_predictions.head(num_recommendations)

    recommended_movie_ids = recommendations.index.tolist()
    recommended_movies = movies_df[movies_df['movieId'].isin(recommended_movie_ids)]

    return recommended_movies, recommendations



user_id = 56
num_recommendations = 10

recommended_movies, recommendations = recommend_movies(user_id, predict_only, movies_df, num_recommendations)


print("\n\nID ")
print(recommendations.index.tolist())


print("\n\n recommendations for " , user_id, ":\n")
print(recommended_movies.to_string(index=False))

