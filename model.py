import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import implicit
from flask import Flask, render_template, request
import logging
app = Flask(__name__)

class AnimeRecommender:
    def __init__(self, anime_data_path, ratings_data_path):
        self.anime_data = pd.read_csv(anime_data_path)
        self.ratings_data = pd.read_csv(ratings_data_path)
        self.preprocess_data()
        self.model = None
        self.train_model(factors=20, regularization=0.1, iterations=20, use_gpu=False)
        
    def preprocess_data(self):
        all_anime_ids = pd.concat([self.anime_data['anime_id'], self.ratings_data['anime_id']]).unique()
        self.anime_ids = all_anime_ids
        self.anime_id_to_idx = {anime_id: idx for idx, anime_id in enumerate(self.anime_ids)}
        self.idx_to_anime_id = {idx: anime_id for idx, anime_id in enumerate(self.anime_ids)}
        self.ratings_data['anime_idx'] = self.ratings_data['anime_id'].map(self.anime_id_to_idx)
        self.user_ids = self.ratings_data['user_id'].unique()
        self.user_id_to_idx = {user_id: idx for idx, user_id in enumerate(self.user_ids)}
        self.idx_to_user_id = {idx: user_id for idx, user_id in enumerate(self.user_ids)}
        self.ratings_data['user_idx'] = self.ratings_data['user_id'].map(self.user_id_to_idx)
        self.ratings_data['confidence'] = self.ratings_data['rating'].apply(self.get_confidence)
        num_users = len(self.user_ids)
        num_items = len(self.anime_ids)
        self.interaction_matrix = csr_matrix(
            (self.ratings_data['confidence'], (self.ratings_data['user_idx'], self.ratings_data['anime_idx'])),
            shape=(num_users, num_items)
        )


    @staticmethod
    def get_confidence(rating):

        if rating == -1:
            return 1 
        else:
            return 1 + rating / 10  

    def train_model(self, factors=20, regularization=0.1, iterations=20, use_gpu=False):
        item_user_data = self.interaction_matrix.T.tocsr()
        self.model = implicit.als.AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            use_gpu=use_gpu
        )
        logging.getLogger('implicit').setLevel(logging.ERROR)
        self.model.fit(item_user_data)

    def recommend_based_on_anime(self, selected_anime_ids, N=10):
        selected_anime_idx = [self.anime_id_to_idx.get(aid) for aid in selected_anime_ids]
        selected_anime_idx = [idx for idx in selected_anime_idx if idx is not None]
        
        if not selected_anime_idx:
            print("No valid anime IDs provided.")
            return None
        
        selected_item_factors = self.model.item_factors[selected_anime_idx]
        user_vector = np.mean(selected_item_factors, axis=0)
        scores = self.model.item_factors.dot(user_vector)
        top_indices = np.argsort(-scores)
        top_indices = [idx for idx in top_indices if idx not in selected_anime_idx]
        recommended_anime_ids = []
        for idx in top_indices:
            anime_id = self.idx_to_anime_id.get(idx)
            if anime_id:
                recommended_anime_ids.append(anime_id)
            if len(recommended_anime_ids) == N:
                break
        
        if not recommended_anime_ids:
            print("No recommendations found.")
            return None
        recommended_anime = self.anime_data[self.anime_data['anime_id'].isin(recommended_anime_ids)].copy()
        
        if recommended_anime.empty:
            print("No recommended anime found in anime data.")
            return None
        
        anime_id_to_score = {self.idx_to_anime_id[idx]: scores[idx] for idx in top_indices if idx in self.idx_to_anime_id}
        recommended_anime['score'] = recommended_anime['anime_id'].map(anime_id_to_score)
        recommended_anime = recommended_anime.dropna(subset=['score'])
        recommended_anime = recommended_anime.sort_values(by='score', ascending=False)
    
        return recommended_anime[['anime_id', 'name', 'genre', 'type', 'episodes', 'rating', 'members', 'score']]

    def get_popular_anime(self, N=100):
        """
        Return a list of popular anime (top N by member count).
        """
        popular_anime = self.anime_data.sort_values(by='members', ascending=False).head(N)
        return popular_anime[['anime_id', 'name']].drop_duplicates().to_dict('records')

anime_data_path = 'data/anime.csv'
ratings_data_path = 'data/rating.csv'
recommender = AnimeRecommender(anime_data_path, ratings_data_path)

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = None
    error_message = ''
    anime_list = recommender.get_popular_anime(N=100)
    if request.method == 'POST':
        selected_anime_ids = request.form.getlist('selected_anime')
        if selected_anime_ids and len(selected_anime_ids) <= 5:
            selected_anime_ids = [int(aid) for aid in selected_anime_ids]
            recommended_anime = recommender.recommend_based_on_anime(selected_anime_ids, N=10)
            if recommended_anime is not None and not recommended_anime.empty:
                recommendations = recommended_anime.to_dict('records')
            else:
                error_message = "No recommendations found based on the selected anime."
        else:
            error_message = "Please select up to 5 anime."
    return render_template('index.html', recommendations=recommendations, error_message=error_message, anime_list=anime_list)

if __name__ == '__main__':
    app.run(debug=True)