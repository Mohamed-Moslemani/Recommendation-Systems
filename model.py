from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import implicit

app = Flask(__name__)

class AnimeRecommender:
    def __init__(self, anime_data_path, ratings_data_path):
        self.anime_data = pd.read_csv(anime_data_path)
        self.ratings_data = pd.read_csv(ratings_data_path)
        self.preprocess_data()
        self.model = None
        self.train_model(factors=20, regularization=0.1, iterations=20, use_gpu=False)
        
    def preprocess_data(self):
        self.user_ids = self.ratings_data['user_id'].unique()
        self.user_id_to_idx = {user_id: idx for idx, user_id in enumerate(self.user_ids)}
        self.idx_to_user_id = {idx: user_id for user_id, idx in self.user_id_to_idx.items()}
        self.ratings_data['user_idx'] = self.ratings_data['user_id'].map(self.user_id_to_idx)
        all_anime_ids = pd.concat([self.anime_data['anime_id'], self.ratings_data['anime_id']]).unique()
        self.anime_id_to_idx = {anime_id: idx for idx, anime_id in enumerate(all_anime_ids)}
        self.idx_to_anime_id = {idx: anime_id for anime_id, idx in self.anime_id_to_idx.items()}
        self.ratings_data['anime_idx'] = self.ratings_data['anime_id'].map(self.anime_id_to_idx)
        self.ratings_data['confidence'] = self.ratings_data['rating'].apply(self.get_confidence)
        num_users = len(self.user_ids)
        num_items = len(all_anime_ids)
        self.interaction_matrix = csr_matrix(
            (self.ratings_data['confidence'], (self.ratings_data['user_idx'], self.ratings_data['anime_idx'])),
            shape=(num_users, num_items)
        )

    @staticmethod
    def get_confidence(rating):
        """
        Assign confidence values based on the rating.
        """
        if rating == -1:
            return 1  # Low confidence for implicit feedback
        else:
            return 1 + rating / 10  # Higher confidence for higher ratings

    def train_model(self, factors=20, regularization=0.1, iterations=20, use_gpu=False):
        item_user_data = self.interaction_matrix.T.tocsr()
        

        self.model = implicit.als.AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            use_gpu=use_gpu
        )
        

        import logging
        logging.getLogger('implicit').setLevel(logging.ERROR)
        

        self.model.fit(item_user_data)

    def recommend_anime(self, user_id, N=10):

        if self.model is None:
            return None
        

        if user_id in self.user_id_to_idx:
            user_idx = self.user_id_to_idx[user_id]
        else:
            return None
        

        item_ids, scores = self.model.recommend(
            user_idx,
            self.interaction_matrix[user_idx],
            N=N,
            filter_already_liked_items=True
        )        
        recommended_anime_ids = [self.idx_to_anime_id.get(idx) for idx in item_ids]
        recommended_anime_ids = [aid for aid in recommended_anime_ids if aid is not None]
        
        recommended_anime = self.anime_data[self.anime_data['anime_id'].isin(recommended_anime_ids)].copy()
        
        if recommended_anime.empty:
            return None
        

        anime_id_to_score = {self.idx_to_anime_id[idx]: score for idx, score in zip(item_ids, scores) if idx in self.idx_to_anime_id}
        recommended_anime['score'] = recommended_anime['anime_id'].map(anime_id_to_score)
        recommended_anime = recommended_anime.dropna(subset=['score'])
        recommended_anime = recommended_anime.sort_values(by='score', ascending=False)
        
        return recommended_anime[['anime_id', 'name', 'genre', 'type', 'episodes', 'rating', 'members', 'score']]

anime_data_path = 'data/anime.csv'
ratings_data_path = 'data/rating.csv'
recommender = AnimeRecommender(anime_data_path, ratings_data_path)

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = None
    user_id = ''
    error_message = ''
    if request.method == 'POST':
        user_id_input = request.form.get('user_id')
        if user_id_input.isdigit():
            user_id = int(user_id_input)
            recommended_anime = recommender.recommend_anime(user_id, N=10)
            if recommended_anime is not None and not recommended_anime.empty:
                recommendations = recommended_anime.to_dict('records')
            else:
                error_message = f"No recommendations found for User ID {user_id}."
        else:
            error_message = "Please enter a valid numeric User ID."
    return render_template('index.html', recommendations=recommendations, user_id=user_id, error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)
