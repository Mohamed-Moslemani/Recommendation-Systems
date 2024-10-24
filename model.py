import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
import implicit


class AnimeRecommender:
    def __init__(self, anime_data_path, ratings_data_path):
        self.anime_data = pd.read_csv(anime_data_path)
        self.ratings_data = pd.read_csv(ratings_data_path)
        self.preprocess_data()
        self.model = None


def preprocess_data(self):

        self.user_encoder = LabelEncoder()
        self.ratings_data['user_idx'] = self.user_encoder.fit_transform(self.ratings_data['user_id'])
        self.anime_encoder = LabelEncoder()
        self.ratings_data['anime_idx'] = self.anime_encoder.fit_transform(self.ratings_data['anime_id'])
        self.ratings_data['confidence'] = self.ratings_data['rating'].apply(self.get_confidence)
        self.interaction_matrix = csr_matrix(
            (self.ratings_data['confidence'], (self.ratings_data['user_idx'], self.ratings_data['anime_idx']))
        )
        self.user_id_map = dict(zip(self.user_encoder.transform(self.user_encoder.classes_), self.user_encoder.classes_))
        self.anime_id_map = dict(zip(self.anime_encoder.transform(self.anime_encoder.classes_), self.anime_encoder.classes_))
    

def get_confidence(rating):

        if rating == -1:
            return 1  
        else:
            return 1 + rating / 10  
    

def train_model(self, factors=20, regularization=0.1, iterations=20, use_gpu=False):
   
        item_user_data = self.interaction_matrix.T
        
        self.model = implicit.als.AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            use_gpu=use_gpu
        )
        
        self.model.fit(item_user_data)
        print("Model training completed.")

    
def recommend_anime(self, user_id, N=10):
 
        if self.model is None:
            print("Model has not been trained yet.")
            return None
        
        if user_id in self.user_encoder.classes_:
            user_idx = self.user_encoder.transform([user_id])[0]
        else:
            print("User ID not found.")
            return None