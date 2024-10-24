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
    