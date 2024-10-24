import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import implicit

class AnimeRecommender:
    def __init__(self, anime_data_path, ratings_data_path):
        """
        Constructor to load and preprocess data.
        """
        # Load data
        self.anime_data = pd.read_csv(anime_data_path)
        self.ratings_data = pd.read_csv(ratings_data_path)
        
        # Preprocess data
        self.preprocess_data()
        
        # Initialize the model as None
        self.model = None
    
    def preprocess_data(self):
        """
        Preprocess the data by encoding IDs and creating the interaction matrix.
        """
        # Create mappings for user IDs
        self.user_ids = self.ratings_data['user_id'].unique()
        self.user_id_to_idx = {user_id: idx for idx, user_id in enumerate(self.user_ids)}
        self.idx_to_user_id = {idx: user_id for user_id, idx in self.user_id_to_idx.items()}
        self.ratings_data['user_idx'] = self.ratings_data['user_id'].map(self.user_id_to_idx)
        
        # Create mappings for anime IDs using all available IDs
        all_anime_ids = pd.concat([self.anime_data['anime_id'], self.ratings_data['anime_id']]).unique()
        self.anime_id_to_idx = {anime_id: idx for idx, anime_id in enumerate(all_anime_ids)}
        self.idx_to_anime_id = {idx: anime_id for anime_id, idx in self.anime_id_to_idx.items()}
        self.ratings_data['anime_idx'] = self.ratings_data['anime_id'].map(self.anime_id_to_idx)
        
        # Define the confidence values
        self.ratings_data['confidence'] = self.ratings_data['rating'].apply(self.get_confidence)
        
        # Build the user-item interaction matrix
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
        """
        Train the recommendation model using Implicit ALS.
        """
        # Convert the interaction matrix to item-user format
        item_user_data = self.interaction_matrix.T.tocsr()
        
        # Initialize the model
        self.model = implicit.als.AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            use_gpu=use_gpu
        )
        
        # Train the model
        self.model.fit(item_user_data)
        print("Model training completed.")
    
    def recommend_anime(self, user_id, N=10):
        """
        Generate top N anime recommendations for a given user ID.
        """
        # Check if the model is trained
        if self.model is None:
            print("Model has not been trained yet.")
            return None
        
        # Encode the user ID
        if user_id in self.user_id_to_idx:
            user_idx = self.user_id_to_idx[user_id]
        else:
            print("User ID not found.")
            return None
        
        # Get recommendations
        item_ids, scores = self.model.recommend(
            user_idx,
            self.interaction_matrix[user_idx],
            N=N,
            filter_already_liked_items=True
        )
        
        # Map item indices back to anime IDs
        recommended_anime_ids = [self.idx_to_anime_id.get(idx) for idx in item_ids]
        # Filter out any None values in case of missing mappings
        recommended_anime_ids = [aid for aid in recommended_anime_ids if aid is not None]
        
        # Get anime details
        recommended_anime = self.anime_data[self.anime_data['anime_id'].isin(recommended_anime_ids)].copy()
        
        if recommended_anime.empty:
            print("No recommended anime found in anime data.")
            return None
        
        # Map anime IDs to their corresponding scores
        anime_id_to_score = {self.idx_to_anime_id[idx]: score for idx, score in zip(item_ids, scores) if idx in self.idx_to_anime_id}
        # Add scores to the dataframe
        recommended_anime['score'] = recommended_anime['anime_id'].map(anime_id_to_score)
        # Remove any entries without scores
        recommended_anime = recommended_anime.dropna(subset=['score'])
        
        # Sort by scores
        recommended_anime = recommended_anime.sort_values(by='score', ascending=False)
        
        return recommended_anime[['anime_id', 'name', 'genre', 'type', 'episodes', 'rating', 'members', 'score']]
    
    def display_recommendations(self, recommended_anime):
        """
        Display the recommended anime.
        """
        if recommended_anime is None or recommended_anime.empty:
            print("No recommendations to display.")
            return
        
        print(f"\nTop {len(recommended_anime)} Anime Recommendations:\n")
        for idx, row in recommended_anime.iterrows():
            print(f"Anime ID: {row['anime_id']}")
            print(f"Name: {row['name']}")
            print(f"Genres: {row['genre']}")
            print(f"Type: {row['type']}, Episodes: {row['episodes']}")
            print(f"Rating: {row['rating']}, Members: {row['members']}")
            print(f"Score: {row['score']}\n")

def main():
    anime_data_path = 'data//anime.csv'
    ratings_data_path = 'data//rating.csv'
    recommender = AnimeRecommender(anime_data_path, ratings_data_path)
    recommender.train_model(factors=20, regularization=0.1, iterations=20, use_gpu=False)

    try:
        user_id_input = int(input("Enter a user ID to get recommendations: "))
    except ValueError:
        print("Invalid user ID.")
        return
        
    recommended_anime = recommender.recommend_anime(user_id_input, N=10)
    recommender.display_recommendations(recommended_anime)

if __name__ == "__main__":
    main()
