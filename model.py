import pandas as pd 


class AnimeRecommender:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.preprocess_data()
        self.all_genres = self.get_all_genres()
    
    def preprocess_data(self):
        self.data['genre'] = self.data['genre'].fillna('')
        self.data['genre'] = self.data['genre'].str.split(', ')   
        self.data['rating'] = pd.to_numeric(self.data['rating'], errors='coerce')
        self.data['members'] = pd.to_numeric(self.data['members'], errors='coerce')
        self.data.dropna(subset=['rating', 'members'], inplace=True)
    
    def get_all_genres(self):
 
        all_genres = set()
        for genres in self.data['genre']:
            all_genres.update(genres)
        return all_genres
    
    def get_user_genres(self):
        print("Available Genres:")
        print(', '.join(sorted(self.all_genres)))  
        user_genres = input("\nEnter your preferred genres, separated by commas: ")
        user_genres = [genre.strip() for genre in user_genres.split(',')]
        return user_genres
    
    def genre_match(self, anime_genres, user_genres):
        return any(genre in anime_genres for genre in user_genres)
    
    