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