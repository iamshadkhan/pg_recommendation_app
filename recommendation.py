import numpy as np
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

class PGRecommender:
    def __init__(self):
        self.preprocessor = joblib.load('preprocessor.joblib')
        self.features_df = pd.read_parquet('processed_features.parquet')
        self.original_df = pd.read_parquet('original_data.parquet')
        self.feature_columns = self.features_df.drop(columns=['PG_Name']).columns.tolist()
        
    def prepare_user_input(self, user_data):
        """Convert user input to feature vector with proper handling of missing values"""
        # Create a base dictionary with default values
        base_data = {
            'Location': "NOT_SPECIFIED",
            'Price': 0,
            'Gender_Preference': "NOT_SPECIFIED",
            'WiFi': 0,
            'Food': 0,
            'AC': 0,
            'Laundry': 0,
            'Parking': 0,
            'Security': 0,
            'Rating': 0,
            'Distance_to_Metro(km)': 0
        }
        
        # Update with user provided values
        for key in user_data:
            if key in base_data:
                # Convert binary strings to integers
                if key in ['WiFi', 'Food', 'AC', 'Laundry', 'Parking', 'Security']:
                    if isinstance(user_data[key], str):
                        base_data[key] = 1 if user_data[key].lower() == 'yes' else 0
                    else:
                        base_data[key] = user_data[key]
                else:
                    base_data[key] = user_data[key]
        
        # Create DataFrame
        user_df = pd.DataFrame([base_data])
        
        # Transform using preprocessor
        try:
            user_features = self.preprocessor.transform(user_df)
            return user_features.flatten()
        except Exception as e:
            print(f"Error transforming user input: {e}")
            return np.zeros(len(self.feature_columns))
    
    def recommend(self, user_data, top_n=5):
        """Get top recommendations based on user input"""
        # Prepare user input
        user_vec = self.prepare_user_input(user_data)
        
        # Get PG features
        pg_features = self.features_df.drop(columns=['PG_Name']).values
        
        # Create mask for specified features
        mask = np.zeros(len(self.feature_columns))
        for i, feature in enumerate(self.feature_columns):
            # Check if feature was specified by user
            if 'Location_' in feature:
                if user_data.get('Location'):
                    # Check if this one-hot feature matches the user's location
                    loc_val = user_data['Location']
                    if feature == f"Location_{loc_val}":
                        mask[i] = 1
            elif feature in ['Price', 'Distance_to_Metro(km)', 'Rating']:
                if user_data.get(feature) is not None:
                    mask[i] = 1
            elif 'Gender_Preference_' in feature:
                if user_data.get('Gender_Preference'):
                    gender_val = user_data['Gender_Preference']
                    if feature == f"Gender_Preference_{gender_val}":
                        mask[i] = 1
            elif feature in ['WiFi', 'Food', 'AC', 'Laundry', 'Parking', 'Security']:
                if user_data.get(feature) is not None:
                    mask[i] = 1
        
        # Apply mask to ignore unspecified features
        masked_user = user_vec * mask
        masked_pg_features = pg_features * mask
        
        # Calculate cosine similarity
        if np.any(mask):  # Only calculate if some features are specified
            similarities = cosine_similarity([masked_user], masked_pg_features)[0]
        else:
            similarities = np.zeros(len(pg_features))
        
        # Add to DataFrame
        results = self.features_df.copy()
        results['Similarity'] = similarities
        
        # Merge with original data, keep original columns and suffix scaled ones
        merged_results = results.merge(
            self.original_df,
            on='PG_Name',
            how='left',
            suffixes=('_feat', '')
        )
        # Remove scaled feature columns (suffix _feat), keep original data columns
        merged_results = merged_results.loc[:, ~merged_results.columns.str.endswith('_feat')]
        
        # Filter top results
        return merged_results.sort_values('Similarity', ascending=False).head(top_n)

# Example usage
# if __name__ == "__main__":
#     recommender = PGRecommender()
    
#     # Sample user query
#     user_query = {
#         'Location': 'Pitampura',
#         'Price': 8000,
#         'AC': 'Yes'
#     }
    
#     recommendations = recommender.recommend(user_query)
#     print(recommendations[['PG_Name', 'Location', 'Price', 'AC', 'Similarity']])