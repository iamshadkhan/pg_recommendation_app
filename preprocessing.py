import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

def load_data(file_path):
    df = pd.read_csv(file_path)
    # Ensure required columns exist
    if 'Location' not in df or 'Price' not in df:
        raise ValueError("Dataset must contain 'Location' and 'Price' columns")
    # Drop rows missing location or price
    df = df.dropna(subset=['Location', 'Price'])
    return df

def create_preprocessor():
    # Define feature types
    numerical_features = ['Price', 'Distance_to_Metro(km)', 'Rating']
    categorical_features = ['Gender_Preference']
    binary_features = ['WiFi', 'Food', 'AC', 'Laundry', 'Parking', 'Security']
    location_feature = ['Location']
    
    # Create transformers
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Convert binary features to integers
    binary_transformer = Pipeline(steps=[
        ('binary', OneHotEncoder(drop='if_binary', sparse_output=False))
    ])
    
    location_transformer = Pipeline(steps=[
        ('location', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Create column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features),
            ('binary', binary_transformer, binary_features),
            ('loc', location_transformer, location_feature)
        ],
        remainder='drop'
    )
    
    return preprocessor

def process_data(input_file, output_file):
    # Load data
    df = load_data(input_file)
    
    # Create and fit preprocessor
    preprocessor = create_preprocessor()
    
    # Convert binary features to integers
    binary_features = ['WiFi', 'Food', 'AC', 'Laundry', 'Parking', 'Security']
    for col in binary_features:
        df[col] = df[col].map({'Yes': 1, 'No': 0})
    
    features = preprocessor.fit_transform(df)
    
    # Get feature names
    num_features = preprocessor.named_transformers_['num'].get_feature_names_out()
    cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out()
    bin_features = preprocessor.named_transformers_['binary'].get_feature_names_out()
    loc_features = preprocessor.named_transformers_['loc'].get_feature_names_out()
    
    all_features = list(num_features) + list(cat_features) + list(bin_features) + list(loc_features)
    
    # Create feature DataFrame
    features_df = pd.DataFrame(features, columns=all_features)
    
    # Add PG name for reference
    features_df['PG_Name'] = df['PG_Name'].values
    
    # Save artifacts
    joblib.dump(preprocessor, 'preprocessor.joblib')
    features_df.to_parquet(output_file, index=False)
    
    # Save original data with consistent columns
    original_df = df[['PG_Name', 'Location', 'Price', 'Gender_Preference', 
                     'WiFi', 'Food', 'AC', 'Laundry', 'Parking', 'Security', 
                     'Rating', 'Distance_to_Metro(km)', 'Contact']]
    original_df.to_parquet('original_data.parquet', index=False)
    
    return features_df

if __name__ == "__main__":
    features_df = process_data('pg_data.csv', 'processed_features.parquet')
    print("Preprocessing completed. Artifacts saved.")