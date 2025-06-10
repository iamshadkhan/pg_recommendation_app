"""
Utility script to build and serialize the PGRecommender instance once, so the Streamlit app can load it quickly.
Run this script after preprocessing to create `recommender.joblib`.
Usage:
    python build_model.py
"""
from recommendation import PGRecommender
import joblib

if __name__ == "__main__":
    print("Building PGRecommender...")
    recommender = PGRecommender()
    joblib.dump(recommender, 'recommender.joblib')
    print("Saved serialized model to recommender.joblib")
