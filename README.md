# pg_recommendation_app

A simple content-based PG accommodation recommendation system built with Streamlit and scikit-learn.

## Prerequisites

Ensure you have Python 3.7+ installed and `pip` available.

## Installation

```bash
# (Optional) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```  
*If you don't have a `requirements.txt`, install manually:*  
```bash
pip install streamlit pandas scikit-learn joblib
```

## Data Preprocessing

Convert the raw CSV into feature artifacts:
```bash
python preprocessing.py
```
This will generate:
- `preprocessor.joblib`  
- `processed_features.parquet`  
- `original_data.parquet`

## Build Model

Serialize the recommender so it only builds once:
```bash
python build_model.py
```
Generates: `recommender.joblib`

## Run the App

Launch the Streamlit interface:
```bash
streamlit run app.py
```

Navigate to the URL shown in the terminal (usually `http://localhost:8501`).

---

*PG Recommendation System v1.0*