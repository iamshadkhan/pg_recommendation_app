import streamlit as st
import pandas as pd
import numpy as np
from recommendation import PGRecommender

# Initialize recommender
@st.cache_resource
def load_recommender():
    try:
        return PGRecommender()
    except Exception as e:
        st.error(f"Error loading recommender: {e}")
        st.error("Make sure you've run preprocessing.py first")
        return None

recommender = load_recommender()

# App title
st.title("🏠 PG Accommodation Finder")
st.markdown("Find the perfect PG based on your preferences")

# User input section
st.sidebar.header("Your Preferences")

# Get available locations from data
try:
    original_df = pd.read_parquet('original_data.parquet')
    locations = [""] + sorted(original_df['Location'].unique().tolist())
    genders = [""] + sorted(original_df['Gender_Preference'].unique().tolist())
except:
    locations = [""]
    genders = [""]

# Location input
location = st.sidebar.selectbox("Preferred Location", locations)

# Price input
price = st.sidebar.number_input("Max Budget (₹)", min_value=1500, value=None, step=500)

# Gender preference
gender = st.sidebar.selectbox("Gender Preference", genders)

# Amenities
st.sidebar.subheader("Amenities (Optional)")
amenity_options = ["", "Yes", "No"]
wifi = st.sidebar.selectbox("WiFi", amenity_options)
food = st.sidebar.selectbox("Food Included", amenity_options)
ac = st.sidebar.selectbox("Air Conditioning", amenity_options)
laundry = st.sidebar.selectbox("Laundry Service", amenity_options)
parking = st.sidebar.selectbox("Parking", amenity_options)
security = st.sidebar.selectbox("Security", amenity_options)

# Additional preferences
st.sidebar.subheader("Additional Preferences")
rating = st.sidebar.slider("Minimum Rating", 0.0, 5.0, 0.0, 0.1)
distance = st.sidebar.number_input("Max Distance to Metro (km)", min_value=0.0, value=None, step=0.1)

# Prepare user data
user_data = {}
if location: user_data['Location'] = location
if price: user_data['Price'] = price
if gender: user_data['Gender_Preference'] = gender
if wifi: user_data['WiFi'] = wifi
if food: user_data['Food'] = food
if ac: user_data['AC'] = ac
if laundry: user_data['Laundry'] = laundry
if parking: user_data['Parking'] = parking
if security: user_data['Security'] = security
if rating > 0: user_data['Rating'] = rating
if distance: user_data['Distance_to_Metro(km)'] = distance

# Recommendation button
if st.sidebar.button("Find PGs", type="primary") or user_data:
    if not any([location, price]):  # Require at least location or price
        st.warning("Please provide at least Location or Price")
    elif recommender is None:
        st.error("Recommendation system failed to load. Please check data files.")
    else:
        with st.spinner("Finding best matches..."):
            try:
                results = recommender.recommend(user_data)
            except Exception as e:
                st.error(f"Error getting recommendations: {e}")
                results = pd.DataFrame()
        
        if results.empty:
            st.error("No PGs match your criteria. Try different preferences.")
        else:
            st.subheader(f"Top {len(results)} Recommendations")
            
            # Display results
            for _, row in results.iterrows():
                with st.expander(f"🏆 {row['PG_Name']} - Similarity: {row['Similarity']:.2f}"):
                    col1, col2 = st.columns(2)
                    col1.metric("Price", f"₹{row['Price']}")
                    
                    # Handle possible missing distance
                    distance_val = row.get('Distance_to_Metro(km)', 'N/A')
                    col2.metric("Distance to Metro", f"{float(distance_val):.3f} km" if distance_val != 'N/A' else "N/A")
                    
                    # Amenities
                    amenities = []
                    if row.get('WiFi') == 1: amenities.append("WiFi")
                    if row.get('Food') == 1: amenities.append("Food")
                    if row.get('AC') == 1: amenities.append("AC")
                    if row.get('Laundry') == 1: amenities.append("Laundry")
                    if row.get('Parking') == 1: amenities.append("Parking")
                    if row.get('Security') == 1: amenities.append("Security")
                    
                    st.write(f"**Location:** {row['Location']} | **Gender:** {row['Gender_Preference']}")
                    st.write(f"**Amenities:** {', '.join(amenities) if amenities else 'None'}")
                    st.write(f"**Rating:** {float(row['Rating']):.1f} ⭐")
                    st.write(f"**Contact:** {row['Contact']}")
else:
    st.info("Please enter your preferences in the sidebar and click 'Find PGs'")
    st.image("https://images.unsplash.com/photo-1560448204-e02f11c3d0e2?auto=format&fit=crop&w=1200&h=600", 
             caption="Find your perfect PG accommodation")

# Footer
st.markdown("---")
st.caption("PG Recommendation System v1.0 | Uses content-based filtering with cosine similarity")
