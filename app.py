import streamlit as st
import pickle
import pandas as pd

# --------------------------------------------------
# 1. PAGE CONFIGURATION (Frontend Setup)
# --------------------------------------------------
st.set_page_config(page_title="Netflix Recommender", page_icon="🍿", layout="centered")

st.title("🍿 Netflix AI Recommendation Engine")
st.write("Built with TF-IDF & Cosine Similarity")

# --------------------------------------------------
# 2. LOAD THE BRAIN (Backend Connection)
# --------------------------------------------------
# We use @st.cache_resource so the app only loads the heavy file ONCE.
# Otherwise, it would reload the 600MB matrix every time the user clicks a button!
@st.cache_resource
def load_model():
    with open('recommender.pkl', 'rb') as file:
        return pickle.load(file)

# Load the dictionary and unpack the variables
model = load_model()
indices = model['indices_map']
cosine_sim = model['similarity_matrix']
df = model['catalog_df']

# Extract just the movie titles for our dropdown menu
movie_titles = df['title'].values

# --------------------------------------------------
# 3. USER INTERFACE (Frontend Inputs)
# --------------------------------------------------
st.markdown("### Find your next favorite show")

# Create a dropdown menu. 
selected_movie = st.selectbox(
    "Type or select a movie you recently enjoyed:",
    movie_titles
)

# --------------------------------------------------
# 4. INFERENCE ENGINE (The Logic)
# --------------------------------------------------
def get_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Grab top 5 matches
    top_5 = sim_scores[1:6]
    movie_indices = [i[0] for i in top_5]
    
    return df.iloc[movie_indices]

# --------------------------------------------------
# 5. BUTTON & OUTPUT (Frontend Display)
# --------------------------------------------------
# st.button creates a clickable button on the screen
if st.button("Show Recommendations"):
    
    # Show a loading spinner while the math runs (good UI/UX practice)
    with st.spinner("Calculating mathematical distances..."):
        
        # Call our logic function
        recommendations = get_recommendations(selected_movie)
        
        st.success(f"Here are the top 5 matches for '{selected_movie}':")
        
        # Display the results cleanly 
        for index, row in recommendations.iterrows():
            # Use columns to make it look nice (Title on left, Genre/Cluster on right)
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.subheader(f"🎬 {row['title']}")
                st.write(f"**Genres:** {row['listed_in']}")
            
            with col2:
                # We show the cluster number just to prove our AI grouped it!
                st.caption(f"Category Box: {row['cluster']}")
            
            # Add a visual divider between movies
            st.divider()