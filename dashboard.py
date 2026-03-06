"""
IMDb Movie Success Analysis - Interactive Dashboard
This dashboard provides comprehensive visualizations and insights into IMDb movie success factors.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="IMDb Success Analysis Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    
    /* Enhanced Metric Cards */
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: bold;
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 16px;
        font-weight: 500;
    }
    
    /* Custom metric card styling */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        color: white;
        margin: 10px 0;
    }
    
    .metric-card-blue {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .metric-card-green {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    
    .metric-card-orange {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    
    .metric-card-purple {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    
    .metric-card-red {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    }
    
    .metric-value {
        font-size: 32px;
        font-weight: bold;
        margin: 10px 0;
    }
    
    .metric-label {
        font-size: 14px;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-icon {
        font-size: 24px;
        margin-bottom: 10px;
    }
    
    h1 {
        color: #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("🎬 IMDb Movie Success Analysis Dashboard")
st.markdown("### Explore factors that contribute to success on IMDb")
st.markdown("---")

# Sidebar for filters and controls
with st.sidebar:
    st.header("📊 Dashboard Controls")
    st.markdown("### Data Filters")
    
    # Year range filter
    year_min = st.number_input("Start Year", min_value=1920, max_value=2025, value=1920)
    year_max = st.number_input("End Year", min_value=1920, max_value=2025, value=2025)
    
    # Rating filter
    rating_min = st.slider("Minimum Rating", 0.0, 10.0, 0.0, 0.1)
    
    # Votes filter
    votes_min = st.number_input("Minimum Number of Votes", min_value=0, value=0)
    
    st.markdown("---")
    st.markdown("### About")
    st.info("""
    This dashboard analyzes IMDb data to understand:
    - Rating distributions
    - Content type performance
    - Popularity vs. quality
    - Temporal trends
    - Genre analysis
    """)

# Metric cards (already defined in CSS)
def create_stat_summary(df, col):
    """Calculate advanced stats"""
    mean_val = df[col].mean()
    median_val = df[col].median()
    skew_val = df[col].skew()
    kurtosis_val = df[col].kurtosis()
    return f"**Mean:** {mean_val:.2f} | **Median:** {median_val:.2f} | **Skewness:** {skew_val:.2f} | **Kurtosis:** {kurtosis_val:.2f}"

# Cache data loading for performance
@st.cache_data
def load_data():
    """Load and prepare the IMDb dataset efficiently"""
    with st.spinner("Loading data... This may take a moment."):
        target_types = ['movie', 'tvSeries', 'tvMiniSeries', 'tvMovie', 'tvSpecial', 'video']
        chunks = []
        
        try:
            # We add runtimeMinutes for extra analysis
            cols_to_use = ['tconst', 'titleType', 'startYear', 'genres', 'primaryTitle']
            chunk_iterator = pd.read_csv(
                "title.basics.tsv", 
                sep="\t", 
                chunksize=100000, 
                usecols=cols_to_use, 
                encoding='utf-8', 
                on_bad_lines='skip'
            )
            
            for chunk in chunk_iterator:
                filtered_chunk = chunk[chunk['titleType'].isin(target_types)].copy()
                filtered_chunk['startYear'] = pd.to_numeric(filtered_chunk['startYear'], errors='coerce')
                chunks.append(filtered_chunk)
            
            basics = pd.concat(chunks, axis=0)
            ratings = pd.read_csv("title.ratings.tsv", sep="\t", low_memory=False)
            df = pd.merge(basics, ratings, on='tconst')
            
            # Feature Engineering: Success Index (balancing rating and volume)
            # Use log10 to handle huge discrepancy between 100 votes and 2M votes
            df['success_score'] = df['averageRating'] * np.log10(df['numVotes'] + 1)
            
            return df
        except FileNotFoundError:
            st.error("⚠️ Data files not found! Please ensure 'title.basics.tsv' and 'title.ratings.tsv' are in the working directory.")
            return None
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None

# Load data
df = load_data()

if df is not None:
    # Sidebar Filters
    with st.sidebar:
        st.markdown("### Search & Highlight")
        search_title = st.text_input("Find a specific movie:", placeholder="e.g. The Shawshank Redemption")
        
    # Apply filters
    filtered_df = df[
        (df['startYear'] >= year_min) & 
        (df['startYear'] <= year_max) &
        (df['averageRating'] >= rating_min) &
        (df['numVotes'] >= votes_min)
    ].copy()
    
    # Key Metrics Section
    st.header("📈 Key Metrics")
    
    total_titles = len(filtered_df)
    avg_rating = filtered_df['averageRating'].mean()
    avg_success = filtered_df['success_score'].mean()
    total_votes = filtered_df['numVotes'].sum()
    content_types = filtered_df['titleType'].nunique()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f'<div class="metric-card metric-card-blue"><div class="metric-icon">🎬</div><div class="metric-label">Titles</div><div class="metric-value">{total_titles:,}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card metric-card-green"><div class="metric-icon">⭐</div><div class="metric-label">Avg Rating</div><div class="metric-value">{avg_rating:.2f}</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card metric-card-orange"><div class="metric-icon">🚀</div><div class="metric-label">Success Index</div><div class="metric-value">{avg_success:.2f}</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card metric-card-purple"><div class="metric-icon">👥</div><div class="metric-label">Total Votes</div><div class="metric-value">{total_votes:,.0f}</div></div>', unsafe_allow_html=True)
    with col5:
        st.markdown(f'<div class="metric-card metric-card-red"><div class="metric-icon">🎭</div><div class="metric-label">Formats</div><div class="metric-value">{content_types}</div></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Distribution", 
        "🎭 Content Review", 
        "⭐ Success Analysis",
        "🔗 Pairwise Insight",
        "📅 Time Machine",
        "🎪 Genre Spotlight",
        "📋 Raw Data"
    ])
    
    # Tab 1: Overview - Rating Distribution
    with tab1:
        st.header("How are ratings spread?")
        st.markdown(create_stat_summary(filtered_df, 'averageRating'))
        
        # Enhanced Histogram with Marginal Plots
        fig = px.histogram(
            filtered_df, 
            x='averageRating', 
            nbins=40,
            marginal="box", # Add a box plot on top
            title='IMDb Rating Distribution',
            labels={'averageRating': 'Average Rating'},
            color_discrete_sequence=['#764ba2']
        )
        # Add Mean and Median lines
        fig.add_vline(x=avg_rating, line_dash="dash", line_color="green", annotation_text=f"Mean: {avg_rating:.2f}")
        fig.add_vline(x=filtered_df['averageRating'].median(), line_dash="dot", line_color="orange", annotation_text=f"Med: {filtered_df['averageRating'].median():.2f}")
        
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("🎓 Statistical Note: Is it a Bell Curve?"):
            st.write("""
            IMDb ratings are typically **negatively skewed** (right-leaning), meaning most users are helpful and rate above 6.0. 
            A high **Kurtosis** indicates that ratings are tightly packed around the median, with few extremely 'bad' movies becoming viral.
            """)
    
    # Tab 2: Content Type Analysis
    with tab2:
        st.header("Format Performance")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.box(filtered_df, x='titleType', y='averageRating', color='titleType', title='Content Type Quality Range')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Leaderboard")
            st.dataframe(filtered_df.groupby('titleType')['averageRating'].mean().sort_values(ascending=False), use_container_width=True)

    # Tab 3: Success Analysis (New scientific focus)
    with tab3:
        st.header("Quality vs. Popularity")
        
        # Sample for performance
        sample_size = min(15000, len(filtered_df))
        sample_df = filtered_df.sample(n=sample_size, random_state=42)
        
        # Scatter with Search Highlighting
        sample_df['is_highlight'] = 'Normal'
        if search_title:
            sample_df.loc[sample_df['primaryTitle'].str.contains(search_title, case=False, na=False), 'is_highlight'] = 'Highlighted'
        
        fig = px.scatter(
            sample_df, x='numVotes', y='averageRating', 
            color='is_highlight', size='success_score', 
            hover_data=['primaryTitle', 'startYear'],
            log_x=True, title=f"Success Quadrant (Sample: {sample_size:,} entries)",
            color_discrete_map={'Normal': 'rgba(100, 100, 100, 0.4)', 'Highlighted': '#f5576c'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation Analysis
        corr = filtered_df[['numVotes', 'averageRating']].corr().iloc[0, 1]
        st.metric("Pearson Correlation Coefficient", f"{corr:.3f}")
        
        st.info(f"**Interpretation:** A correlation of {corr:.3f} suggests that popularity and quality are {'strongly' if abs(corr)>0.5 else 'moderately' if abs(corr)>0.3 else 'weakly'} linked.")

        colA, colB = st.columns(2)
        with colA:
            st.subheader("🏆 The 'Ultimate' Winners (Success Index)")
            st.write("*Highest calculated Rating × log(Votes)*")
            top_success = filtered_df.nlargest(10, 'success_score')[['primaryTitle', 'averageRating', 'numVotes', 'success_score']]
            st.dataframe(top_success, use_container_width=True)
        with colB:
            st.subheader("💎 Hidden Gems")
            st.write("*High Rating but fewer than 5,000 votes*")
            gems = filtered_df[(filtered_df['averageRating'] >= 8.5) & (filtered_df['numVotes'] < 5000)].sort_values('averageRating', ascending=False)
            st.dataframe(gems[['primaryTitle', 'averageRating', 'numVotes']].head(10), use_container_width=True)

    # New Tab 4: Pairwise Insights (The Interactive 'Pair Plot')
    with tab4:
        st.header("🔗 Multi-Dimensional Relationships")
        st.markdown("""
        Below is an interactive **Scatter Matrix** (interactive Pair Plot). 
        You can see how Year, Rating, and Votes interlink.
        - **Diagonal**: Distributions.
        - **Others**: Pairwise correlations.
        """)
        
        # Use a smaller sample for the matrix for performance
        matrix_sample = filtered_df.sample(n=min(2000, len(filtered_df)), random_state=42)
        
        fig_matrix = px.scatter_matrix(
            matrix_sample,
            dimensions=['startYear', 'averageRating', 'numVotes', 'success_score'],
            color='titleType',
            title="Interactive Pair Plots (Sample: 2,000 points)",
            hover_data=['primaryTitle'],
            opacity=0.4
        )
        fig_matrix.update_layout(height=800)
        # Use log scale for numVotes in the matrix if possible for better visibility
        st.plotly_chart(fig_matrix, use_container_width=True)
        
        st.info("💡 Hint: Select a Content Type in the legend to filter the matrix view!")

    # Tab 5: Temporal Trends
    with tab5:
        st.header("The Evolution of IMDb")
        yearly = filtered_df.groupby('startYear').agg({'averageRating': 'mean', 'tconst': 'count'}).reset_index()
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=yearly['startYear'], y=yearly['averageRating'], name="Avg Rating"), secondary_y=False)
        fig.add_trace(go.Bar(x=yearly['startYear'], y=yearly['tconst'], name="Title Volume", opacity=0.3), secondary_y=True)
        
        fig.update_layout(title="Volume vs Quality Over Time", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    # Tab 6: Genre Analysis
    with tab6:
        st.header("The Best & Worst Genres")
        
        # Explode genres
        genre_df = filtered_df.dropna(subset=['genres']).copy()
        genre_df['genres'] = genre_df['genres'].str.split(',')
        genre_exploded = genre_df.explode('genres')
        
        g_stats = genre_exploded.groupby('genres').agg({
            'averageRating': 'mean',
            'tconst': 'count',
            'success_score': 'mean'
        }).reset_index()
        
        # Min titles filter
        min_p = st.selectbox("Frequency Threshold", options=[10, 50, 100, 500, 1000], index=2)
        g_stats = g_stats[g_stats['tconst'] >= min_p].sort_values('success_score', ascending=False)
        
        col1, col2 = st.columns(2)
        with col1:
            fig_success = px.bar(g_stats.head(20), x='success_score', y='genres', orientation='h', 
                                color='averageRating',
                                title="Top 20 Genres by Success Index", 
                                labels={'success_score': 'Scientific Success Index'})
            st.plotly_chart(fig_success, use_container_width=True)
            
        with col2:
            vol_stats = g_stats.sort_values('tconst', ascending=False).head(20)
            fig_vol = px.bar(vol_stats, x='tconst', y='genres', orientation='h', 
                            color='tconst',
                            title="Top 20 Genres by Title Volume", 
                            labels={'tconst': 'Number of Titles'})
            st.plotly_chart(fig_vol, use_container_width=True)
            
        with st.expander("📚 What is the 'Success Index'?"):
            st.write("""
            Success isn't just a high rating. A movie with 10.0 from 5 friends is less 'successful' than a 9.0 from 2 million voters. 
            The **Success Index** is calculated as:  
            `Rating * log10(Votes + 1)`  
            This balances the quality (Rating) with the cultural impact (Votes).
            """)

    # Tab 7: Data Explorer
    with tab7:
        st.header("Search Data")
        st.dataframe(filtered_df.sort_values('success_score', ascending=False).head(100), use_container_width=True)
        st.download_button("📥 Download Filtered TSV", filtered_df.to_csv(sep='\t', index=False).encode('utf-8'), "imdb_analysis.tsv", "text/tab-separated-values")

else:
    st.error("Data Load Failed.")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; opacity: 0.5;'>IMDb Success Dashboard v2.0 | Advanced Analytics Mode</div>", unsafe_allow_html=True)
