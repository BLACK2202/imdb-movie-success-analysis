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

# Cache data loading for performance
@st.cache_data
def load_data():
    """Load and prepare the IMDb dataset efficiently"""
    with st.spinner("Loading data... This may take a moment."):
        target_types = ['movie', 'tvSeries', 'tvMiniSeries', 'tvMovie', 'tvSpecial', 'video']
        chunks = []
        
        try:
            cols_to_use = ['tconst', 'titleType', 'startYear', 'genres', 'primaryTitle']
            chunk_iterator = pd.read_csv(
                "title.basics.tsv", 
                sep="\t", 
                chunksize=50000, 
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
    # Apply filters
    filtered_df = df[
        (df['startYear'] >= year_min) & 
        (df['startYear'] <= year_max) &
        (df['averageRating'] >= rating_min) &
        (df['numVotes'] >= votes_min)
    ].copy()
    
    # Key Metrics Section
    st.header("📈 Key Metrics")
    
    # Calculate metrics
    total_titles = len(filtered_df)
    avg_rating = filtered_df['averageRating'].mean()
    median_rating = filtered_df['averageRating'].median()
    total_votes = filtered_df['numVotes'].sum()
    content_types = filtered_df['titleType'].nunique()
    
    # Create custom metric cards with colors
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
            <div class="metric-card metric-card-blue">
                <div class="metric-icon">🎬</div>
                <div class="metric-label">Total Titles</div>
                <div class="metric-value">{total_titles:,}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="metric-card metric-card-green">
                <div class="metric-icon">⭐</div>
                <div class="metric-label">Average Rating</div>
                <div class="metric-value">{avg_rating:.2f}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class="metric-card metric-card-orange">
                <div class="metric-icon">📊</div>
                <div class="metric-label">Median Rating</div>
                <div class="metric-value">{median_rating:.2f}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
            <div class="metric-card metric-card-purple">
                <div class="metric-icon">👥</div>
                <div class="metric-label">Total Votes</div>
                <div class="metric-value">{total_votes:,.0f}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
            <div class="metric-card metric-card-red">
                <div class="metric-icon">🎭</div>
                <div class="metric-label">Content Types</div>
                <div class="metric-value">{content_types}</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview", 
        "🎭 Content Types", 
        "⭐ Popularity vs Quality",
        "📅 Temporal Trends",
        "🎪 Genre Analysis",
        "📋 Data Explorer"
    ])
    
    # Tab 1: Overview - Rating Distribution
    with tab1:
        st.header("Rating Distribution")
        st.markdown("How are ratings spread across all titles?")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Histogram with Plotly
            fig = px.histogram(
                filtered_df, 
                x='averageRating', 
                nbins=30,
                title='Distribution of IMDb Ratings',
                labels={'averageRating': 'Average Rating', 'count': 'Number of Titles'},
                color_discrete_sequence=['#636EFA']
            )
            fig.update_layout(
                showlegend=False,
                height=500,
                xaxis_title="Average Rating",
                yaxis_title="Count"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Statistics")
            st.write(f"**Mean:** {filtered_df['averageRating'].mean():.2f}")
            st.write(f"**Median:** {filtered_df['averageRating'].median():.2f}")
            st.write(f"**Mode:** {filtered_df['averageRating'].mode()[0]:.2f}")
            st.write(f"**Std Dev:** {filtered_df['averageRating'].std():.2f}")
            st.write(f"**Min:** {filtered_df['averageRating'].min():.2f}")
            st.write(f"**Max:** {filtered_df['averageRating'].max():.2f}")
            
            st.subheader("Percentiles")
            st.write(f"**25th:** {filtered_df['averageRating'].quantile(0.25):.2f}")
            st.write(f"**50th:** {filtered_df['averageRating'].quantile(0.50):.2f}")
            st.write(f"**75th:** {filtered_df['averageRating'].quantile(0.75):.2f}")
            st.write(f"**90th:** {filtered_df['averageRating'].quantile(0.90):.2f}")
    
    # Tab 2: Content Type Analysis
    with tab2:
        st.header("Rating Distribution by Content Type")
        st.markdown("Which format tends to get higher ratings: Movies, TV Series, or others?")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Box plot
            fig = px.box(
                filtered_df, 
                x='titleType', 
                y='averageRating',
                title='Rating Distribution by Content Type',
                labels={'titleType': 'Content Type', 'averageRating': 'Average Rating'},
                color='titleType'
            )
            fig.update_layout(height=500, showlegend=False)
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Content Type Breakdown")
            type_stats = filtered_df.groupby('titleType').agg({
                'averageRating': 'mean',
                'numVotes': 'sum',
                'tconst': 'count'
            }).round(2)
            type_stats.columns = ['Avg Rating', 'Total Votes', 'Count']
            type_stats = type_stats.sort_values('Avg Rating', ascending=False)
            st.dataframe(type_stats, use_container_width=True)
            
            # Pie chart of content types
            fig_pie = px.pie(
                filtered_df, 
                names='titleType',
                title='Distribution of Content Types',
                hole=0.4
            )
            fig_pie.update_layout(height=300)
            st.plotly_chart(fig_pie, use_container_width=True)
    
    # Tab 3: Popularity vs Quality
    with tab3:
        st.header("Popularity vs. Quality Analysis")
        st.markdown("Do popular titles (more votes) always have higher ratings?")
        
        # Sample for performance
        sample_size = min(10000, len(filtered_df))
        sample_df = filtered_df.sample(n=sample_size, random_state=42)
        
        # Scatter plot with log scale
        fig = px.scatter(
            sample_df,
            x='numVotes',
            y='averageRating',
            color='titleType',
            title=f'Popularity (Votes) vs. Quality (Rating) - Sample of {sample_size:,} titles',
            labels={'numVotes': 'Number of Votes (Log Scale)', 'averageRating': 'Average Rating'},
            hover_data=['primaryTitle'],
            opacity=0.6
        )
        fig.update_xaxes(type="log")
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation analysis
        st.subheader("Correlation Analysis")
        correlation = filtered_df[['numVotes', 'averageRating']].corr().iloc[0, 1]
        st.write(f"**Correlation between Votes and Rating:** {correlation:.3f}")
        
        if correlation > 0.3:
            st.success("✅ Positive correlation: More popular titles tend to have higher ratings")
        elif correlation < -0.3:
            st.error("❌ Negative correlation: More popular titles tend to have lower ratings")
        else:
            st.info("ℹ️ Weak correlation: Popularity and quality are relatively independent")
        
        # Top voted titles
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🔥 Most Popular Titles")
            top_voted = filtered_df.nlargest(10, 'numVotes')[['primaryTitle', 'averageRating', 'numVotes', 'titleType']]
            st.dataframe(top_voted, use_container_width=True)
        
        with col2:
            st.subheader("⭐ Highest Rated Titles (min 1000 votes)")
            top_rated = filtered_df[filtered_df['numVotes'] >= 1000].nlargest(10, 'averageRating')[
                ['primaryTitle', 'averageRating', 'numVotes', 'titleType']
            ]
            st.dataframe(top_rated, use_container_width=True)
    
    # Tab 4: Temporal Trends
    with tab4:
        st.header("Success Over Time")
        st.markdown("How have ratings and production output changed over the years?")
        
        # Filter valid years
        year_df = filtered_df[
            (filtered_df['startYear'] >= 1920) & 
            (filtered_df['startYear'] <= 2025)
        ].copy()
        
        # Calculate yearly statistics
        yearly_stats = year_df.groupby('startYear').agg({
            'averageRating': 'mean',
            'tconst': 'count',
            'numVotes': 'sum'
        }).reset_index()
        yearly_stats.columns = ['Year', 'Avg Rating', 'Title Count', 'Total Votes']
        
        # Line chart for ratings over time
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Average Rating Over Time', 'Number of Titles Released Per Year'),
            vertical_spacing=0.15
        )
        
        # Rating trend
        fig.add_trace(
            go.Scatter(
                x=yearly_stats['Year'], 
                y=yearly_stats['Avg Rating'],
                mode='lines+markers',
                name='Average Rating',
                line=dict(color='purple', width=2),
                marker=dict(size=4)
            ),
            row=1, col=1
        )
        
        # Title count trend
        fig.add_trace(
            go.Scatter(
                x=yearly_stats['Year'], 
                y=yearly_stats['Title Count'],
                mode='lines',
                name='Title Count',
                line=dict(color='orange', width=2),
                fill='tozeroy'
            ),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text="Year", row=2, col=1)
        fig.update_yaxes(title_text="Average Rating", row=1, col=1)
        fig.update_yaxes(title_text="Number of Titles", row=2, col=1)
        fig.update_layout(height=800, showlegend=False)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Decade analysis
        st.subheader("📊 Analysis by Decade")
        year_df['decade'] = (year_df['startYear'] // 10) * 10
        decade_stats = year_df.groupby('decade').agg({
            'averageRating': 'mean',
            'tconst': 'count',
            'numVotes': 'mean'
        }).round(2)
        decade_stats.columns = ['Avg Rating', 'Title Count', 'Avg Votes']
        st.dataframe(decade_stats, use_container_width=True)
    
    # Tab 5: Genre Analysis
    with tab5:
        st.header("Genre Performance Analysis")
        st.markdown("Which genres have the highest average ratings?")
        
        # Process genres
        genre_df = filtered_df.dropna(subset=['genres']).copy()
        genre_df['genres'] = genre_df['genres'].str.split(',')
        genre_exploded = genre_df.explode('genres')
        
        # Calculate genre statistics
        genre_stats = genre_exploded.groupby('genres').agg(
            avg_rating=('averageRating', 'mean'),
            count=('tconst', 'count'),
            total_votes=('numVotes', 'sum'),
            avg_votes=('numVotes', 'mean')
        ).reset_index()
        
        # Filter for statistical significance
        min_titles = st.slider("Minimum titles per genre", 10, 1000, 100)
        genre_stats = genre_stats[genre_stats['count'] >= min_titles].sort_values('avg_rating', ascending=False)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Top genres bar chart
            top_n = st.slider("Number of top genres to display", 5, 30, 20)
            fig = px.bar(
                genre_stats.head(top_n),
                x='avg_rating',
                y='genres',
                orientation='h',
                title=f'Top {top_n} Genres by Average Rating (min. {min_titles} titles)',
                labels={'avg_rating': 'Average Rating', 'genres': 'Genre'},
                color='avg_rating',
                color_continuous_scale='viridis'
            )
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📊 Genre Statistics")
            display_stats = genre_stats.head(10)[['genres', 'avg_rating', 'count', 'avg_votes']].round(2)
            display_stats.columns = ['Genre', 'Avg Rating', 'Count', 'Avg Votes']
            st.dataframe(display_stats, use_container_width=True)
            
            # Most popular genres
            st.subheader("🔥 Most Produced Genres")
            top_genres = genre_stats.nlargest(10, 'count')[['genres', 'count', 'avg_rating']].round(2)
            top_genres.columns = ['Genre', 'Count', 'Avg Rating']
            st.dataframe(top_genres, use_container_width=True)
        
        # Genre comparison
        st.subheader("🎭 Genre Comparison Matrix")
        selected_genres = st.multiselect(
            "Select genres to compare",
            options=genre_stats['genres'].tolist(),
            default=genre_stats.head(5)['genres'].tolist()
        )
        
        if selected_genres:
            comparison_data = genre_stats[genre_stats['genres'].isin(selected_genres)]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=comparison_data['count'],
                y=comparison_data['avg_rating'],
                mode='markers+text',
                text=comparison_data['genres'],
                textposition="top center",
                marker=dict(
                    size=comparison_data['total_votes'] / comparison_data['total_votes'].max() * 100,
                    color=comparison_data['avg_rating'],
                    colorscale='viridis',
                    showscale=True,
                    colorbar=dict(title="Avg Rating")
                )
            ))
            fig.update_layout(
                title="Genre Comparison: Count vs Rating (bubble size = total votes)",
                xaxis_title="Number of Titles",
                yaxis_title="Average Rating",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 6: Data Explorer
    with tab6:
        st.header("📋 Data Explorer")
        st.markdown("Explore the raw data with interactive filtering")
        
        # Display options
        col1, col2 = st.columns(2)
        with col1:
            show_rows = st.number_input("Number of rows to display", 10, 1000, 100)
        with col2:
            sort_by = st.selectbox("Sort by", ['averageRating', 'numVotes', 'startYear', 'primaryTitle'])
        
        sort_order = st.radio("Sort order", ['Descending', 'Ascending'], horizontal=True)
        ascending = (sort_order == 'Ascending')
        
        # Display data
        display_df = filtered_df.sort_values(sort_by, ascending=ascending).head(show_rows)
        st.dataframe(display_df, use_container_width=True, height=600)
        
        # Download option
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=filtered_df.to_csv(index=False).encode('utf-8'),
            file_name='imdb_filtered_data.csv',
            mime='text/csv',
        )
        
        # Data summary
        st.subheader("📊 Data Summary")
        st.write(filtered_df.describe())

else:
    st.error("Failed to load data. Please check if the data files are available.")
    st.info("Required files: title.basics.tsv and title.ratings.tsv")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>🎬 IMDb Movie Success Analysis Dashboard | Built with Streamlit</p>
        <p>Data source: IMDb Non-Commercial Datasets</p>
    </div>
    """, unsafe_allow_html=True)
