# 🎬 IMDb Movie Success Analysis

A comprehensive data analysis project that explores factors contributing to success on IMDb, featuring an interactive dashboard for visualizing insights.

## 📊 Project Overview

This project analyzes IMDb datasets to understand:

- **Rating Distributions**: How ratings are spread across different titles
- **Content Type Performance**: Comparative analysis of movies, TV series, and other formats
- **Popularity vs. Quality**: Relationship between number of votes and ratings
- **Temporal Trends**: How ratings and content production have changed over time
- **Genre Analysis**: Which genres perform best and why

## 🚀 Features

### Interactive Dashboard

- **Real-time Filtering**: Filter data by year range, rating, and vote count
- **Multiple Visualizations**: Interactive charts and plots using Plotly
- **Key Metrics**: At-a-glance statistics about the dataset
- **Genre Comparison**: Deep dive into genre performance
- **Data Explorer**: Browse and download filtered datasets

### Analysis Notebooks

- `notebooks/data.ipynb`: Data loading and preprocessing
- `notebooks/visualizations.ipynb`: Detailed visualization analysis

## 📋 Prerequisites

- Python 3.8 or higher
- IMDb dataset files:
  - `title.basics.tsv`
  - `title.ratings.tsv`

  Download from [IMDb Non-Commercial Datasets](https://datasets.imdbws.com/)

## 🔧 Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd imdb-movie-success-analysis
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

3. Download IMDb datasets and place them in the project root directory:
   - [title.basics.tsv.gz](https://datasets.imdbws.com/title.basics.tsv.gz)
   - [title.ratings.tsv.gz](https://datasets.imdbws.com/title.ratings.tsv.gz)
4. Extract the .gz files to get the .tsv files

## 🎯 Usage

### Running the Dashboard

Launch the interactive dashboard with:

```bash
streamlit run dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`

### Using the Dashboard

1. **Sidebar Controls**: Use filters to customize your view
   - Adjust year range
   - Set minimum rating threshold
   - Filter by minimum vote count

2. **Navigation Tabs**:
   - **Overview**: Rating distribution and statistics
   - **Content Types**: Compare performance across different formats
   - **Popularity vs Quality**: Explore the relationship between votes and ratings
   - **Temporal Trends**: See how things have changed over time
   - **Genre Analysis**: Discover top-performing genres
   - **Data Explorer**: Browse raw data and export to CSV

### Running Jupyter Notebooks

Start Jupyter and explore the analysis notebooks:

```bash
jupyter notebook
```

Navigate to the `notebooks/` directory and open:

- `data.ipynb` for data loading
- `visualizations.ipynb` for detailed visualizations

## 📁 Project Structure

```
imdb-movie-success-analysis/
│
├── dashboard.py              # Main Streamlit dashboard
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
│
├── notebooks/
│   ├── data.ipynb            # Data loading notebook
│   ├── visualizations.ipynb  # Visualization notebook
│   └── improved_data_loading.py  # Data loading utilities
│
└── title.basics.tsv          # IMDb basics dataset (download separately)
└── title.ratings.tsv         # IMDb ratings dataset (download separately)
```

## 📊 Dashboard Features

### Key Metrics Panel

- Total number of titles analyzed
- Average and median ratings
- Total votes across all titles
- Number of unique content types

### Visualizations

1. **Rating Distribution Histogram**
   - Shows the overall spread of ratings
   - Includes statistical summary

2. **Content Type Box Plots**
   - Compare rating distributions across content types
   - Identifies outliers and trends

3. **Popularity vs Quality Scatter Plot**
   - Interactive scatter with log scale
   - Color-coded by content type
   - Shows top-voted and top-rated titles

4. **Temporal Trends Line Charts**
   - Average ratings over time
   - Production volume changes
   - Decade-by-decade analysis

5. **Genre Performance Charts**
   - Top genres by average rating
   - Most produced genres
   - Interactive genre comparison

## 🔍 Data Processing

The dashboard implements efficient data loading:

- **Chunked Reading**: Processes large files in manageable chunks
- **Early Filtering**: Removes irrelevant data during load
- **Caching**: Uses Streamlit's caching for improved performance
- **Optimized Memory**: Loads only necessary columns

## 📈 Insights & Findings

Use the dashboard to discover:

- Which content types consistently receive higher ratings
- Whether popularity (votes) correlates with quality (ratings)
- How movie production and quality have evolved over decades
- Which genres dominate and which are underrated gems
- Statistical distributions and outliers in the data

## 🛠️ Troubleshooting

### Data Files Not Found

**Error**: "Data files not found!"
**Solution**: Ensure `title.basics.tsv` and `title.ratings.tsv` are in the same directory as `dashboard.py`

### Memory Issues

**Error**: Out of memory errors
**Solution**:

- The dashboard uses chunked loading
- Close other applications
- Consider sampling the data for analysis

### Package Installation Issues

**Error**: Pip install errors
**Solution**:

```bash
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Data provided by [IMDb](https://www.imdb.com/)
- Built with [Streamlit](https://streamlit.io/)
- Visualizations powered by [Plotly](https://plotly.com/)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or feedback, please open an issue in the repository.

---

**Note**: This project uses IMDb's non-commercial datasets. Please review IMDb's terms of use before using this data for any purpose.
