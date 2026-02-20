import pandas as pd
import numpy as np

# Using chunksize or low_memory=False depending on RAM, user used low_memory=False
try:
    basics = pd.read_csv("title.basics.tsv", sep="\t", low_memory=False)
    ratings = pd.read_csv("title.ratings.tsv", sep="\t", low_memory=False)
except FileNotFoundError:
    print("Error: .tsv files not found in the current directory.")
    exit()

print("\n--- Data Inspection ---")
print(basics.head())
print(basics.info())

# Display distribution of all title types before filtering
print("\n--- Original Title Types Distribution ---")
print(basics['titleType'].value_counts())
#blaset types mteena movies we use el definitions el mawjoudin
target_types = [
    'movie',          # Feature films
    'tvSeries',       # TV Series
    'tvMiniSeries',   # Mini-series
    'tvMovie',        # TV Movies
    'tvSpecial',      # TV Specials
    'video'           # Direct-to-video content
]

print(f"\nFiltering data to keep only: {target_types}")

#filtriw el dataset
basics_filtered = basics[basics["titleType"].isin(target_types)]

print("\n--- Filtered Title Types Distribution ---")
print(basics_filtered['titleType'].value_counts())

print(f"\nRemaining records: {len(basics_filtered)}")

#merge the dataset
df = pd.merge(basics_filtered, ratings, on='tconst')
df.head()

np.random.seed(42)


