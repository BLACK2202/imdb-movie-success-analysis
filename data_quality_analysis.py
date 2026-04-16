
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("IMDb DATA QUALITY ANALYSIS")
print("=" * 80)

try:
    print("\n📂 Loading data in chunks...")
    target_types = ['movie', 'tvSeries', 'tvMiniSeries', 'tvMovie', 'tvSpecial', 'video']
    chunks = []
    
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
    
    print(f"✓ Data loaded successfully!")
    print(f"  - Total records: {len(df):,}")
    print(f"  - Total columns: {len(df.columns)}")
    

    print("\n" + "=" * 80)
    print("1. MISSING VALUES ANALYSIS")
    print("=" * 80)
    
    missing_data = pd.DataFrame({
        'Column': df.columns,
        'Missing Count': df.isnull().sum().values,
        'Missing %': (df.isnull().sum().values / len(df) * 100).round(2)
    })
    missing_data = missing_data[missing_data['Missing Count'] > 0].sort_values('Missing %', ascending=False)
    
    if len(missing_data) > 0:
        print("\n🔴 COLUMNS WITH MISSING VALUES:")
        print(missing_data.to_string(index=False))
    else:
        print("\n✅ No missing values found in the dataset!")
    
    print("\n" + "=" * 80)
    print("2. OUTLIERS ANALYSIS (Using IQR Method)")
    print("=" * 80)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    outlier_summary = []
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_count = len(outliers)
        outlier_pct = (outlier_count / len(df) * 100)
        
        outlier_summary.append({
            'Column': col,
            'Outlier Count': outlier_count,
            'Outlier %': outlier_pct,
            'Lower Bound': round(lower_bound, 2),
            'Upper Bound': round(upper_bound, 2),
            'Min': df[col].min(),
            'Max': df[col].max()
        })
    
    outlier_df = pd.DataFrame(outlier_summary).sort_values('Outlier %', ascending=False)
    
    print("\n📊 OUTLIER STATISTICS BY COLUMN:")
    print(outlier_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("3. DETAILED STATISTICS FOR NUMERIC COLUMNS")
    print("=" * 80)
    
    print("\n📈 AVERAGE RATING:")
    print(f"  Mean:      {df['averageRating'].mean():.2f}")
    print(f"  Median:    {df['averageRating'].median():.2f}")
    print(f"  Std Dev:   {df['averageRating'].std():.2f}")
    print(f"  Min:       {df['averageRating'].min():.2f}")
    print(f"  Max:       {df['averageRating'].max():.2f}")
    print(f"  Range:     {df['averageRating'].max() - df['averageRating'].min():.2f}")
    
    print("\n📊 NUMBER OF VOTES:")
    print(f"  Mean:      {df['numVotes'].mean():,.0f}")
    print(f"  Median:    {df['numVotes'].median():,.0f}")
    print(f"  Std Dev:   {df['numVotes'].std():,.0f}")
    print(f"  Min:       {df['numVotes'].min():,.0f}")
    print(f"  Max:       {df['numVotes'].max():,.0f}")
    print(f"  Range:     {df['numVotes'].max() - df['numVotes'].min():,.0f}")
    
    if 'startYear' in df.columns:
        print("\n📅 START YEAR:")
        print(f"  Mean:      {df['startYear'].mean():.0f}")
        print(f"  Median:    {df['startYear'].median():.0f}")
        print(f"  Std Dev:   {df['startYear'].std():.2f}")
        print(f"  Min:       {df['startYear'].min():.0f}")
        print(f"  Max:       {df['startYear'].max():.0f}")
        print(f"  Range:     {df['startYear'].max() - df['startYear'].min():.0f}")
    

    print("\n" + "=" * 80)
    print("4. CATEGORICAL DATA ANALYSIS")
    print("=" * 80)
    
    print("\n🎬 TITLE TYPE DISTRIBUTION:")
    title_type_dist = df['titleType'].value_counts()
    for title_type, count in title_type_dist.items():
        pct = (count / len(df) * 100)
        print(f"  {title_type:20s} {count:10,} ({pct:5.2f}%)")
    
    print("\n" + "=" * 80)
    print("5. SUSPICIOUS PATTERNS & POTENTIAL ISSUES")
    print("=" * 80)
    
    issues = []
    
    zero_ratings = len(df[df['averageRating'] == 0])
    if zero_ratings > 0:
        issues.append(f"⚠️  {zero_ratings:,} titles with 0 rating")
    
    extreme_ratings = len(df[(df['averageRating'] > 9.5)])
    if extreme_ratings > 0:
        issues.append(f"⚠️  {extreme_ratings:,} titles with rating > 9.5")
    
    single_vote = len(df[df['numVotes'] == 1])
    if single_vote > 0:
        issues.append(f"⚠️  {single_vote:,} titles with only 1 vote")
    
    future_years = len(df[df['startYear'] > 2026])
    if future_years > 0:
        issues.append(f"⚠️  {future_years:,} titles with future dates (> 2026)")
    
    very_old = len(df[df['startYear'] < 1900])
    if very_old > 0:
        issues.append(f"⚠️  {very_old:,} titles before 1900 (may be data errors)")
    
    missing_genres = df['genres'].isnull().sum()
    if missing_genres > 0:
        issues.append(f"⚠️  {missing_genres:,} titles with missing genres")
    
    if issues:
        print("\n🔍 FOUND POTENTIAL ISSUES:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✅ No major suspicious patterns detected!")
    

    print("\n" + "=" * 80)
    print("6. DATA QUALITY ASSESSMENT")
    print("=" * 80)
    
    completeness = (1 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))) * 100
    outlier_ratio = (outlier_df['Outlier Count'].sum() / (len(df) * len(numeric_cols)) * 100)
    
    print(f"\n📋 OVERALL DATA QUALITY:")
    print(f"  Completeness:      {completeness:.2f}%")
    print(f"  Outlier Ratio:     {outlier_ratio:.2f}%")
    print(f"  Total Records:     {len(df):,}")
    print(f"  Total Columns:     {len(df.columns)}")
    
    if completeness > 95 and outlier_ratio < 5:
        quality = "EXCELLENT ✅"
    elif completeness > 90 and outlier_ratio < 10:
        quality = "GOOD ✅"
    elif completeness > 80 and outlier_ratio < 15:
        quality = "ACCEPTABLE ⚠️"
    else:
        quality = "NEEDS ATTENTION 🔴"
    
    print(f"\n  Overall Quality:   {quality}")
    

    print("\n" + "=" * 80)
    print("7. RECOMMENDATIONS")
    print("=" * 80)
    
    recommendations = []
    
    if missing_genres > 0:
        recommendations.append("• Fill missing genres or exclude affected titles from genre analysis")
    
    if single_vote > 0:
        recommendations.append("• Consider filtering out titles with < 100 votes for more reliable ratings")
    
    if very_old > 0 or future_years > 0:
        recommendations.append("• Validate and clean dates outside the reasonable range (1900-2026)")
    
    if outlier_ratio > 10:
        recommendations.append("• Investigate outliers - they may represent truly exceptional cases or data errors")
    
    if not recommendations:
        recommendations.append("✅ Data looks clean! No major recommendations.")
    
    print("\n")
    for rec in recommendations:
        print(f"  {rec}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80 + "\n")

except FileNotFoundError as e:
    print(f"❌ Error: Data files not found!")
    print(f"   Make sure 'title.basics.tsv' and 'title.ratings.tsv' are in the current directory.")
except Exception as e:
    print(f"❌ An error occurred: {e}")
    import traceback
    traceback.print_exc()
