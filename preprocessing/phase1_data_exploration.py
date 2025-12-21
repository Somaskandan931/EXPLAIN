"""
Explainable Multilingual Fake News Detection System
Phase 1: Dataset Exploration & Preparation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

#=============================================================================
# STEP 1: LOAD ALL DATASETS
#=============================================================================

print("="*70)
print("STEP 1: Loading Datasets")
print("="*70)

# Update these paths to match your downloaded dataset locations
BHARAT_PATH = "C:/Users/somas/PycharmProjects/EXPLAIN/data/bharatfakenewskosh.csv"  # Update this
IFND_PATH = "C:/Users/somas/PycharmProjects/EXPLAIN/data/IFND.csv"          # Update this
FAKE_REAL_PATH = "C:/Users/somas/PycharmProjects/EXPLAIN/data/news_dataset.csv"   # Update this

# Load datasets
try:
    df_bharat = pd.read_csv(BHARAT_PATH)
    print(f"✓ BharatFakeNewsKosh loaded: {df_bharat.shape}")
except Exception as e:
    print(f"✗ Error loading BharatFakeNewsKosh: {e}")
    df_bharat = None

try:
    df_ifnd = pd.read_csv(IFND_PATH)
    print(f"✓ IFND Dataset loaded: {df_ifnd.shape}")
except Exception as e:
    print(f"✗ Error loading IFND: {e}")
    df_ifnd = None

try:
    df_fake_real = pd.read_csv(FAKE_REAL_PATH)
    print(f"✓ Fake-Real News loaded: {df_fake_real.shape}")
except Exception as e:
    print(f"✗ Error loading Fake-Real News: {e}")
    df_fake_real = None

#=============================================================================
# STEP 2: EXPLORE EACH DATASET
#=============================================================================

print("\n" + "="*70)
print("STEP 2: Dataset Exploration")
print("="*70)

def explore_dataset(df, name):
    """Explore a single dataset"""
    if df is None:
        print(f"\n{name}: Not loaded")
        return

    print(f"\n--- {name} ---")
    print(f"Shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head(2))
    print(f"\nData types:")
    print(df.dtypes)
    print(f"\nMissing values:")
    print(df.isnull().sum())
    print(f"\nLabel distribution:")
    if 'label' in df.columns:
        print(df['label'].value_counts())
    elif 'category' in df.columns:
        print(df['category'].value_counts())

# Explore each dataset
if df_bharat is not None:
    explore_dataset(df_bharat, "BharatFakeNewsKosh")
if df_ifnd is not None:
    explore_dataset(df_ifnd, "IFND Dataset")
if df_fake_real is not None:
    explore_dataset(df_fake_real, "Fake-Real News")

#=============================================================================
# STEP 3: STANDARDIZE DATASETS
#=============================================================================

print("\n" + "="*70)
print("STEP 3: Standardizing Datasets")
print("="*70)

def standardize_dataset(df, name, text_col, label_col):
    """
    Standardize dataset to common format: ['text', 'label']
    label: 0 = Real, 1 = Fake
    """
    if df is None:
        return None

    print(f"\nStandardizing {name}...")

    # Create standardized dataframe
    df_std = pd.DataFrame()

    # Extract text column
    if text_col in df.columns:
        df_std['text'] = df[text_col]
    else:
        print(f"Warning: Text column '{text_col}' not found in {name}")
        return None

    # Extract and standardize labels
    if label_col in df.columns:
        # Handle boolean labels (BharatFakeNewsKosh uses boolean)
        if df[label_col].dtype == bool:
            df_std['label'] = df[label_col].astype(int)  # False=0 (Real), True=1 (Fake)
        else:
            # Map various label formats to binary (0=Real, 1=Fake)
            labels = df[label_col].astype(str).str.lower().str.strip()
            df_std['label'] = labels.map({
                'real': 0, 'true': 0, '0': 0, 'legitimate': 0,
                'fake': 1, 'false': 1, '1': 1, 'misleading': 1
            })
    else:
        print(f"Warning: Label column '{label_col}' not found in {name}")

    # Remove rows with missing text or labels
    df_std = df_std.dropna(subset=['text', 'label'])

    # Remove empty strings
    df_std = df_std[df_std['text'].str.strip() != '']

    # Add source column for tracking
    df_std['source'] = name

    print(f"✓ Standardized shape: {df_std.shape}")
    print(f"  Label distribution: Fake={df_std['label'].sum()}, Real={(df_std['label']==0).sum()}")

    return df_std

# Standardize each dataset with correct column names

# BharatFakeNewsKosh: Use 'Statement' for text, 'Label' for label
df_bharat_std = standardize_dataset(
    df_bharat,
    "BharatFakeNewsKosh",
    text_col='Statement',
    label_col='Label'
)

# IFND: Use 'Statement' for text, 'Label' for label
df_ifnd_std = standardize_dataset(
    df_ifnd,
    "IFND",
    text_col='Statement',
    label_col='Label'
)

# Fake-Real News: Use 'text' for text, 'label' for label
df_fake_real_std = standardize_dataset(
    df_fake_real,
    "FakeReal",
    text_col='text',
    label_col='label'
)

#=============================================================================
# STEP 4: COMBINE DATASETS
#=============================================================================

print("\n" + "="*70)
print("STEP 4: Combining Datasets")
print("="*70)

# Combine all standardized datasets
dfs_to_combine = [df for df in [df_bharat_std, df_ifnd_std, df_fake_real_std] if df is not None]

if len(dfs_to_combine) > 0:
    df_combined = pd.concat(dfs_to_combine, ignore_index=True)
    print(f"\n✓ Combined dataset shape: {df_combined.shape}")
    print(f"\nLabel distribution:")
    print(df_combined['label'].value_counts())
    print(f"\nSource distribution:")
    print(df_combined['source'].value_counts())
else:
    print("✗ No datasets to combine!")
    df_combined = None

#=============================================================================
# STEP 5: DATA QUALITY CHECKS
#=============================================================================

if df_combined is not None:
    print("\n" + "="*70)
    print("STEP 5: Data Quality Checks")
    print("="*70)

    # Text length statistics
    df_combined['text_length'] = df_combined['text'].str.len()
    df_combined['word_count'] = df_combined['text'].str.split().str.len()

    print(f"\nText length statistics:")
    print(df_combined['text_length'].describe())

    print(f"\nWord count statistics:")
    print(df_combined['word_count'].describe())

    # Visualize distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Label distribution
    df_combined['label'].value_counts().plot(kind='bar', ax=axes[0], color=['green', 'red'])
    axes[0].set_title('Label Distribution')
    axes[0].set_xlabel('Label (0=Real, 1=Fake)')
    axes[0].set_ylabel('Count')

    # Text length distribution
    df_combined.boxplot(column='text_length', by='label', ax=axes[1])
    axes[1].set_title('Text Length by Label')
    axes[1].set_xlabel('Label (0=Real, 1=Fake)')

    # Source distribution
    df_combined['source'].value_counts().plot(kind='bar', ax=axes[2], color='steelblue')
    axes[2].set_title('Source Distribution')
    axes[2].set_xlabel('Source')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig('dataset_overview.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved visualization: dataset_overview.png")

    # Remove very short texts (likely noise)
    min_length = 10
    df_combined = df_combined[df_combined['text_length'] >= min_length]
    print(f"\n✓ Filtered texts shorter than {min_length} chars: {df_combined.shape}")

#=============================================================================
# STEP 6: SAVE PROCESSED DATASET
#=============================================================================

if df_combined is not None:
    print("\n" + "="*70)
    print("STEP 6: Saving Processed Dataset")
    print("="*70)

    # Create output directory
    output_dir = Path("processed_data")
    output_dir.mkdir(exist_ok=True)

    # Save combined dataset
    output_path = output_dir / "combined_dataset.csv"
    df_combined[['text', 'label', 'source']].to_csv(output_path, index=False)
    print(f"\n✓ Saved combined dataset: {output_path}")
    print(f"  Total samples: {len(df_combined)}")
    print(f"  Fake news: {(df_combined['label']==1).sum()}")
    print(f"  Real news: {(df_combined['label']==0).sum()}")

print("\n" + "="*70)
print("Dataset Preparation Complete!")
print("="*70)
print("\nNext Steps:")
print("1. Update the file paths in this script to match your dataset locations")
print("2. Update the column names (text_col, label_col) based on your datasets")
print("3. Run this script to create combined_dataset.csv")
print("4. Proceed to language detection and preprocessing")