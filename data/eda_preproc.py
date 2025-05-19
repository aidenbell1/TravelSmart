import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('tripadvisor_hotel_reviews.csv')

# Display basic information
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nColumn information:")
print(df.info())

print("\nSummary statistics:")
print(df.describe())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Exploratory Data Analysis

plt.figure(figsize=(10, 6))
rating_counts = df['Rating'].value_counts().sort_index()
sns.barplot(x=rating_counts.index, y=rating_counts.values)
plt.title('Distribution of Ratings', fontsize=16)
plt.xlabel('Rating', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Add count labels
for i, count in enumerate(rating_counts.values):
    plt.text(i, count + 100, f"{count} ({count/len(df)*100:.1f}%)", 
             ha='center', fontsize=12)

plt.tight_layout()
plt.show()

df['review_length'] = df['Review'].apply(len)

# Summary statistics for review length
print("\nReview Length Statistics:")
print(df['review_length'].describe())

plt.figure(figsize=(12, 6))
sns.boxplot(x='Rating', y='review_length', data=df)
plt.title('Review Length by Rating', fontsize=16)
plt.xlabel('Rating', fontsize=14)
plt.ylabel('Review Length (characters)', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()

avg_length = df.groupby('Rating')['review_length'].mean().reset_index()
median_length = df.groupby('Rating')['review_length'].median().reset_index()
median_length.columns = ['Rating', 'median_length']
length_stats = pd.merge(avg_length, median_length, on='Rating')

plt.figure(figsize=(12, 6))
sns.barplot(x='Rating', y='review_length', data=avg_length, color='skyblue')
plt.title('Average Review Length by Rating', fontsize=16)
plt.xlabel('Rating', fontsize=14)
plt.ylabel('Average Length (characters)', fontsize=14)

# Add average values
for i, row in avg_length.iterrows():
    plt.text(i, row['review_length'] + 20, f"{row['review_length']:.0f}", 
             ha='center', fontsize=12)

plt.tight_layout()
plt.show()

print("\nReview Length Statistics by Rating:")
print(length_stats)

Q1 = df['review_length'].quantile(0.25)
Q3 = df['review_length'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers = df[(df['review_length'] < lower_bound) | (df['review_length'] > upper_bound)]
print(f"\nNumber of outliers: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")
print(f"Lower bound: {lower_bound:.2f}, Upper bound: {upper_bound:.2f}")

df['sentiment'] = df['Rating'].apply(lambda x: 'Negative' if x <= 2 else ('Neutral' if x == 3 else 'Positive'))
sentiment_counts = df['sentiment'].value_counts()

plt.figure(figsize=(10, 6))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette=['#ff9999', '#ffcc99', '#99cc99'])
plt.title('Sentiment Distribution', fontsize=16)
plt.xlabel('Sentiment', fontsize=14)
plt.ylabel('Count', fontsize=14)

# Add count labels
for i, count in enumerate(sentiment_counts.values):
    plt.text(i, count + 100, f"{count} ({count/len(df)*100:.1f}%)", 
             ha='center', fontsize=12)

plt.tight_layout()
plt.show()

def clean_text(text):
    """Basic text cleaning"""
    text = text.lower() 
    text = re.sub(r'\s+', ' ', text) 
    return text.strip()

def tokenize_text(text):
    """Tokenize text into words"""
    return word_tokenize(text)
# Apply text preprocessing
df['cleaned_review'] = df['Review'].apply(clean_text)

def get_common_words(texts, min_length=4, top_n=20):
    """Extract most common words from a list of texts"""
    all_words = []
    
    for text in texts:
        words = text.split()
        # Filter out short words
        words = [word for word in words if len(word) >= min_length]
        all_words.extend(words)
    
    # Count word frequencies
    word_counts = Counter(all_words)
    
    # Get top N words
    top_words = word_counts.most_common(top_n)
    return top_words

# Get top words overall
top_words = get_common_words(df['cleaned_review'])
top_words_df = pd.DataFrame(top_words, columns=['word', 'count'])

plt.figure(figsize=(12, 8))
sns.barplot(x='count', y='word', data=top_words_df)
plt.title('Most Common Words in Reviews', fontsize=16)
plt.xlabel('Count', fontsize=14)
plt.ylabel('Word', fontsize=14)
plt.tight_layout()
plt.show()

# Create a word frequency dictionary for each rating
word_freq_by_rating = {}
for rating in range(1, 6):
    rating_reviews = df[df['Rating'] == rating]['cleaned_review']
    word_freq_by_rating[rating] = get_common_words(rating_reviews, top_n=10)

# Display top words for each rating
for rating, words in word_freq_by_rating.items():
    print(f"\nTop 10 words in {rating}-star reviews:")
    for word, count in words:
        print(f"  {word}: {count}")
# Create bins for review length
bins = [0, 500, 1000, 2000, df['review_length'].max()]
labels = ['0-500 chars', '501-1000 chars', '1001-2000 chars', '2001+ chars']
df['length_bin'] = pd.cut(df['review_length'], bins=bins, labels=labels)

# Calculate average rating by length bin
bin_stats = df.groupby('length_bin').agg({
    'Rating': ['mean', 'count']
}).reset_index()
bin_stats.columns = ['length_bin', 'avg_rating', 'count']

plt.figure(figsize=(12, 6))
ax1 = plt.subplot(111)
sns.barplot(x='length_bin', y='avg_rating', data=bin_stats, ax=ax1, color='skyblue')
ax1.set_ylabel('Average Rating', fontsize=14)
ax1.set_ylim(1, 5)

ax2 = ax1.twinx()
ax2.plot(range(len(bin_stats)), bin_stats['count'], 'ro-', linewidth=2, markersize=8)
ax2.set_ylabel('Number of Reviews', fontsize=14, color='r')

plt.title('Average Rating by Review Length', fontsize=16)
plt.xlabel('Review Length', fontsize=14)
plt.tight_layout()
plt.show()
# Save the cleaned dataset
df.to_csv('cleaned_tripadvisor_hotel_reviews.csv', index=False)
print("\nCleaned dataset saved as 'cleaned_tripadvisor_hotel_reviews.csv'")
