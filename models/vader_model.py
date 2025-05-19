# VADER Sentiment Analysis Model

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

class VaderModel:
    """VADER-only sentiment analysis model for hotel reviews"""
    
    def __init__(self):
    # Download VADER lexicon if not already downloaded, with SSL workaround
        try:
            nltk.data.find('vader_lexicon.zip')
        except LookupError:
            try:
                # SSL workaround for macOS
                import ssl
                try:
                    _create_unverified_https_context = ssl._create_unverified_context
                except AttributeError:
                    pass
                else:
                    ssl._create_default_https_context = _create_unverified_https_context
                
                nltk.download('vader_lexicon')
            except Exception as e:
                print(f"Error downloading VADER lexicon: {e}")
                print("Please manually download by running: python -m nltk.downloader vader_lexicon")
        
        # Initialize VADER
        self.analyzer = SentimentIntensityAnalyzer()
    
    def analyze(self, text):
        """Analyze sentiment of a single text"""
        if not isinstance(text, str) or not text.strip():
            return {
                'sentiment': 'neutral',
                'score': 0,
                'explanation': 'Empty or invalid text'
            }
        
        # Get VADER scores
        scores = self.analyzer.polarity_scores(text)
        
        # Determine sentiment based on compound score
        if scores['compound'] >= 0.05:
            sentiment = 'positive'
        elif scores['compound'] <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        # Create explanation
        if sentiment == 'positive':
            explanation = f"VADER found more positive than negative elements (compound score: {scores['compound']:.2f})"
        elif sentiment == 'negative':
            explanation = f"VADER found more negative than positive elements (compound score: {scores['compound']:.2f})"
        else:
            explanation = f"VADER found a balanced mix of elements (compound score: {scores['compound']:.2f})"
        
        return {
            'sentiment': sentiment,
            'score': scores['compound'],
            'explanation': explanation,
            'details': {
                'pos': scores['pos'],
                'neu': scores['neu'],
                'neg': scores['neg'],
                'compound': scores['compound']
            }
        }
    
    def analyze_batch(self, texts):
        """Analyze sentiment for a batch of texts"""
        return [self.analyze(text) for text in texts]
    
    def map_rating_to_sentiment(self, rating):
        """Map star rating to expected sentiment"""
        if rating >= 4:  # 4-5 stars = positive
            return 'positive'
        elif rating <= 2:  # 1-2 stars = negative
            return 'neutral' if rating == 2 else 'negative'
        else:  # 3 stars = neutral
            return 'neutral'
    
    def evaluate(self, df, text_col='Review', rating_col='Rating'):
        """Evaluate model performance against ratings"""
        # Ensure the dataframe has the required columns
        if text_col not in df.columns or rating_col not in df.columns:
            raise ValueError(f"DataFrame must contain columns: {text_col} and {rating_col}")
        
        # Create predicted and expected sentiment
        predictions = []
        expected = []
        
        # Analyze each review
        for _, row in df.iterrows():
            text = row[text_col]
            rating = row[rating_col]
            
            # Skip invalid entries
            if not isinstance(text, str) or not isinstance(rating, (int, float)):
                continue
                
            # Get prediction and expected sentiment
            result = self.analyze(text)
            predicted_sentiment = result['sentiment']
            expected_sentiment = self.map_rating_to_sentiment(rating)
            
            predictions.append(predicted_sentiment)
            expected.append(expected_sentiment)
        
        # Calculate metrics
        accuracy = accuracy_score(expected, predictions)
        conf_matrix = confusion_matrix(expected, predictions, labels=['positive', 'neutral', 'negative'])
        report = classification_report(expected, predictions, labels=['positive', 'neutral', 'negative'])
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': report,
            'predictions': predictions,
            'expected': expected
        }
    
    def save(self, filepath):
        """Save the model to a file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath):
        """Load the model from a file"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)