# Hybrid VADER+BERT Sentiment Analysis Model

import torch
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Import VADER and BERT models
from models.vader_model import VaderModel
from models.bert_model import BertModel

class HybridModel:
    """Combined VADER+BERT sentiment analysis model for hotel reviews"""
    
    def __init__(self, use_pretrained_bert=True, device=None):
        # Initialize VADER model
        self.vader_model = VaderModel()
        
        # Initialize BERT model
        self.bert_model = BertModel(device=device) if use_pretrained_bert else None
        
        # Set default weights
        self.short_text_threshold = 100  # character threshold for short vs long text
        self.vader_weight_short = 0.7    # VADER weight for short texts
        self.vader_weight_long = 0.3     # VADER weight for long texts
        self.high_confidence_threshold = 0.8  # threshold for high confidence
    
    def analyze(self, text):
        """Analyze sentiment using both VADER and BERT"""
        if not isinstance(text, str) or not text.strip():
            return {
                'sentiment': 'neutral',
                'score': 0,
                'explanation': 'Empty or invalid text'
            }
        
        # Get VADER results
        vader_result = self.vader_model.analyze(text)
        
        # For short texts or if BERT is not available, return VADER result with adjusted explanation
        if self.bert_model is None or len(text) < 20:
            vader_result['explanation'] = f"Using VADER only: {vader_result['explanation']}"
            return vader_result
        
        # Get BERT results
        bert_result = self.bert_model.analyze(text)
        
        # Determine if this is a short text
        is_short_text = len(text) < self.short_text_threshold
        
        # Determine weights
        vader_weight = self.vader_weight_short if is_short_text else self.vader_weight_long
        bert_weight = 1 - vader_weight
        
        # If both agree, use that sentiment
        if vader_result['sentiment'] == bert_result['sentiment']:
            final_sentiment = vader_result['sentiment']
            confidence = (vader_result['score'] + bert_result['score']) / 2
            explanation = f"Both VADER and BERT agree: {final_sentiment} sentiment"
        else:
            # If BERT has high confidence, trust it more for complex text
            if bert_result['score'] > self.high_confidence_threshold and not is_short_text:
                final_sentiment = bert_result['sentiment']
                confidence = bert_result['score']
                explanation = f"Using BERT (high confidence): {final_sentiment} sentiment"
            # Otherwise, use VADER for short text and BERT for long text
            else:
                if is_short_text:
                    final_sentiment = vader_result['sentiment']
                    confidence = vader_result['score']
                    explanation = f"Using VADER for short text: {final_sentiment} sentiment"
                else:
                    final_sentiment = bert_result['sentiment']
                    confidence = bert_result['score']
                    explanation = f"Using BERT for long text: {final_sentiment} sentiment"
        
        return {
            'sentiment': final_sentiment,
            'score': confidence,
            'explanation': explanation,
            'details': {
                'vader_result': vader_result,
                'bert_result': bert_result,
                'is_short_text': is_short_text,
                'vader_weight': vader_weight,
                'bert_weight': bert_weight
            }
        }
    
    def analyze_batch(self, texts, batch_size=8):
        """Analyze sentiment for a batch of texts"""
        # For very short texts, use only VADER (faster)
        vader_only_indices = [i for i, text in enumerate(texts) 
                            if not isinstance(text, str) or len(text) < 20]
        
        # Get VADER results for all texts (VADER is fast)
        vader_results = self.vader_model.analyze_batch(texts)
        
        # If BERT is not available or all texts are very short, return VADER results
        if self.bert_model is None or all(i < len(texts) for i in vader_only_indices):
            for result in vader_results:
                result['explanation'] = f"Using VADER only: {result['explanation']}"
            return vader_results
        
        # Get BERT results for non-short texts
        if vader_only_indices:
            bert_texts = [text for i, text in enumerate(texts) if i not in vader_only_indices]
            bert_indices = [i for i in range(len(texts)) if i not in vader_only_indices]
            bert_results = self.bert_model.analyze_batch(bert_texts, batch_size=batch_size)
        else:
            bert_results = self.bert_model.analyze_batch(texts, batch_size=batch_size)
            bert_indices = list(range(len(texts)))
        
        # Combine results
        results = []
        bert_result_idx = 0
        
        for i, vader_result in enumerate(vader_results):
            # If very short text, use VADER only
            if i in vader_only_indices:
                vader_result['explanation'] = f"Using VADER only (short text): {vader_result['explanation']}"
                results.append(vader_result)
                continue
            
            # Get BERT result
            bert_result = bert_results[bert_result_idx]
            bert_result_idx += 1
            
            text = texts[i]
            
            # Determine if this is a short text
            is_short_text = len(text) < self.short_text_threshold if isinstance(text, str) else True
            
            # Determine weights
            vader_weight = self.vader_weight_short if is_short_text else self.vader_weight_long
            bert_weight = 1 - vader_weight
            
            # If both agree, use that sentiment
            if vader_result['sentiment'] == bert_result['sentiment']:
                final_sentiment = vader_result['sentiment']
                confidence = (vader_result['score'] + bert_result['score']) / 2
                explanation = f"Both VADER and BERT agree: {final_sentiment} sentiment"
            else:
                # If BERT has high confidence, trust it more for complex text
                if bert_result['score'] > self.high_confidence_threshold and not is_short_text:
                    final_sentiment = bert_result['sentiment']
                    confidence = bert_result['score']
                    explanation = f"Using BERT (high confidence): {final_sentiment} sentiment"
                # Otherwise, use VADER for short text and BERT for long text
                else:
                    if is_short_text:
                        final_sentiment = vader_result['sentiment']
                        confidence = vader_result['score']
                        explanation = f"Using VADER for short text: {final_sentiment} sentiment"
                    else:
                        final_sentiment = bert_result['sentiment']
                        confidence = bert_result['score']
                        explanation = f"Using BERT for long text: {final_sentiment} sentiment"
            
            results.append({
                'sentiment': final_sentiment,
                'score': confidence,
                'explanation': explanation,
                'details': {
                    'vader_result': vader_result,
                    'bert_result': bert_result,
                    'is_short_text': is_short_text,
                    'vader_weight': vader_weight,
                    'bert_weight': bert_weight
                }
            })
        
        return results
    
    def map_rating_to_sentiment(self, rating):
        """Map star rating to expected sentiment"""
        return self.vader_model.map_rating_to_sentiment(rating)
    
    def evaluate(self, df, text_col='Review', rating_col='Rating', batch_size=8):
        """Evaluate model performance against ratings"""
        # Ensure the dataframe has the required columns
        if text_col not in df.columns or rating_col not in df.columns:
            raise ValueError(f"DataFrame must contain columns: {text_col} and {rating_col}")
        
        # Get texts and expected sentiments
        texts = df[text_col].tolist()
        expected = [self.map_rating_to_sentiment(r) for r in df[rating_col]]
        
        # Get predictions
        results = self.analyze_batch(texts, batch_size=batch_size)
        predictions = [r['sentiment'] for r in results]
        
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
        """Save the hybrid model configuration"""
        config = {
            'short_text_threshold': self.short_text_threshold,
            'vader_weight_short': self.vader_weight_short,
            'vader_weight_long': self.vader_weight_long,
            'high_confidence_threshold': self.high_confidence_threshold
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(config, f)
    
    @classmethod
    def load(cls, filepath, vader_path=None, bert_path=None, device=None):
        """Load the hybrid model configuration"""
        # Create instance
        instance = cls(use_pretrained_bert=(bert_path is not None), device=device)
        
        # Load configuration
        with open(filepath, 'rb') as f:
            config = pickle.load(f)
        
        # Update instance configuration
        instance.short_text_threshold = config['short_text_threshold']
        instance.vader_weight_short = config['vader_weight_short']
        instance.vader_weight_long = config['vader_weight_long']
        instance.high_confidence_threshold = config['high_confidence_threshold']
        
        # Load VADER model if path provided
        if vader_path:
            instance.vader_model = VaderModel.load(vader_path)
        
        # Load BERT model if path provided
        if bert_path:
            instance.bert_model = BertModel.load(bert_path, device=device)
        
        return instance

