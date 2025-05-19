from flask import Flask, render_template, request, jsonify
import pandas as pd
import sys
import os

# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Now import the models (use absolute imports, not relative)
from models.vader_model import VaderModel
from models.bert_model import BertModel 
from models.hybrid_model import HybridModel

app = Flask(__name__)

# Initialize models
vader_model = VaderModel()
bert_model = None  # Will be initialized on demand to save memory
hybrid_model = None  # Will be initialized on demand to save memory

# Load a sample of the dataset
try:
    df = pd.read_csv('../data/tripadvisor_hotel_reviews.csv')
    sample_reviews = df.sample(100, random_state=42)['Review'].tolist()
except Exception as e:
    print(f"Error loading dataset: {e}")
    sample_reviews = [
        "The hotel was beautiful and the staff was very friendly. Great location!",
        "Terrible experience. Room was dirty and staff was rude.",
        "Average hotel, nothing special but clean and functional."
    ]

@app.route('/')
def index():
    return render_template('index.html', sample_reviews=sample_reviews[:10])

@app.route('/analyze', methods=['POST'])
def analyze():
    # Get data from request
    data = request.get_json()
    text = data.get('text', '')
    model_type = data.get('model', 'vader')
    
    # Check if text is provided
    if not text:
        return jsonify({
            'error': 'No text provided'
        }), 400
    
    # Analyze text using the selected model
    try:
        if model_type == 'vader':
            result = vader_model.analyze(text)
        elif model_type == 'bert':
            # Initialize BERT model if not already initialized
            global bert_model
            if bert_model is None:
                print("Initializing BERT model...")
                bert_model = BertModel()
            result = bert_model.analyze(text)
        elif model_type == 'hybrid':
            # Initialize Hybrid model if not already initialized
            global hybrid_model
            if hybrid_model is None:
                print("Initializing Hybrid model...")
                hybrid_model = HybridModel()
            result = hybrid_model.analyze(text)
        else:
            return jsonify({
                'error': f'Invalid model type: {model_type}'
            }), 400
        
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'error': f'Error analyzing text: {str(e)}'
        }), 500

@app.route('/compare', methods=['POST'])
def compare():
    # Get data from request
    data = request.get_json()
    text = data.get('text', '')
    
    # Check if text is provided
    if not text:
        return jsonify({
            'error': 'No text provided'
        }), 400
    
    # Analyze text using all models
    try:
        # VADER analysis
        vader_result = vader_model.analyze(text)
        
        # BERT analysis
        global bert_model
        if bert_model is None:
            print("Initializing BERT model...")
            bert_model = BertModel()
        bert_result = bert_model.analyze(text)
        
        # Hybrid analysis
        global hybrid_model
        if hybrid_model is None:
            print("Initializing Hybrid model...")
            hybrid_model = HybridModel()
        hybrid_result = hybrid_model.analyze(text)
        
        return jsonify({
            'vader': vader_result,
            'bert': bert_result,
            'hybrid': hybrid_result
        })
    except Exception as e:
        return jsonify({
            'error': f'Error comparing models: {str(e)}'
        }), 500

@app.route('/stats')
def stats():
    """Return some basic statistics about the dataset"""
    try:
        # Load the full dataset
        df = pd.read_csv('../data/tripadvisor_hotel_reviews.csv')
        
        # Calculate basic statistics
        total_reviews = len(df)
        avg_rating = df['Rating'].mean()
        rating_counts = df['Rating'].value_counts().sort_index().to_dict()
        
        # Calculate review length statistics
        df['review_length'] = df['Review'].str.len()
        avg_length = df['review_length'].mean()
        
        # Analyze sentiment of a random sample
        sample_df = df.sample(min(100, len(df)), random_state=42)
        vader_results = vader_model.analyze_batch(sample_df['Review'].tolist())
        sentiment_counts = {}
        for result in vader_results:
            sentiment = result['sentiment']
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
        
        return jsonify({
            'total_reviews': total_reviews,
            'average_rating': avg_rating,
            'rating_distribution': rating_counts,
            'average_review_length': avg_length,
            'sentiment_distribution': sentiment_counts
        })
    except Exception as e:
        return jsonify({
            'error': f'Error fetching stats: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(debug=True)