# Travel Sentiment Analysis

A simple sentiment analysis system for hotel reviews using VADER, BERT, and a hybrid approach.

## Overview

This project implements three sentiment analysis models to analyze hotel reviews:

1. **VADER Model**: A fast, lexicon-based approach that works well for simple text
2. **BERT Model**: A deep learning model that understands context but requires more computational resources
3. **Hybrid Model**: Combines both approaches, using VADER for short texts and BERT for complex reviews

The project includes a simple web interface for analyzing reviews and comparing the different models.

## Dataset

The system uses the TripAdvisor Hotel Reviews dataset, which contains reviews with ratings from 1-5 stars.

## Project Structure

```
travel-sentiment-analysis/
│
├── models/                       # Sentiment analysis models
│   ├── vader_model.py            # VADER-only implementation
│   ├── bert_model.py             # BERT-only implementation
│   └── hybrid_model.py           # Combined VADER+BERT model
│
├── app/                          # Simple web application
│   ├── static/                   # CSS, JS files (not used in this simple version)
│   ├── templates/                # HTML templates
│   │   └── index.html            # Main application page
│   └── app.py                    # Flask web application
│
├── data/                         # Dataset
│   └── tripadvisor_hotel_reviews.csv  # TripAdvisor hotel reviews
│
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation
```

## Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/travel-sentiment-analysis.git
   cd travel-sentiment-analysis
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Place the dataset:**
   - Put the `tripadvisor_hotel_reviews.csv` file in the `data/` directory

## Running the Application

1. **Start the Flask server:**
   ```bash
   cd app
   python app.py
   ```

2. **Access the web interface:**
   - Open your browser and navigate to http://localhost:5000

## Using the Interface

1. **Input a hotel review:** 
   - Type in the textbox or select one of the sample reviews
   
2. **Choose a model:**
   - VADER (fast, rule-based)
   - BERT (deep learning, context-aware)
   - Hybrid (combines both approaches)
   
3. **Analyze or Compare:**
   - Click "Analyze Sentiment" to use the selected model
   - Click "Compare All Models" to see results from all three models

## Model Details

### VADER (Valence Aware Dictionary and Sentiment Reasoner)
- Lexicon and rule-based sentiment analysis
- Specialized for social media content and short texts
- Fast, doesn't require training
- Good at handling emoticons, slang, and simple sentiment expressions

### BERT (Bidirectional Encoder Representations from Transformers)
- Deep learning transformer model
- Understands context and word relationships
- More accurate for complex linguistic patterns
- Computationally intensive

### Hybrid Approach
- Uses VADER for short texts (<100 characters)
- Uses BERT for longer, more complex texts
- Weighted combination based on text length
- Prioritizes BERT when it has high confidence