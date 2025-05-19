import pandas as pd
import sys
import os
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the models
from models.bert_model import BertModel

def train_bert_model(data_path, output_path, epochs=3, batch_size=8):
    """Train and save the BERT model"""
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    print(f"Initializing BERT model...")
    bert_model = BertModel()
    
    print(f"Fine-tuning BERT on {len(df)} reviews for {epochs} epochs...")
    bert_model.fine_tune(
        df, 
        text_col='Review', 
        rating_col='Rating', 
        epochs=epochs,
        batch_size=batch_size
    )
    
    print(f"Saving model to {output_path}...")
    bert_model.save(output_path)
    print("Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train sentiment analysis models')
    parser.add_argument('--data', type=str, default='data/tripadvisor_hotel_reviews.csv',
                        help='Path to the dataset CSV file')
    parser.add_argument('--output', type=str, default='models/fine_tuned_bert.pth',
                        help='Path to save the trained model')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Training batch size')
                        
    args = parser.parse_args()
    
    train_bert_model(args.data, args.output, args.epochs, args.batch_size)