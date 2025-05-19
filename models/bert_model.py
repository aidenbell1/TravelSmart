# BERT Sentiment Analysis Model

import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

class BertModel:
    """BERT-only sentiment analysis model for hotel reviews"""
    
    def __init__(self, model_name='bert-base-uncased', device=None):
        # Set device (CPU or CUDA)
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pre-trained model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3  # positive, neutral, negative
        )
        
        # Move model to device
        self.model.to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Define label mapping
        self.id_to_label = {0: 'negative', 1: 'neutral', 2: 'positive'}
        self.label_to_id = {'negative': 0, 'neutral': 1, 'positive': 2}
    
    def analyze(self, text):
        """Analyze sentiment of a single text"""
        if not isinstance(text, str) or not text.strip():
            return {
                'sentiment': 'neutral',
                'score': 0.33,  # Default neutral confidence
                'explanation': 'Empty or invalid text'
            }
        
        # Tokenize and prepare input
        encoded_input = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding='max_length'
        )
        
        # Move input to device
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        
        # Forward pass
        with torch.no_grad():
            output = self.model(**encoded_input)
            logits = output.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
        
        # Get predicted label and confidence
        predicted_id = torch.argmax(probabilities).item()
        sentiment = self.id_to_label[predicted_id]
        confidence = probabilities[predicted_id].item()
        
        # Get all probabilities
        probs = {self.id_to_label[i]: prob.item() for i, prob in enumerate(probabilities)}
        
        # Create explanation
        explanation = f"BERT predicted {sentiment} sentiment with {confidence:.2f} confidence"
        
        return {
            'sentiment': sentiment,
            'score': confidence,
            'explanation': explanation,
            'details': {
                'probabilities': probs,
                'predicted_id': predicted_id
            }
        }
    
    def analyze_batch(self, texts, batch_size=8):
        """Analyze sentiment for a batch of texts"""
        results = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            valid_texts = [t if isinstance(t, str) else "" for t in batch_texts]
            
            # Tokenize and prepare input
            encoded_input = self.tokenizer(
                valid_texts,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding='max_length'
            )
            
            # Move input to device
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
            
            # Forward pass
            with torch.no_grad():
                output = self.model(**encoded_input)
                logits = output.logits
                probabilities = torch.nn.functional.softmax(logits, dim=1)
            
            # Process each result
            for j, probs in enumerate(probabilities):
                text = batch_texts[j]
                
                if not isinstance(text, str) or not text.strip():
                    results.append({
                        'sentiment': 'neutral',
                        'score': 0.33,
                        'explanation': 'Empty or invalid text'
                    })
                    continue
                
                # Get predicted label and confidence
                predicted_id = torch.argmax(probs).item()
                sentiment = self.id_to_label[predicted_id]
                confidence = probs[predicted_id].item()
                
                # Get all probabilities
                all_probs = {self.id_to_label[i]: prob.item() for i, prob in enumerate(probs)}
                
                # Create explanation
                explanation = f"BERT predicted {sentiment} sentiment with {confidence:.2f} confidence"
                
                results.append({
                    'sentiment': sentiment,
                    'score': confidence,
                    'explanation': explanation,
                    'details': {
                        'probabilities': all_probs,
                        'predicted_id': predicted_id
                    }
                })
        
        return results
    
    def map_rating_to_sentiment(self, rating):
        """Map star rating to expected sentiment"""
        if rating >= 4:  # 4-5 stars = positive
            return 'positive'
        elif rating <= 2:  # 1-2 stars = negative
            return 'neutral' if rating == 2 else 'negative'
        else:  # 3 stars = neutral
            return 'neutral'
    
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
    
    def fine_tune(self, df, text_col='Review', rating_col='Rating', epochs=3, batch_size=8, learning_rate=5e-5):
        """Fine-tune the BERT model on hotel reviews"""
        # Prepare training data
        texts = df[text_col].tolist()
        labels = [self.label_to_id[self.map_rating_to_sentiment(r)] for r in df[rating_col]]
        
        # Tokenize all texts
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Create dataset
        input_ids = encodings['input_ids']
        attention_mask = encodings['attention_mask']
        labels_tensor = torch.tensor(labels)
        
        # Create DataLoader
        dataset = torch.utils.data.TensorDataset(input_ids, attention_mask, labels_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Set model to training mode
        self.model.train()
        
        # Prepare optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        for epoch in range(epochs):
            total_loss = 0
            for batch in loader:
                batch_input_ids, batch_attention_mask, batch_labels = [b.to(self.device) for b in batch]
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    labels=batch_labels
                )
                
                # Get loss
                loss = outputs.loss
                total_loss += loss.item()
                
                # Backward pass
                loss.backward()
                
                # Update weights
                optimizer.step()
            
            print(f"Epoch {epoch+1}/{epochs} - Average loss: {total_loss/len(loader):.4f}")
        
        # Set model back to evaluation mode
        self.model.eval()
        
        print("Fine-tuning complete")
    
    def save(self, filepath):
        """Save the model to a file"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'id_to_label': self.id_to_label,
            'label_to_id': self.label_to_id
        }, filepath)
    
    @classmethod
    def load(cls, filepath, model_name='bert-base-uncased', device=None):
        """Load the model from a file"""
        # Create instance
        instance = cls(model_name=model_name, device=device)
        
        # Load state dict
        checkpoint = torch.load(filepath, map_location=instance.device)
        instance.model.load_state_dict(checkpoint['model_state_dict'])
        instance.id_to_label = checkpoint['id_to_label']
        instance.label_to_id = checkpoint['label_to_id']
        
        return instance