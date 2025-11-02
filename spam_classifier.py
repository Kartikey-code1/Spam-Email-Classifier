"""
Spam email classifier using Naive Bayes and TF-IDF
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import pickle
import os

class SpamClassifier:
    def __init__(self, model_type='naive_bayes'):
        self.model_type = model_type
        self.pipeline = None
        self.is_trained = False
        self.metrics = {}
        
    def create_pipeline(self):
        """Create ML pipeline with TF-IDF and classifier"""
        if self.model_type == 'naive_bayes':
            classifier = MultinomialNB(alpha=1.0)
        elif self.model_type == 'logistic_regression':
            classifier = LogisticRegression(random_state=42, max_iter=1000)
        else:
            raise ValueError("Model type must be 'naive_bayes' or 'logistic_regression'")
        
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words='english',
                lowercase=True,
                min_df=2,
                max_df=0.95
            )),
            ('classifier', classifier)
        ])
        
    def train(self, emails, labels, test_size=0.2, random_state=42):
        """Train the classifier"""
        # Create pipeline if not exists
        if self.pipeline is None:
            self.create_pipeline()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            emails, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        
        # Train the model
        self.pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.pipeline.predict(X_test)
        
        # Calculate metrics
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, pos_label='spam'),
            'recall': recall_score(y_test, y_pred, pos_label='spam'),
            'f1_score': f1_score(y_test, y_pred, pos_label='spam'),
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'test_size': len(X_test),
            'train_size': len(X_train)
        }
        
        self.is_trained = True
        return self.metrics
    
    def predict(self, email):
        """Predict if an email is spam or ham"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        prediction = self.pipeline.predict([email])[0]
        probability = self.pipeline.predict_proba([email])[0]
        
        return {
            'prediction': prediction,
            'spam_probability': probability[1] if len(probability) > 1 else (1.0 if prediction == 'spam' else 0.0),
            'ham_probability': probability[0] if len(probability) > 1 else (0.0 if prediction == 'spam' else 1.0)
        }
    
    def get_feature_importance(self, top_n=20):
        """Get most important features for spam classification"""
        if not self.is_trained:
            return None
        
        # Get feature names and coefficients
        vectorizer = self.pipeline.named_steps['tfidf']
        classifier = self.pipeline.named_steps['classifier']
        
        feature_names = vectorizer.get_feature_names_out()
        
        if hasattr(classifier, 'coef_'):
            # For logistic regression
            coefficients = classifier.coef_[0]
            feature_importance = list(zip(feature_names, coefficients))
            # Sort by absolute coefficient value
            feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        elif hasattr(classifier, 'feature_log_prob_'):
            # For Naive Bayes
            log_prob_spam = classifier.feature_log_prob_[1]  # spam class
            log_prob_ham = classifier.feature_log_prob_[0]   # ham class
            # Calculate difference (spam - ham)
            importance = log_prob_spam - log_prob_ham
            feature_importance = list(zip(feature_names, importance))
            feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        else:
            return None
        
        return feature_importance[:top_n]
    
    def save_model(self, filepath='spam_classifier.pkl'):
        """Save trained model to file"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.pipeline, f)
    
    def load_model(self, filepath='spam_classifier.pkl'):
        """Load trained model from file"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self.pipeline = pickle.load(f)
            self.is_trained = True
            return True
        return False