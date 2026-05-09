"""
Text Analysis Module - LLM-based sentiment and intent analysis
Uses Hugging Face transformers for customer profile analysis (optional)
Falls back to keyword-based analysis if transformers unavailable
"""

import numpy as np
from typing import Dict, Tuple

# Try to import transformers, but provide graceful fallback
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  Transformers not available - using keyword-based sentiment analysis")

class TextAnalyzer:
    """Analyzes customer text data using LLM or keyword fallback"""
    
    def __init__(self):
        """Initialize text analysis pipelines"""
        self.transformers_available = TRANSFORMERS_AVAILABLE
        
        if TRANSFORMERS_AVAILABLE:
            try:
                # Sentiment analysis
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english"
                )
                
                # Zero-shot classification for intent
                self.intent_pipeline = pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli"
                )
                print("✅ Transformers loaded successfully")
            except Exception as e:
                print(f"⚠️  Could not load Transformers: {e}")
                self.transformers_available = False
    
    def analyze_sentiment(self, text: str) -> Dict:
        """
        Analyze sentiment using LLM or keyword fallback
        Returns: {'label': 'POSITIVE'|'NEGATIVE', 'score': float}
        """
        if not text or len(text.strip()) == 0:
            return {'label': 'NEUTRAL', 'score': 0.5}
        
        if self.transformers_available:
            try:
                result = self.sentiment_pipeline(text[:512])[0]
                return {
                    'label': result['label'],
                    'score': result['score'],
                    'is_positive': result['label'] == 'POSITIVE'
                }
            except Exception as e:
                print(f"⚠️  Sentiment analysis failed: {e}")
                # Fall through to keyword analysis
        
        # Keyword-based fallback
        positive_words = ['good', 'great', 'excellent', 'love', 'amazing', 'happy', 'deposit', 'invest', 'save', 'interested']
        negative_words = ['bad', 'poor', 'hate', 'angry', 'disappointed', 'no', 'don\'t', 'won\'t']
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            return {'label': 'POSITIVE', 'score': 0.8, 'is_positive': True}
        elif neg_count > pos_count:
            return {'label': 'NEGATIVE', 'score': 0.2, 'is_positive': False}
        else:
            return {'label': 'NEUTRAL', 'score': 0.5, 'is_positive': False}
    
    def analyze_intent(self, text: str, candidate_labels: list = None) -> Dict:
        """
        Classify customer intent or use keyword matching
        """
        if not candidate_labels:
            candidate_labels = [
                "loan inquiry",
                "account management",
                "complaint",
                "investment consultation",
                "general inquiry"
            ]
        
        if not text or len(text.strip()) == 0:
            return {'intent': 'unknown', 'confidence': 0.0}
        
        if self.transformers_available:
            try:
                result = self.intent_pipeline(
                    text[:512],
                    candidate_labels,
                    multi_class=True
                )
                return {
                    'intent': result['labels'][0],
                    'confidence': result['scores'][0],
                    'all_intents': dict(zip(result['labels'], result['scores']))
                }
            except Exception as e:
                print(f"⚠️  Intent analysis failed: {e}")
                # Fall through to keyword analysis
        
        # Keyword-based fallback
        text_lower = text.lower()
        keywords = {
            'loan inquiry': ['loan', 'borrow', 'credit'],
            'account management': ['account', 'manage', 'settings'],
            'complaint': ['complaint', 'issue', 'problem', 'angry', 'upset'],
            'investment consultation': ['invest', 'deposit', 'fd', 'fixed', 'save'],
            'general inquiry': ['question', 'info', 'know', 'tell']
        }
        
        scores = {}
        for intent, words in keywords.items():
            score = sum(1 for word in words if word in text_lower) / len(words)
            scores[intent] = score
        
        best_intent = max(scores, key=scores.get)
        confidence = scores[best_intent]
        
        return {
            'intent': best_intent,
            'confidence': min(confidence, 0.9),
            'all_intents': scores
        }
    
    def extract_features(self, text: str) -> np.ndarray:
        """
        Extract numerical features from text for ML model
        Returns: [sentiment_score, positive_indicator, intent_confidence]
        """
        sentiment = self.analyze_sentiment(text)
        intent = self.analyze_intent(text)
        
        features = np.array([
            sentiment['score'],
            1.0 if sentiment['is_positive'] else 0.0,
            intent['confidence']
        ])
        
        return features
    
    def analyze_customer_profile(self, profile_text: str) -> Dict:
        """
        Comprehensive analysis of customer profile/notes
        Returns detailed sentiment, intent, and key features
        """
        sentiment = self.analyze_sentiment(profile_text)
        intent = self.analyze_intent(profile_text)
        
        return {
            'sentiment': sentiment,
            'intent': intent,
            'text_features': self.extract_features(profile_text),
            'summary': f"Customer shows {sentiment['label'].lower()} sentiment with primary intent: {intent['intent']}"
        }


# Example usage
if __name__ == "__main__":
    analyzer = TextAnalyzer()
    
    # Test samples
    sample_text = "I'm interested in opening a savings account and would like to know about your deposit options and interest rates."
    
    print("Text Analysis Results:")
    print("-" * 50)
    
    sentiment = analyzer.analyze_sentiment(sample_text)
    print(f"Sentiment: {sentiment}")
    
    intent = analyzer.analyze_intent(sample_text)
    print(f"Intent: {intent}")
    
    features = analyzer.extract_features(sample_text)
    print(f"Extracted Features: {features}")
