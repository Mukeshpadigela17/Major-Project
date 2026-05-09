"""
Ensemble Voting System - Combines multiple models for robust predictions
Integrates traditional ML, LLM text analysis, and image processing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler
import pickle

class EnsembleVoter:
    """
    Voting ensemble that combines predictions from:
    1. Traditional ML model (LogisticRegression)
    2. LLM-based text analysis
    3. Image document analysis
    """
    
    def __init__(self, model=None, scaler=None):
        """
        Initialize ensemble with pre-trained model and scaler
        
        Args:
            model: Trained sklearn model for traditional ML
            scaler: StandardScaler fitted to training data
        """
        self.ml_model = model
        self.scaler = scaler
        self.voting_weights = {
            'ml_model': 0.5,      # 50% - traditional ML is most reliable
            'text_analysis': 0.25, # 25% - LLM text sentiment/intent
            'image_analysis': 0.25 # 25% - Document quality and OCR
        }
        
        self.prediction_history = []
    
    def set_weights(self, ml_weight: float, text_weight: float, image_weight: float):
        """
        Customize voting weights (must sum to 1.0)
        
        Args:
            ml_weight: Weight for traditional ML prediction
            text_weight: Weight for text analysis prediction
            image_weight: Weight for image analysis prediction
        """
        total = ml_weight + text_weight + image_weight
        if total != 1.0:
            # Normalize
            ml_weight /= total
            text_weight /= total
            image_weight /= total
        
        self.voting_weights = {
            'ml_model': ml_weight,
            'text_analysis': text_weight,
            'image_analysis': image_weight
        }
    
    def _safe_predict_proba(self, model, X):
        if hasattr(model, 'coef_') and not hasattr(model, 'multi_class'):
            model.multi_class = 'ovr'

        if hasattr(model, 'predict_proba'):
            try:
                return model.predict_proba(X)
            except Exception:
                pass

        if hasattr(model, 'decision_function'):
            df = model.decision_function(X)
            if df.ndim == 1:
                p = 1 / (1 + np.exp(-df))
                return np.vstack([1 - p, p]).T
            if df.ndim == 2:
                exp = np.exp(df - np.max(df, axis=1, keepdims=True))
                return exp / np.sum(exp, axis=1, keepdims=True)

        preds = model.predict(X)
        return np.vstack([1 - preds, preds]).T

    def predict_ml_component(self, features: np.ndarray) -> Tuple[float, float]:
        """
        Get prediction from traditional ML model
        
        Args:
            features: Scaled feature array
            
        Returns:
            (prediction_score, confidence)
        """
        if self.ml_model is None:
            return 0.5, 0.0  # Default if no model
        
        # Scale features
        if self.scaler:
            features = self.scaler.transform(features.reshape(1, -1))
        
        proba = self._safe_predict_proba(self.ml_model, features)
        prediction = proba[0][1]
        confidence = max(proba[0])
        
        return prediction, confidence
    
    def predict_text_component(self, text_features: np.ndarray) -> Tuple[float, float]:
        """
        Get prediction from text analysis features
        
        Args:
            text_features: [sentiment_score, positive_indicator, intent_confidence]
            
        Returns:
            (prediction_score, confidence)
        """
        if len(text_features) == 0:
            return 0.5, 0.0
        
        # Combine text features: sentiment + positive indicator
        text_prediction = np.mean(text_features[:2])  # Average of sentiment and positive flag
        text_confidence = text_features[2]  # Intent confidence
        
        return text_prediction, text_confidence
    
    def predict_image_component(self, image_features: np.ndarray) -> Tuple[float, float]:
        """
        Get prediction from image analysis features
        
        Args:
            image_features: [brightness, contrast, edge_density, validity]
            
        Returns:
            (prediction_score, confidence)
        """
        if len(image_features) == 0:
            return 0.5, 0.0
        
        # Image quality indicators - better quality docs suggest reliable customer
        image_prediction = np.mean(image_features[:3])  # Avg of brightness, contrast, edges
        image_confidence = image_features[3]  # Document validity
        
        return image_prediction, image_confidence
    
    def ensemble_vote(self, 
                     ml_prediction: float,
                     text_prediction: float,
                     image_prediction: float,
                     voting_strategy: str = 'weighted') -> Dict:
        """
        Combine predictions using voting strategy
        
        Args:
            ml_prediction: Score from ML model (0-1)
            text_prediction: Score from text analysis (0-1)
            image_prediction: Score from image analysis (0-1)
            voting_strategy: 'weighted' or 'majority'
            
        Returns:
            {
                'final_prediction': float,
                'confidence': float,
                'component_predictions': {...},
                'strategy_used': str
            }
        """
        component_predictions = {
            'ml_model': ml_prediction,
            'text_analysis': text_prediction,
            'image_analysis': image_prediction
        }
        
        if voting_strategy == 'weighted':
            # Weighted average
            final_pred = (
                ml_prediction * self.voting_weights['ml_model'] +
                text_prediction * self.voting_weights['text_analysis'] +
                image_prediction * self.voting_weights['image_analysis']
            )
            
            # Confidence is the minimum component confidence
            confidence = min([
                ml_prediction if ml_prediction != 0.5 else 0.5,
                text_prediction if text_prediction != 0.5 else 0.5,
                image_prediction if image_prediction != 0.5 else 0.5
            ])
        
        else:  # Majority voting
            votes = [
                1 if ml_prediction > 0.5 else 0,
                1 if text_prediction > 0.5 else 0,
                1 if image_prediction > 0.5 else 0
            ]
            final_pred = 1.0 if sum(votes) > 1 else 0.0
            confidence = abs(sum(votes) - 1.5) / 1.5  # Higher when there's strong agreement
        
        return {
            'final_prediction': final_pred,
            'confidence': confidence,
            'component_predictions': component_predictions,
            'strategy_used': voting_strategy,
            'predicted_class': 1 if final_pred > 0.5 else 0
        }
    
    def predict(self, 
                ml_features: np.ndarray = None,
                text_features: np.ndarray = None,
                image_features: np.ndarray = None) -> Dict:
        """
        Make ensemble prediction combining all components
        
        Args:
            ml_features: Traditional ML feature array
            text_features: Text analysis feature array
            image_features: Image analysis feature array
            
        Returns:
            Ensemble prediction result
        """
        # Get component predictions
        ml_pred, ml_conf = self.predict_ml_component(ml_features) if ml_features is not None else (0.5, 0.0)
        text_pred, text_conf = self.predict_text_component(text_features) if text_features is not None else (0.5, 0.0)
        image_pred, image_conf = self.predict_image_component(image_features) if image_features is not None else (0.5, 0.0)
        
        # Ensemble vote
        result = self.ensemble_vote(ml_pred, text_pred, image_pred)
        
        # Add confidence scores
        result['component_confidences'] = {
            'ml_model': ml_conf,
            'text_analysis': text_conf,
            'image_analysis': image_conf
        }
        
        # Store in history
        self.prediction_history.append(result)
        
        return result
    
    def get_prediction_explanation(self, result: Dict) -> str:
        """
        Generate human-readable explanation of ensemble prediction
        """
        pred_class = "WILL DEPOSIT" if result['predicted_class'] == 1 else "WON'T DEPOSIT"
        confidence = result['confidence']
        
        explanation = f"""
        ═════════════════════════════════════════════════════════
        ENSEMBLE PREDICTION: {pred_class}
        Confidence: {confidence:.2%}
        ═════════════════════════════════════════════════════════
        
        Component Predictions:
        • ML Model:        {result['component_predictions']['ml_model']:.2%}
        • Text Analysis:   {result['component_predictions']['text_analysis']:.2%}
        • Image Analysis:  {result['component_predictions']['image_analysis']:.2%}
        
        Voting Weights:
        • ML Model:        {self.voting_weights['ml_model']:.0%}
        • Text Analysis:   {self.voting_weights['text_analysis']:.0%}
        • Image Analysis:  {self.voting_weights['image_analysis']:.0%}
        
        Strategy: {result['strategy_used']}
        """
        
        return explanation


# Example usage
if __name__ == "__main__":
    voter = EnsembleVoter()
    
    # Simulate component predictions
    ml_features = np.array([1, 0, 1, 1, 0])  # Some dummy features
    text_features = np.array([0.8, 1.0, 0.9])  # High sentiment score
    image_features = np.array([0.7, 0.6, 0.8, 1.0])  # Good document quality
    
    result = voter.predict(ml_features, text_features, image_features)
    print(voter.get_prediction_explanation(result))
