import pandas as pd
import numpy as np
import pickle
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ensemble_voter import EnsembleVoter
from text_analyzer import TextAnalyzer

# ----------------------------
# 1. LOAD DATA
# ----------------------------
def load_data(file_path="data/bank.csv"):
    """Load bank marketing dataset"""
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    return df

# ----------------------------
# 2. PREPROCESSING
# ----------------------------
def preprocess_data(df):
    """Preprocess data for ML model"""
    # Convert target
    df["deposit"] = df["deposit"].map({"yes": 1, "no": 0})
    # One-hot encoding for categorical features
    df = pd.get_dummies(df, drop_first=True)
    return df

# ----------------------------
# 3. SPLIT & SCALE
# ----------------------------
def split_scale(df):
    """Split data into train/test and scale features"""
    X = df.drop("deposit", axis=1)
    y = df["deposit"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler

# ----------------------------
# 4. TRAIN TRADITIONAL ML MODEL
# ----------------------------
def train_model(X_train, y_train):
    """Train logistic regression model"""
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)
    return model

# ----------------------------
# 5. EVALUATE MODEL
# ----------------------------
def evaluate_model(model, X_test, y_test, model_name="ML Model"):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*60}")
    print(f"📊 {model_name} EVALUATION")
    print(f"{'='*60}")
    print(f"✅ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n📈 Confusion Matrix:")
    print(cm)
    
    return y_pred, accuracy

# ----------------------------
# 6. SAVE MODELS & SCALER
# ----------------------------
def save_models(model, scaler, ensemble, 
                model_path="models/customer_model.pkl", 
                scaler_path="models/scaler.pkl",
                ensemble_path="models/ensemble.pkl"):
    """Save trained models"""
    os.makedirs("models", exist_ok=True)
    
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    with open(ensemble_path, "wb") as f:
        pickle.dump(ensemble, f)
    
    print("✅ All models saved!")

# ----------------------------
# 7. LOAD MODELS & SCALER
# ----------------------------
def load_models(model_path="models/customer_model.pkl", 
                scaler_path="models/scaler.pkl",
                ensemble_path="models/ensemble.pkl"):
    """Load trained models"""
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(ensemble_path, "rb") as f:
        ensemble = pickle.load(f)
    return model, scaler, ensemble

# ----------------------------
# 8. CREATE ENSEMBLE SYSTEM
# ----------------------------
def create_ensemble(model, scaler):
    """Create ensemble voting system"""
    ensemble = EnsembleVoter(model=model, scaler=scaler)
    return ensemble

# ----------------------------
# 9. PREDICTION FUNCTION
# ----------------------------
def predict_new(data, model, scaler):
    """
    Make predictions on new data
    
    Args:
        data: pd.DataFrame (raw input)
        model: Trained ML model
        scaler: Fitted scaler
        
    Returns:
        predictions: Array of predictions
    """
    df = preprocess_data(data.copy())
    X = df.drop("deposit", axis=1, errors="ignore")
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
    return predictions

def predict_ensemble(ml_features, 
                    text_features=None, 
                    image_features=None,
                    ensemble=None):
    """
    Make ensemble predictions combining multiple components
    
    Args:
        ml_features: Traditional ML features
        text_features: Optional text analysis features
        image_features: Optional image analysis features
        ensemble: EnsembleVoter instance
        
    Returns:
        Ensemble prediction result
    """
    if ensemble is None:
        return None
    
    return ensemble.predict(ml_features, text_features, image_features)

# ----------------------------
# MAIN TRAINING PIPELINE
# ----------------------------
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 ADVANCED ENSEMBLE BANKING AI SYSTEM")
    print("="*60)
    
    # Step 1: Load and preprocess data
    print("\n📥 Loading data...")
    df = load_data()
    df = preprocess_data(df)
    print(f"✅ Data loaded: {df.shape}")
    
    # Step 2: Split and scale
    print("\n🔄 Splitting and scaling data...")
    X_train, X_test, y_train, y_test, scaler = split_scale(df)
    print(f"✅ Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Step 3: Train traditional ML model
    print("\n🤖 Training Traditional ML Model (Logistic Regression)...")
    ml_model = train_model(X_train, y_train)
    print("✅ ML model training complete!")
    
    # Step 4: Evaluate ML model
    print("\n📊 Evaluating Traditional ML Model...")
    y_pred_ml, ml_accuracy = evaluate_model(ml_model, X_test, y_test, "ML Model")
    
    # Step 5: Initialize Text Analyzer (LLM)
    print("\n🔤 Initializing Text Analyzer...")
    try:
        text_analyzer = TextAnalyzer()
        print("✅ Text analyzer initialized!")
    except Exception as e:
        print(f"⚠️  Text analyzer initialization failed: {e}")
        print("   Using keyword-based fallback for text analysis")
        text_analyzer = TextAnalyzer()  # Still works with fallback
    
    # Step 6: Create ensemble
    print("\n🎯 Creating Ensemble Voting System...")
    ensemble = create_ensemble(ml_model, scaler)
    print("✅ Ensemble created with weights:")
    for component, weight in ensemble.voting_weights.items():
        print(f"   • {component}: {weight:.0%}")
    
    # Step 7: Save all models
    print("\n💾 Saving models...")
    save_models(ml_model, scaler, ensemble)
    print("✅ All models saved successfully!")
    
    # Step 8: Demo ensemble predictions
    print("\n" + "="*60)
    print("📈 ENSEMBLE PREDICTION DEMO")
    print("="*60)
    
    # Make a few test predictions
    for i in range(min(3, X_test.shape[0])):
        test_features = X_test[i:i+1]
        result = predict_ensemble(test_features, ensemble=ensemble)
        
        print(f"\n🔹 Sample {i+1}:")
        print(f"   ML Prediction: {result['component_predictions']['ml_model']:.2%}")
        print(f"   Text (Default): 50% (no text data)")
        print(f"   Image (Default): 50% (no image data)")
        print(f"   ➜ Final Ensemble: {result['final_prediction']:.2%}")
        print(f"   ➜ Predicted: {'DEPOSIT' if result['predicted_class'] == 1 else 'NO DEPOSIT'}")
        print(f"   Confidence: {result['confidence']:.2%}")
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print("\n📌 Next steps:")
    print("   1. Run: streamlit run app/app.py")
    print("   2. Upload customer data and documents")
    print("   3. Get multimodal ensemble predictions!")
    print("="*60)