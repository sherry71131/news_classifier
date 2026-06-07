# main.py
# Simple News Topic Classifier using TF-IDF + Logistic Regression

import pandas as pd  # For loading and handling the dataset
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.feature_extraction.text import TfidfVectorizer  # For converting text to numbers
from sklearn.linear_model import LogisticRegression  # Our classifier
from sklearn.pipeline import Pipeline  # To chain TF-IDF and the model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  # For evaluation
import seaborn as sns  # For plotting confusion matrix
import matplotlib.pyplot as plt  # For showing plots

# 1. Load dataset
# Make sure the JSON file is in the same folder as this script.
# You can download the HuffPost News Category Dataset and rename it to this file name.
DATA_PATH = "News_Category_Dataset_v3.json"

def load_data(path):
    """Load the HuffPost news dataset from JSON."""
    # Read the JSON file into a pandas DataFrame
    df = pd.read_json(path, lines=True)
    # We will use the 'headline' as text and 'category' as label
    df = df[['headline', 'category']]
    # Drop any rows with missing values just to be safe
    df = df.dropna()
    return df

# 2. Simple text preprocessing (optional, TF-IDF already handles a lot)
def basic_clean(text):
    """Very simple text cleaning: lowercase only."""
    # Convert text to lowercase
    return text.lower()

def prepare_data(df):
    """Apply basic cleaning and return texts and labels."""
    # Apply cleaning to each headline
    df['clean_headline'] = df['headline'].apply(basic_clean)
    # X = input texts, y = labels
    X = df['clean_headline'].values
    y = df['category'].values
    return X, y

# 3. Split data into train, validation, and test sets
def split_data(X, y):
    """Split data into train, validation, and test sets."""
    # First split: train + temp (validation + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    # Second split: validation + test from temp
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

# 4. Build a simple TF-IDF + Logistic Regression pipeline
def build_model():
    """Create a pipeline that converts text to TF-IDF features and trains Logistic Regression."""
    model = Pipeline([
        # Step 1: Convert text to TF-IDF features
        ('tfidf', TfidfVectorizer(
            max_features=20000,  # Limit vocabulary size to keep it simple
            ngram_range=(1, 2),  # Use unigrams and bigrams
            stop_words='english'  # Remove common English stopwords
        )),
        # Step 2: Logistic Regression classifier
        ('clf', LogisticRegression(
            max_iter=1000,  # Increase iterations so it converges
            n_jobs=-1       # Use all CPU cores if available
        ))
    ])
    return model

# 5. Train the model
def train_model(model, X_train, y_train, X_val, y_val):
    """Train the model and print validation accuracy."""
    # Fit the model on training data
    model.fit(X_train, y_train)
    # Predict on validation data
    y_val_pred = model.predict(X_val)
    # Compute validation accuracy
    val_acc = accuracy_score(y_val, y_val_pred)
    print(f"Validation Accuracy: {val_acc:.4f}")
    return model

# 6. Evaluate the model on the test set
def evaluate_model(model, X_test, y_test):
    """Evaluate the model on the test set and print metrics."""
    # Predict labels for test data
    y_pred = model.predict(X_test)
    # Print overall accuracy
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    # Print detailed classification report (precision, recall, F1 per class)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    # Plot confusion matrix (this can be large with many classes)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

# 7. Simple prediction function for your interactive demo
def predict_topic(model, text):
    """Given a raw text string, return the predicted topic."""
    # Clean the text in the same way as training data
    clean_text = basic_clean(text)
    # Model expects a list/array of texts
    prediction = model.predict([clean_text])[0]
    return prediction

# 8. Main script to run everything
def main():
    """Run the full pipeline: load data, train model, evaluate, and test a sample prediction."""
    # Load data
    df = load_data(DATA_PATH)
    print(f"Loaded {len(df)} headlines.")
    
    # Prepare data
    X, y = prepare_data(df)
    
    # Split into train/val/test
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")
    
    # Build model
    model = build_model()
    
    # Train model
    model = train_model(model, X_train, y_train, X_val, y_val)
    
    # Evaluate on test set
    evaluate_model(model, X_test, y_test)
    
    # Try a sample prediction
    sample_text = "Government passes new law on data privacy"
    predicted_topic = predict_topic(model, sample_text)
    print(f"\nSample text: {sample_text}")
    print(f"Predicted topic: {predicted_topic}")

if __name__ == "__main__":
    main()
