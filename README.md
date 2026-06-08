## News Topic Classifier

A simple NLP project where I train a model to automatically classify news headlines into different topics (like politics, sports, tech, entertainment, etc.). The goal is to build something that’s easy to understand, works well, and shows the full NLP workflow from data prep → model training → evaluation.

This project takes a news headline or short text and predicts what category it belongs to.  
Examples:  
- “Government passes new data privacy law” → Politics  
- “New iPhone model breaks sales records” → Technology  
- “NBA star scores 50 points in comeback game” → Sports

It uses TF‑IDF features + Logistic Regression  

### Dataset

I’m using the HuffPost News Category Dataset, which has around 200k headlines labeled with 41 different categories.  
Each entry includes:
- headline
- category
- short_description
- date  

For this project, I only use the headline and category fields.  

Dataset link:  
https://www.kaggle.com/datasets/rmisra/news-category-dataset

### Methods
1. Preprocessing: Lowercasing text, TF-IDF vectorization, Stopword removal
2. Model: Logistic Regression
3. Evaluation: Accuracy, Precision, Recall, F1-score, Confusion matrix

### Results
- Training progress
- Validation accuracy
- Test accuracy
- Classification report
- Confusion matrix plot
- A sample prediction 
