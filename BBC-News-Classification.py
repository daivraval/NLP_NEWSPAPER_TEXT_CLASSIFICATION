# BBC-News-Classification.py (Rewritten)

import os
import re
import joblib  # For saving the model
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import NuSVC
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline  # <-- Import Pipeline

# --- NLTK Downloader (run once if needed) ---
# nltk.download('punkt')
# nltk.download('stopwords')
# ----------------------------------------------


def load_and_preprocess_data(data_folder_path, class_labels):
    """
    Loads data from subfolders and applies the same preprocessing
    as the original script.
    """
    print("Loading and preprocessing BBC data...")
    preprocessed_texts = []
    labels = []
    
    # Store English stopwords in a set for faster lookup
    stop_words = set(stopwords.words('english'))

    for label in class_labels:
        folder_path = os.path.join(data_folder_path, label)
        
        if not os.path.isdir(folder_path):
            print(f"Warning: Folder not found, skipping: {folder_path}")
            continue

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                
                # 1. Lowercase
                text = text.lower()
                # 2. Remove punctuation
                text = re.sub(r'[^\w\s]', '', text)
                # 3. Remove numbers/dates
                text = re.sub(r'\d+', '', text)
                
                # 4. Tokenize
                tokens = word_tokenize(text)
                # 5. Remove stopwords
                tokens = [word for word in tokens if word not in stop_words]
                
                # Re-join tokens into a single string for the vectorizer
                final_text = ' '.join(tokens)
                
                preprocessed_texts.append(final_text)
                labels.append(label)
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
                
    if not preprocessed_texts:
        print("\nError: No data was loaded.")
        print("Please check the 'data_folder_path' variable in the script.")
        return None, None
    
    print(f"Successfully loaded and preprocessed {len(preprocessed_texts)} documents.")
    return preprocessed_texts, labels

# --- Main execution ---
if __name__ == "__main__":
    
    # --- 1. Load Data ---
    BBC_FOLDER_PATH = 'C:/Users/Deiv/Desktop/NewsClassification/bbc/'
    CLASS_LABELS = ['business', 'entertainment', 'politics', 'sport', 'tech']
    
    texts, labels = load_and_preprocess_data(BBC_FOLDER_PATH, CLASS_LABELS)

    if texts:
        # --- 2. Split Data (BEFORE fitting) ---
        # This is the correct way to prevent data leakage
        X_train, X_test, y_train, y_test = train_test_split(
            texts, 
            labels, 
            test_size=0.2, 
            shuffle=True, 
            random_state=42
        )
        
        print(f"Data split: {len(X_train)} training samples, {len(X_test)} testing samples.")

        # --- 3. Build the Pipeline ---
        # This bundles the feature extractor (TF-IDF) and the classifier (SVM)
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),  # Step 1: Convert text to TF-IDF vectors
            ('svm', NuSVC(nu=0.5))          # Step 2: Classify using the vectors
        ])

        # --- 4. Train the Pipeline ---
        print("Training the model...")
        pipeline.fit(X_train, y_train)
        print("Training complete.")

        # --- 5. Evaluate the Model ---
        print("Evaluating the model on the test set...")
        y_pred = pipeline.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        report = classification_report(y_test, y_pred, target_names=CLASS_LABELS)

        print("\n--- Model Evaluation ---")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-score:  {f1:.4f}")
        print("\nClassification Report:")
        print(report)

        # --- 6. Save the Trained Pipeline ---
        model_filename = 'bbc_pipeline.joblib'
        joblib.dump(pipeline, model_filename)

        print("---")
        print(f"✅ Successfully trained and saved the complete pipeline to '{model_filename}'")