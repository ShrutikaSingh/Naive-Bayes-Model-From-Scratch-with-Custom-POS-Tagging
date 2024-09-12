import os
import pandas as pd
import numpy as np
from collections import defaultdict
import re

# Tokenizer Class (No changes here)
class Tokenizer:
    def __init__(self):
        self.vocab = defaultdict(int)  # Dictionary to store word frequencies
        self.stop_words = set([
            'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'aren\'t', 
            'as', 'at', 'be', 'because', 'been', 'before', 'being', 'between', 'both', 'but', 'by', 'can', 
            'could', 'did', 'didn\'t', 'do', 'does', 'doesn\'t', 'doing', 'don\'t', 'down', 'during', 'each', 'few', 
            'for', 'from', 'further', 'had', 'hadn\'t', 'has', 'hasn\'t', 'have', 'haven\'t', 'having', 'he', 'her', 
            'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'isn\'t', 'it', 
            'its', 'itself', 'just', 'll', 'm', 'me', 'might', 'more', 'most', 'mustn\'t', 'my', 'myself', 'needn\'t', 
            'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 
            'out', 'over', 'own', 're', 's', 'same', 'shan\'t', 'she', 'should', 'shouldn\'t', 'so', 't', 'than', 'that', 
            'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 'through', 
            'to', 'too', 'under', 'until', 'up', 've', 'very', 'was', 'wasn\'t', 'we', 'were', 'weren\'t', 'what', 
            'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'won\'t', 'would', 'y', 'you', 
            'your', 'yours', 'yourself', 'yourselves'
        ])
    
    def tokenize(self, text):
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        # Remove stop words
        tokens = [token for token in tokens if token not in self.stop_words]
        return tokens
    
    def build_vocab(self, texts, vocab_size=10000):
        word_counts = defaultdict(int)
        for text in texts:
            tokens = self.tokenize(text)
            for token in tokens:
                word_counts[token] += 1

        # Select the top `vocab_size` most frequent words
        most_common = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:vocab_size]
        self.vocab = {word: idx for idx, (word, _) in enumerate(most_common)}
        print("vocab length =", len(self.vocab))
    
    def text_to_count_vector(self, text):
        vector = np.zeros(len(self.vocab))
        tokens = self.tokenize(text)
        for token in tokens:
            if token in self.vocab:
                vector[self.vocab[token]] += 1
        return vector

# Logistic Regression Classifier from Scratch
class LogisticRegressionClassifierFromScratch:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def initialize_parameters(self, n_features):
        # Initialize weights and bias to zeros
        self.weights = np.zeros(n_features)
        self.bias = 0

    def train(self, X, y):
        n_samples, n_features = X.shape
        self.initialize_parameters(n_features)

        for _ in range(self.num_iterations):
            # Linear model: y_hat = X * w + b
            linear_model = np.dot(X, self.weights) + self.bias
            # Apply sigmoid function
            y_hat = self.sigmoid(linear_model)
            
            # Gradient computation
            dw = (1 / n_samples) * np.dot(X.T, (y_hat - y))
            db = (1 / n_samples) * np.sum(y_hat - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_hat = self.sigmoid(linear_model)
        # Convert probabilities to binary output (0 or 1)
        return np.where(y_hat >= 0.5, 1, 0)

def compute_metrics(true_labels, pred_labels):
    true_positives = np.sum((true_labels == 1) & (pred_labels == 1))
    true_negatives = np.sum((true_labels == 0) & (pred_labels == 0))
    false_positives = np.sum((true_labels == 0) & (pred_labels == 1))
    false_negatives = np.sum((true_labels == 1) & (pred_labels == 0))
    
    # Calculate accuracy, precision, recall, and F1 score
    accuracy = (true_positives + true_negatives) / (true_positives + false_negatives + false_positives + true_negatives)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    
    return accuracy, precision, recall, f1

# Function to load data
def load_data(data_dir, split):
    texts = []
    labels = []
    file_paths = []
    
    if split == 'test':
        paths_file = os.path.join(data_dir, f'{split}_paths.csv')
        df = pd.read_csv(paths_file, header=None)
        for _, row in df.iterrows():
            review_path = os.path.join(data_dir, row[0])
            file_paths.append(row[0])
            with open(review_path, 'r', encoding='utf-8') as file:
                texts.append(file.read())
        return file_paths, texts
    else:
        label_file = os.path.join(data_dir, f'{split}_labels.csv')
        df = pd.read_csv(label_file)
        for _, row in df.iterrows():
            review_path = os.path.join(data_dir, row['review'])
            file_paths.append(row['review'])
            with open(review_path, 'r', encoding='utf-8') as file:
                texts.append(file.read())
            labels.append(row['sentiment'])
        return file_paths, texts, np.array(labels)

# Main function
def main(data_src):
    # Load data
    train_file_paths, train_texts, train_labels = load_data(data_src, 'train')
    val_file_paths, val_texts, val_labels = load_data(data_src, 'val')
    test_file_paths, test_texts = load_data(data_src, 'test')

    # Initialize and train tokenizer
    tokenizer = Tokenizer()
    tokenizer.build_vocab(train_texts, vocab_size=10000)
    
    # Convert text to feature vectors
    X_train = np.array([tokenizer.text_to_count_vector(text) for text in train_texts])
    X_val = np.array([tokenizer.text_to_count_vector(text) for text in val_texts])
    X_test = np.array([tokenizer.text_to_count_vector(text) for text in test_texts])
    
    # Train Logistic Regression classifier
    lr_classifier = LogisticRegressionClassifierFromScratch(learning_rate=0.01, num_iterations=1000)
    lr_classifier.train(X_train, train_labels)
    
    # Predict on validation set
    val_preds = lr_classifier.predict(X_val)
    
    # Compute metrics for validation set
    val_acc, val_prec, val_rec, val_f1 = compute_metrics(val_labels, val_preds)
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Validation Precision: {val_prec:.4f}")
    print(f"Validation Recall: {val_rec:.4f}")
    print(f"Validation F1: {val_f1:.4f}")
    
    # Save predictions to CSV
    val_df = pd.DataFrame({'review': val_file_paths, 'prediction': val_preds})
    val_df.to_csv(os.path.join(data_src, 'predicted_validation_labels_lr.csv'), index=False)
    
    # Predict on test set
    test_preds = lr_classifier.predict(X_test)
    
    # Save test predictions to CSV
    test_df = pd.DataFrame({'review': test_file_paths, 'prediction': test_preds})
    test_df.to_csv(os.path.join(data_src, 'predicted_test_labels_lr.csv'), index=False)

# Execute main function
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Naive Bayes Sentiment Classifier")
    parser.add_argument('--data_src', type=str, required=True, help="Path to the dataset folder")
    args = parser.parse_args()

    main(args.data_src)
