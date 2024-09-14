import os
import pandas as pd
import numpy as np
from collections import defaultdict
import spacy
from scipy.sparse import csr_matrix

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Tokenizer Class
class Tokenizer:
    def __init__(self):
        self.vocab = defaultdict(int)  # Dictionary to store word frequencies
        self.pos_vocab = defaultdict(int)  # Dictionary to store POS tag frequencies
        self.stop_words = set(nlp.Defaults.stop_words)  # Use spaCy's stop words
        
    def tokenize_batch(self, texts):
        docs = list(nlp.pipe(texts, batch_size=50, n_process=-1))
        tokens_list = []
        for doc in docs:
            tokens_list.append([(token.text, token.pos_) for token in doc if token.text not in self.stop_words])
        return tokens_list

    def build_vocab(self, texts, vocab_size=1000):
        word_counts = defaultdict(int)
        pos_counts = defaultdict(int)
        tokens_list = self.tokenize_batch(texts)
        for tokens in tokens_list:
            for token, pos in tokens:
                word_counts[token] += 1
                pos_counts[pos] += 1

        # Select the top `vocab_size` most frequent words
        most_common_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:vocab_size]
        self.vocab = {word: idx for idx, (word, _) in enumerate(most_common_words)}
        self.pos_vocab = {pos: idx + len(self.vocab) for idx, (pos, _) in enumerate(pos_counts.items())}
        self.vocab_size = len(self.vocab) + len(self.pos_vocab)

    def texts_to_feature_matrix(self, texts):
        rows = []
        cols = []
        data = []
        tokens_list = self.tokenize_batch(texts)
        for i, tokens in enumerate(tokens_list):
            for token, pos in tokens:
                if token in self.vocab:
                    rows.append(i)
                    cols.append(self.vocab[token])
                    data.append(1)
                if pos in self.pos_vocab:
                    rows.append(i)
                    cols.append(self.pos_vocab[pos])
                    data.append(1)
        return csr_matrix((data, (rows, cols)), shape=(len(texts), self.vocab_size))

    def text_to_feature_vector(self, text):
        vector = np.zeros(self.vocab_size)
        tokens = self.tokenize_batch([text])[0]  # We need only one result
        for token, pos in tokens:
            if token in self.vocab:
                vector[self.vocab[token]] = 1
            if pos in self.pos_vocab:
                vector[self.pos_vocab[pos]] = 1
        return vector

# Naive Bayes Classifier
class NaiveBayesClassifier:
    def __init__(self, alpha=1.0):
        self.prior_positive = 0
        self.prior_negative = 0
        self.positive_word_probs = None
        self.negative_word_probs = None
        self.vocab_size = 0
        self.alpha = alpha  # Smoothing factor

    def train(self, X, y):
        num_docs = len(y)
        num_positive = np.sum(y)
        num_negative = num_docs - num_positive
        
        self.prior_positive = num_positive / num_docs
        self.prior_negative = num_negative / num_docs
        
        # Count word occurrences in positive and negative documents
        positive_counts = np.zeros(X.shape[1])
        negative_counts = np.zeros(X.shape[1])
        
        for i, label in enumerate(y):
            if label == 1:
                positive_counts += X[i].toarray().flatten()
            else:
                negative_counts += X[i].toarray().flatten()

        # Apply Laplace smoothing and calculate conditional probabilities
        self.positive_word_probs = (positive_counts + self.alpha) / (num_positive + self.alpha * self.vocab_size)
        self.negative_word_probs = (negative_counts + self.alpha) / (num_negative + self.alpha * self.vocab_size)
    
    def predict(self, X):
        # Calculate log-probabilities for each class
        log_prior_positive = np.log(self.prior_positive)
        log_prior_negative = np.log(self.prior_negative)
        
        positive_scores = X @ np.log(self.positive_word_probs)
        negative_scores = X @ np.log(self.negative_word_probs)
        
        return (log_prior_positive + positive_scores) >= (log_prior_negative + negative_scores)

# Function to calculate metrics
def compute_metrics(true_labels, pred_labels):
    true_positives = np.sum((true_labels == 1) & (pred_labels == 1))
    true_negatives = np.sum((true_labels == 0) & (pred_labels == 0))
    false_positives = np.sum((true_labels == 0) & (pred_labels == 1))
    false_negatives = np.sum((true_labels == 1) & (pred_labels == 0))
    
    accuracy = (true_positives + true_negatives) / (true_positives + false_negatives + false_positives + true_negatives)
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return accuracy, precision, recall, f1

# Load the data
def load_data(data_dir, split):
    texts = []
    labels = []
    file_paths = []
    
    if split == 'test':
        paths_file = os.path.join(data_dir, f'{split}_paths.csv')
        df = pd.read_csv(paths_file, header=None)  # Test file might not have headers
        for _, row in df.iterrows():
            review_path = os.path.join(data_dir, row[0])  # row[0] because there's no 'review' column
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

    # Tokenize and convert texts to feature vectors
    tokenizer = Tokenizer()
    tokenizer.build_vocab(train_texts)
    
    X_train = tokenizer.texts_to_feature_matrix(train_texts)
    X_val = tokenizer.texts_to_feature_matrix(val_texts)
    X_test = tokenizer.texts_to_feature_matrix(test_texts)

    # Train Naive Bayes classifier
    model = NaiveBayesClassifier()
    model.train(X_train, train_labels)

    # Predict on validation and test sets
    val_preds = model.predict(X_val)
    test_preds = model.predict(X_test)

    # Save validation and test predictions to CSV
    val_df = pd.DataFrame({'review': val_file_paths, 'sentiment': val_preds.astype(int)})
    val_df.to_csv('val_predictions.csv', index=False)

    test_df = pd.DataFrame({'review': test_file_paths, 'sentiment': test_preds.astype(int)})
    test_df.to_csv('test_predictions.csv', index=False)

    # Evaluate on validation set
    val_acc, val_prec, val_rec, val_f1 = compute_metrics(val_labels, val_preds)
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Validation Precision: {val_prec:.4f}")
    print(f"Validation Recall: {val_rec:.4f}")
    print(f"Validation F1: {val_f1:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Naive Bayes Sentiment Classifier")
    parser.add_argument('--data_src', type=str, required=True, help="Path to the dataset folder")
    args = parser.parse_args()

    main(args.data_src)
