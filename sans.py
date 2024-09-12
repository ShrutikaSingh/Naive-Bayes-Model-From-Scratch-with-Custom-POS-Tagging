import pandas as pd
import os
import numpy as np
import re
from collections import Counter
import argparse

# Tokenizer class
class Tokenizer:
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.vocabulary = {}

    def tokenize(self, text):
        # Lowercase and remove punctuation
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = text.split()
        return tokens

    def build_vocabulary(self, texts):
        word_counts = Counter()
        for text in texts:
            tokens = self.tokenize(text)
            word_counts.update(tokens)
        most_common_words = word_counts.most_common(self.vocab_size)
        self.vocabulary = {word: idx for idx, (word, _) in enumerate(most_common_words)}

    def text_to_vector(self, text):
        tokens = self.tokenize(text)
        vector = [0] * self.vocab_size
        for token in tokens:
            if token in self.vocabulary:
                vector[self.vocabulary[token]] = 1
        return vector

class NaiveBayesClassifier:
    def __init__(self, alpha=1.0):
        self.class_probs = {}
        self.word_probs = {}
        self.vocab_size = 0
        self.alpha = alpha  # Laplace smoothing factor

    def fit(self, X, y):
        n_samples, self.vocab_size = X.shape
        class_counts = Counter(y)
        total_samples = len(y)

        # Print class counts and total samples
        print("Class counts:", class_counts)
        print("Total samples:", total_samples)

        # Adjusting class probabilities to account for any imbalance
        self.class_probs = {c: (class_counts[c] + self.alpha) / (total_samples + self.alpha * len(class_counts)) for c in class_counts}
        print("Class probabilities:", self.class_probs)

        self.word_probs = {c: np.ones(self.vocab_size) * self.alpha for c in class_counts}  # Laplace smoothing for word counts
        word_count_per_class = {c: self.alpha * self.vocab_size for c in class_counts}  # Start with alpha smoothing

        for i in range(n_samples):
            label = y[i]
            self.word_probs[label] += X[i]
            word_count_per_class[label] += np.sum(X[i])

        for c in self.word_probs:
            self.word_probs[c] /= word_count_per_class[c]
        print("Word probabilities:", self.word_probs)

    def predict(self, X):
        predictions = []
        for i in range(X.shape[0]):
            log_probs = {}
            for c in self.class_probs:
                log_prob = np.log(self.class_probs[c])
                log_prob += np.sum(np.log(self.word_probs[c]) * X[i])
                log_probs[c] = log_prob
            predictions.append(max(log_probs, key=log_probs.get))
        return predictions

# Evaluation metrics
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision(y_true, y_pred):
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    predicted_positives = np.sum(y_pred == 1)
    return true_positives / predicted_positives if predicted_positives != 0 else 0

def recall(y_true, y_pred):
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    actual_positives = np.sum(y_true == 1)
    return true_positives / actual_positives if actual_positives != 0 else 0

def f1_score(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) != 0 else 0

# Main function
def main(data_src):
    # Load the data
    train_df = pd.read_csv(os.path.join(data_src, 'train_labels.csv'))
    val_df = pd.read_csv(os.path.join(data_src, 'val_labels.csv'))
    test_df = pd.read_csv(os.path.join(data_src, 'test_paths.csv'), header=None)
    test_df.columns = ['review']

    # Load and preprocess reviews
    train_texts = [open(os.path.join(data_src, path), 'r').read() for path in train_df['review']]
    val_texts = [open(os.path.join(data_src, path), 'r').read() for path in val_df['review']]
    test_texts = [open(os.path.join(data_src, path), 'r').read() for path in test_df['review']]

    # Initialize tokenizer and build vocabulary
    tokenizer = Tokenizer(vocab_size=10000)
    tokenizer.build_vocabulary(train_texts)

    # Convert texts to binary feature vectors
    train_X = np.array([tokenizer.text_to_vector(text) for text in train_texts])
    val_X = np.array([tokenizer.text_to_vector(text) for text in val_texts])
    test_X = np.array([tokenizer.text_to_vector(text) for text in test_texts])

    # Labels
    train_y = np.array(train_df['sentiment'])
    val_y = np.array(val_df['sentiment'])

    # Train Naive Bayes model
    nb = NaiveBayesClassifier(alpha=1.0)  # Consider adjusting alpha
    nb.fit(train_X, train_y)

    # Predict on validation set
    val_pred = nb.predict(val_X)

    # Evaluate model
    val_accuracy = accuracy(val_y, val_pred)
    val_precision = precision(val_y, val_pred)
    val_recall = recall(val_y, val_pred)
    val_f1 = f1_score(val_y, val_pred)

    print(f'Validation Accuracy: {val_accuracy:.4f}')
    print(f'Validation Precision: {val_precision:.4f}')
    print(f'Validation Recall: {val_recall:.4f}')
    print(f'Validation F1 Score: {val_f1:.4f}')

    # Save validation predictions
    pd.DataFrame({'prediction': val_pred}).to_csv('val_predictions.csv', index=False)

    # Predict on test set
    test_pred = nb.predict(test_X)
    
    # Save test predictions
    pd.DataFrame({'prediction': test_pred}).to_csv('test_predictions.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_src', type=str, help='Path to the data directory')
    args = parser.parse_args()
    main(args.data_src)
