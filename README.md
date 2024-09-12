### Overview
This repository contains 50k IMDB movie reviews, collected by [Maas et al. (2011)](https://ai.stanford.edu/~ang/papers/acl11-WordVectorsSentimentAnalysis.pdf). The files have not been modified from their original form, except for the name of the file. Ambiguous reviews (original score is 5 or 6 out of 10) are not included.

### Splits
They are split into train (n=40k), validation (n=5k), and test (n=5k) with a ratio 8:1:1. Splits are evenly stratified with regard to length (number of whitespaces) and sentiment.

### Labels
The `_labels.csv` files have two columns:
 - `review`: path to the review text file. The ID has no purpose other than to distinguish reviews.
 - `sentiment`: 1 for positive and 0 for negative.

### Output format
A complete submission for this assignment will include a file called `test_predictions.csv` that is formatted like `*_labels.csv`. The `sentiment` column should only contain the integers 0 for negative and 1 for positive.


python3 main.py --data_src /Users/206819985/Documents/csci_444_assign/hw1-imdb

main1.py
Validation Accuracy: 0.8428
Validation Precision: 0.8576
Validation Recall: 0.8219
Validation F1: 0.8394


Loads the training, validation, and test data.
Trains a Naive Bayes model on the training data.
Evaluates the model on the validation set and prints the accuracy, precision, recall, and F1 score.
Saves the validation and test predictions to CSV files.

main 2.py workking fully
Validation Accuracy: 0.8534
Validation Precision: 0.8558
Validation Recall: 0.8499
Validation F1: 0.8528

<img width="638" alt="image" src="https://github.com/user-attachments/assets/62c44b32-0379-4b9a-8431-7d148ce08897">

    # Taking this formula from slides
    accuracy = (true_positives + true_negatives) / (true_positives+false_negatives+false_positives+true_negatives)
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1 = 2 * (precision * recall) / (precision + recall)
    
FINSAL
python3 main.py --data_src /Users/206819985/Documents/csci_444_assign/hw1-imdb
vocab length = 10000
Validation Accuracy: 0.8534
Validation Precision: 0.8558
Validation Recall: 0.8499
Validation F1: 0.8528



(venv) 206819985@OTS-FVFHK1DWQ05Q-MAC hw1-imdb % python3 main.py --data_src /Users/206819985/Documents/csci_444_assign/hw1-imdb



BEFORE REMOVING STOP WORDS
vocab length = 10000
Validation Accuracy: 0.8534
Validation Precision: 0.8558
Validation Recall: 0.8499
Validation F1: 0.8528
(venv) 206819985@OTS-FVFHK1DWQ05Q-MAC hw1-imdb % python3 main.py --data_src /Users/206819985/Documents/csci_444_assign/hw1-imdb


AFTER REMOVING STOP WORD ACCURACY INCREASED
vocab length = 10000
Validation Accuracy: 0.8552
Validation Precision: 0.8636
Validation Recall: 0.8435
Validation F1: 0.8534
(venv) 206819985@OTS-FVFHK1DWQ05Q-MAC hw1-imdb % 
