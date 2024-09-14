## Bonus Question


## Problem1: Discriminative Model: Logistic Regression

## Command to Run:
```python3 _bonus_1_logistic_regression_model.py --data_src <path_to_data_folder>```

### In my case
 ``` python3 _bonus_1_logistic_regression_model.py --data_src /Users/206819985/Documents/csci_444_assign/hw1-imdb ```

### Files Generated

test_predictions.csv

val_predictions.csv

### Results

<img width="804" alt="image" src="https://github.com/user-attachments/assets/658eb2d8-5a0a-4c93-9a92-39751bbaad0f">


## Problem2: Naive bayes with Continuous Features TF-IDF

## Command to Run:
``` python3 _bonus_2.py --data_src <path_to_data_folder>```

### In my case
 ``` python3 _bonus_2.py --data_src /Users/206819985/Documents/csci_444_assign/hw1-imdb ```

### Files Generated

test_predictions.csv

val_predictions.csv

## My Observatiosn

**Feature Assumptions:** Naive Bayes assumes that features are independent, thus going well with the bag-of-words approach. Therefore, this model can also result in better performance in those cases where the features are sparse, as in textual data.
Simplicity with High Dimensionality: Naive Bayes is really efficient for high-dimensional feature space and less prone to overfitting, while Logistic Regression may be problematic in the case of high-dimensional data if it is not regularized properly.

**Robustness to Noise**: Naive Bayes does not care about noisy features and/or irrelevant words. Because this is a probabilistic algorithm, it is pretty robust in cases of text data, where noise is quite common.

**Effective with Sparse Data:** Naive Bayes should be more effective in the case of the binary features that are sparse in nature, like the presence or absence of words, because it does not need to normalize or scale. This will most probably work better than Logistic Regression in this case, where scaling of features is not quite appropriate.

### Results

<img width="735" alt="image" src="https://github.com/user-attachments/assets/7f04a46b-dd6e-42ea-bbbd-b342df92b8e6">

# Problem3: Syntactic Features

NOTE: 

Since the code was taking more than 7 hrs to run, I created a dataset with a total of just 100 samples, 70 text files were in the train folder, 15 were in test and 15 was in the validation.  And using the data set provided by the professor, I manually update the  train_labels.csv, val_labels.csv, test_paths.csv corresponding to the same labels and data provided by the professor.

## Command to Run:
> python -m spacy download en_core_web_sm

> python3 _bonus_3.py --data_src <path_to_your_data_folder>

> python3 _bonus_3.py --data_src /Users/206819985/Documents/csci_444_assign/hw1-imdb/_bonus/data

### In my case
 ``` python3 _bonus_3.py --data_src /Users/206819985/Documents/csci_444_assign/hw1-imdb/_bonus/data ```

### Files Generated

test_predictions.csv

val_predictions.csv

## My Observation :

For the smaller dataset (100 samples), the model with POS tagging performs worse, which aligns with the observation that more data might be needed to see if POS tagging could offer any improvements.
But if the simple naive bayes is trained with just 100 samples without POS tagging, the performance is even worse. That means that with POS tagging , accuracy is  improved if we have a large data set.


Results

<img width="732" alt="image" src="https://github.com/user-attachments/assets/900636e7-8d98-406a-a0b8-820e06426a2d">

