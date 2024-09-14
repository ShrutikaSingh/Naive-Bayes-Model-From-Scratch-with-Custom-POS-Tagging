
## Problem1: Discriminative Model: Logistic Regression

## Command to Run:
```python3 _bonus_1_logistic_regression_model.py --data_src <path_to_data_folder>```

### In my case
 ``` python3 _bonus_1_logistic_regression_model.py --data_src /Users/206819985/Documents/csci_444_assign/hw1-imdb ```

### Files Generated

test_predictions.csv

val_predictions.csv

Results
<img width="804" alt="image" src="https://github.com/user-attachments/assets/658eb2d8-5a0a-4c93-9a92-39751bbaad0f">


## Problem2: Naive bayes with Continuous Features TF-IDF

## Command to Run:
``` python3 _bonus_2.py --data_src <path_to_data_folder>```

### In my case
 ``` python3 _bonus_2.py --data_src /Users/206819985/Documents/csci_444_assign/hw1-imdb ```

### Files Generated

test_predictions.csv

val_predictions.csv

Results

<img width="735" alt="image" src="https://github.com/user-attachments/assets/7f04a46b-dd6e-42ea-bbbd-b342df92b8e6">

# Problem3: Syntactic Features

NOTE: Since the code was taking more than 7 hrs to run, I created a dataset with a total of just 100 samples, 70 text files were in the train folder, 15 were in test and 15 was in the validation.  And using the data set provided by the professor, I manually update the  train_labels.csv, val_labels.csv, test_paths.csv corresponding to the same labels and data provided by the professor.

## Command to Run:
> python -m spacy download en_core_web_sm

> python3 _bonus_3.py --data_src <path_to_your_data_folder>

> python3 _bonus_3.py --data_src /Users/206819985/Documents/csci_444_assign/hw1-imdb/_bonus/data

### In my case
 ``` python3 _bonus_1_logistic_regression_model.py --data_src /Users/206819985/Documents/csci_444_assign/hw1-imdb ```

### Files Generated

test_predictions.csv

val_predictions.csv

Results
<img width="804" alt="image" src="https://github.com/user-attachments/assets/658eb2d8-5a0a-4c93-9a92-39751bbaad0f">
