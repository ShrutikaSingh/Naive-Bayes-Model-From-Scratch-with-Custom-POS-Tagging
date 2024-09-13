

## Folder structure with Description

(venv) 206819985@OTS-FVFHK1DWQ05Q-MAC submissions % tree
.
├── K1000_main.py         -> This is Naive Bayes with vocab size = 1000

├── README.md             -> Contains Details for main assignment

├── Shrutika_CSCI444_1_MAIN_REPORT  (1).pdf   -> This is Report

├── main.py               -> This is main code for Naive Bayes with Vocab Size = 10000

├── requirements.txt      -> This is requirements for libraries for main hw (Bonus requirement file is separate in Bonus Folder)
├── test_paths.csv        
├── test_predictions.csv  -> Generated test_predictions.csv with headers as `review,sentiment Rows looks like this  ./test/354863.txt,0`
├── train_labels.csv
├── val_labels.csv
└── val_predictions.csv   -> Generated val_predictions.csv with headers as  `review,sentiment  Rows looks like this  ./val/597852.txt,1 `


## Folder Structure with Description

The following is the folder structure of the project along with descriptions for each file:

- **K1000_main.py**  
  This is Naive Bayes with vocab size = 1000

- **README.md**  
  Contains details for the main assignment

- **Shrutika_CSCI444_1_MAIN_REPORT (1).pdf**  
  This is the report

- **main.py**  
  This is the main code for Naive Bayes with vocab size = 10000

- **requirements.txt**  
  This file lists the libraries required for the main homework. (Bonus requirement file is separate in the Bonus Folder)

- **test_paths.csv**  
  Contains paths to test files.

- **test_predictions.csv**  
  Generated `test_predictions.csv` with headers as `review,sentiment`. Rows look like this: `./test/354863.txt,0`

- **train_labels.csv**  
  Contains training labels.

- **val_labels.csv**  
  Contains validation labels.

- **val_predictions.csv**  
  Generated `val_predictions.csv` with headers as `review,sentiment`. Rows look like this: `./val/597852.txt,1`


## RUNNING THE CODE

###  Step 1
cd into sumbmission folder

(venv) 206819985@OTS-FVFHK1DWQ05Q-MAC submissions % pwd
/Users/206819985/Documents/csci_444_assign/hw1-imdb/submissions

### Step 2
Copy the name of folder where your test, train and val data is located
In my case it is located at : /Users/206819985/Documents/csci_444_assign/hw1-imdb

## For Vocab Size = 10000

### Step 3
Run the following command
> python main.py --data_src <PATH_OF_DATA_FOLDER>

IN MY Case
>python3 main.py --data_src /Users/206819985/Documents/csci_444_assign/hw1-imdb


### Step 4
OUTPUT

<img width="968" alt="image" src="https://github.com/user-attachments/assets/b07a4e65-80d0-4b7e-8dd0-9107b8fbc769">


## For Vocal Size = 1000
python K1000_main.py --data_src <PATH_OF_DATA_FOLDER>

In My Case

python3 K1000_main.py --data_src /Users/206819985/Documents/csci_444_assign/hw1-imdb

<img width="1012" alt="image" src="https://github.com/user-attachments/assets/623fd155-b310-4f9c-b5c6-f52c7b73dd40">


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

DOnot remove not nor etc from stop words , accuracy increased
vocab length = 10000
Validation Accuracy: 0.8570
Validation Precision: 0.8653
Validation Recall: 0.8455
Validation F1: 0.8553

(venv) 206819985@OTS-FVFHK1DWQ05Q-MAC hw1-imdb % 

<img width="908" alt="image" src="https://github.com/user-attachments/assets/3c156184-9e2b-4be9-bf22-b88f1eb0fa2c">
**
When vocab length = 1000**

<img width="959" alt="image" src="https://github.com/user-attachments/assets/f36ddcfe-dbcd-4d5d-a0b7-41c780a60c22">
