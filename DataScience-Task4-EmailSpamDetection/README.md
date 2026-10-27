# Email / SMS Spam Detection with Machine Learning

**Oasis Infobyte SIP | Data Science | Task 4**

An NLP binary classifier that separates spam from legitimate (ham) messages using TF-IDF features.

## Dataset
SMS Spam Collection (`spam.csv`): 5,572 labelled messages (4,825 ham, 747 spam). 403 exact duplicates were removed before splitting (5,169 messages left: 4,516 ham, 653 spam), so no message can appear in both training and test sets. Spam is only about 13% of the data (imbalanced).

## What was done
1. **Data loading and class distribution check** (counts and percentages), duplicate removal
2. **EDA:** message length by class (spam messages are much longer)
3. **Text preprocessing:** lowercase, punctuation removal, stopword removal (done with `re` and a stopword list)
4. **TF-IDF** feature extraction (unigrams and bigrams), with an explanation of what TF-IDF measures
5. **Train/test split:** 80/20, stratified
6. **Models (3):** Multinomial Naive Bayes, Logistic Regression, Linear SVM
7. **Evaluation:** accuracy, precision, recall, F1, confusion matrix, classification report, plus 5-fold cross-validation
8. **Discussion:** why recall matters for spam detection (and why precision matters too)
9. **Bonus:** most common words in spam vs ham (bar charts) and the words most indicative of spam

## Results
| Model | Accuracy | Precision (spam) | Recall (spam) | F1 (spam) | CV F1 |
|---|---|---|---|---|---|
| Linear SVM | 97.9% | 95.0% | 87.8% | 91.3% | 93.5% |
| Multinomial Naive Bayes | 98.0% | 96.6% | 87.0% | 91.6% | 93.2% |
| Logistic Regression | 97.3% | 89.9% | 88.6% | 89.2% | 90.7% |

**Best model: Linear SVM** (highest cross-validated F1); Naive Bayes is practically tied and has the fewest false alarms. Typical spam words: *claim, prize, guaranteed, tone, 150p*.

## Project structure
```
DataScience-Task4-EmailSpamDetection/
├── Spam_Detection.ipynb   # full analysis, commented, with outputs
├── spam.csv
├── README.md
├── requirements.txt
└── images/
```

## How to run
```bash
pip install -r requirements.txt
jupyter notebook Spam_Detection.ipynb
```

## Note
The dataset consists of SMS messages (the classic dataset recommended for this task). Email spam filters use the same technique, but results will differ on emails.
