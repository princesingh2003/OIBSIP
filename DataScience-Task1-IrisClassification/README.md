# Iris Flower Classification

**Oasis Infobyte SIP | Data Science | Task 1**

Machine learning classifiers that identify an iris flower's species (*Setosa*, *Versicolor*, *Virginica*) from its sepal and petal measurements.

## Dataset
Built into scikit-learn (`sklearn.datasets.load_iris`): 150 flowers, 4 features (sepal length/width, petal length/width in cm), 50 flowers per species. No missing values.

## What was done
1. **EDA:** shape, data types, null and duplicate check, descriptive statistics, class balance
2. **Visualisation:** pairplot by species, box plots for each feature, correlation heatmap
3. **Feature selection:** ANOVA F-test + Random Forest importance. Petal length and petal width are the most discriminative features; sepal width is the weakest
4. **Train/test split:** 80/20, stratified, `random_state=42`
5. **Models trained (4):** Logistic Regression, K-Nearest Neighbours, Decision Tree, Random Forest (scaling applied where needed)
6. **Evaluation:** accuracy, confusion matrix, classification report (precision / recall / F1) and 5-fold stratified cross-validation

## Results
| Model | Test accuracy (30 flowers) | 5-fold CV mean (150 flowers) |
|---|---|---|
| K-Nearest Neighbours | 93.3% | **97.3%** |
| Random Forest | 90.0% | 96.0% |
| Decision Tree | **96.7%** | 96.0% |
| Logistic Regression | 93.3% | 95.3% |

**Best model: K-Nearest Neighbours** (highest cross-validated accuracy). Setosa is classified perfectly by all models; all mistakes are Versicolor/Virginica mix-ups. The test set is small (30 flowers), so cross-validation is used for a more reliable comparison. The differences between models are small.

## Project structure
```
DataScience-Task1-IrisClassification/
├── Iris_Classification.ipynb   # full analysis, commented, with outputs
├── README.md
├── requirements.txt
└── images/                     # saved charts from the notebook
```

## How to run
```bash
pip install -r requirements.txt
jupyter notebook Iris_Classification.ipynb
```

## Tech stack
Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, Jupyter Notebook
