# Sales Prediction Using Python

**Oasis Infobyte SIP | Data Science | Task 5**

Regression models that predict product sales from advertising spend on TV, Radio and Newspaper, and an analysis of which channel has the highest impact.

## Dataset
`Advertising.csv`: 200 markets, columns TV, Radio, Newspaper (advertising budgets) and Sales. No missing values or duplicates.

## What was done
1. **EDA:** shape, types, null/duplicate check, descriptive statistics, pairplot, Sales vs each channel scatter plots, correlation heatmap
2. **Train/test split:** 80/20, `random_state=42`
3. **Models:** Linear Regression (baseline), Polynomial Regression (degree 2), Random Forest Regressor
4. **Evaluation:** MAE, RMSE, R-squared on the test set, plus 5-fold cross-validated R-squared
5. **Residual analysis** of the best model (actual vs predicted, residual plot, residual distribution)
6. **Interpretation:** standardised coefficients, raw coefficients and permutation importance

## Results
| Model | MAE | RMSE | R2 (test) | R2 (5-fold CV) |
|---|---|---|---|---|
| Polynomial Regression (deg 2) | 0.53 | 0.64 | 0.987 | 0.983 |
| Random Forest Regressor | 0.61 | 0.74 | 0.983 | 0.980 |
| Linear Regression (baseline) | 1.46 | 1.78 | 0.899 | 0.883 |

- **Best model:** Polynomial Regression (degree 2). The Random Forest is practically tied.
- The linear baseline misses the **TV x Radio interaction**: adding that one term raises its cross-validated R-squared from 0.88 to 0.96.
- **TV has the highest overall impact**, **Radio gives the most sales per unit of spend** (about 4x TV in the linear model), and **Newspaper has almost no impact** once TV and Radio are included.
- Limitations: 200 rows only, observational data (association, not proven causation).

## Project structure
```
DataScience-Task5-SalesPrediction/
├── Sales_Prediction.ipynb   # full analysis, commented, with outputs
├── Advertising.csv
├── README.md
├── requirements.txt
└── images/
```

## How to run
```bash
pip install -r requirements.txt
jupyter notebook Sales_Prediction.ipynb
```
