![Photo by <a href="https://unsplash.com/photos/FPt10LXK0cg">ROBIN WORRALL</a> on <a href="https://unsplash.com/photos/qDgTQOYk6B8">Unsplash</a>
  ](./images/robin-worrall-FPt10LXK0cg-unsplash.jpg)

# Predicting SyriaTel Customer Churn

**Author**: <br>[Taylor Hale Robert](mailto:taylorhale11@gmail.com)


## Overview

This project uses a customer account information to predict churn rate.

## Data Sources & Features

[Kaggle Customer Churn](https://www.kaggle.com/becksddf/churn-in-telecoms-dataset) data was used for our analysis

'Churn' is the target variable for this analysis. 

View EDA code here.

![eda plot](./images/EDA.png)

## Summary
### Baseline Model
![baseline regression](./images/initial_models.png)
Built initial models using:<br><br>
• Logistic Regression • K Nearest Neighbors • Naive Bayes • Linear SVM •<br>
• RBF SVM • Decision Tree • Random Forest • Gradient Boosting • AdaBoost • XGBoost • <br><br>
XGBoost was among the lowest regarding testing and training time and among the highest regarding predictive power, correctly classifying 95% of the testing set.
   

### Observations
Observations here

### Final Model
Final model information here

## For More Information

See the full analysis in the [Jupyter Notebook](./final_model.ipynb) or review this [presentation](./Churn_Presentation.pdf).

For additional info, contact the author at:<br>
[Taylor Hale Robert](mailto:taylorhale11@gmail.com)


## Repository Structure

```
├── EDA.ipynb
├── final_model.ipynb
├── CODE
│   ├── THR_prelim_modeling.ipynb
│   └── more scratchwork
├── data
│   └── customer_churn.csv
├── images
│   ├── nguyen-dang-hoang-nhu-qDgTQOYk6B8-unsplash.jpg
│   ├── dom-fou-YRMWVcdyhmI-unsplash.jpg
│   ├── EDA.png
│   └── baseline.png
├── .gitignore
├── README.md
└── Churn_Presentation.pdf
