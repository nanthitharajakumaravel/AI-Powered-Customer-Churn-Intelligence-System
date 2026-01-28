# AI-Powered-Customer-Churn-Intelligence-System
Project Overview : This project implements an end-to-end Machine Learning pipeline to predict customer churn for a telecom company using Logistic regression, Random forest, Gradient boosting and XGBoosting models

## Models Implemented
- Logistic Regression 
- Random Forest Classifier
- Gradient Boosting Classifier
- XGBoost Classifier

Each model consists of
- Data preprocessing
- Feature encoding
- Baseline comparison
- Performance evaluation
- Business recommendations

## Evaluation Metrics
- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix
- ROC-AUC Curve
- Feature Importance visualization

## Model comparison
- Among the four models evaluated (Logistic Regression, Random Forest, Gradient Boosting, and XGBoost), Logistic Regression served as a strong baseline, offering good interpretability but limited ability to capture non-linear patterns in customer behavior.
- Random Forest improved performance over Logistic Regression by modeling feature interactions, though it showed slightly lower recall for churned customers.
- XGBoost further enhanced predictive power with better handling of complex relationships and consistent improvement over the baseline model.
- Gradient Boosting achieved the best overall performance, delivering the highest accuracy (~80.05%) and ROC-AUC (~0.85).
- Overall, Gradient Boosting is the best-performing model, achieving the highest accuracy and ROC-AUC, making it the most suitable choice for churn prediction.
