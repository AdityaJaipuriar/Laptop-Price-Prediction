# Laptop Price Predictor

An end-to-end Machine Learning web application designed to predict laptop market values based on hardware specifications, achieving an R2 score of 0.884.

## Project Overview
The pricing of laptops is influenced by a complex interplay of hardware variables. This project implements a high-performance regression pipeline that processes unstructured technical specifications to provide accurate price estimations.

## Technical Implementation
- **Feature Engineering**: Implemented advanced string parsing to extract CPU and GPU brands. Calculated Pixels Per Inch (PPI) as a primary feature to replace raw resolution data, improving model sensitivity to display quality.
- **Model Architecture**: Utilized an optimized XGBoost Regressor. Conducted hyperparameter tuning via GridSearchCV to maximize predictive accuracy.
- **Data Pipeline**: Integrated a Scikit-Learn ColumnTransformer for seamless preprocessing of categorical and numerical features.
- **Target Transformation**: Applied Log Transformation (log1p) to the target variable to normalize price distribution and reduce the impact of high-end outliers.
- **Deployment**: Developed a Flask based website to serve the model for real-time inference.

## Performance Metrics
- R2 Score: 0.884
- Mean Absolute Error (MAE): 9328 units

## Web interface of the project
<img width="1110" height="914" alt="image" src="https://github.com/user-attachments/assets/ff6a49af-3c46-4243-8053-822c059b4456" />
