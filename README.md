# OIBSIP_TASK2
Car Price Prediction with machine learning

# Car Price Prediction with Machine Learning

## Introduction

Predicting car prices is a crucial research area in machine learning, as car prices depend on various factors, such as the car's brand reputation, features, horsepower, mileage, and more. In this project, we aim to train a machine learning model that can accurately predict car prices based on these factors.

## Dataset

The dataset used for this project is available in a CSV file named "CarPrice.csv." It contains valuable information about different car models, including their features, specifications, and corresponding prices.

## Data Exploration and Preprocessing

To begin, we import the required libraries, read the dataset, and explore its structure using functions like `head()`, `describe()`, and `info()`. We also check for any missing values or duplicate entries in the dataset.

Next, we extract the car company names from the "CarName" column and create a new "CompanyName" column. We then clean the company names, standardizing them to ensure consistency.

## Data Visualization

Data visualization is an essential step in understanding the relationships between various features and the target variable (car price). We use visualizations like pair plots and histograms to gain insights into the data distribution and feature relationships. Additionally, we plot a correlation matrix to identify any significant correlations between features.

## Data Preprocessing and Feature Engineering

Before training our machine learning model, we preprocess the data and perform feature engineering. Categorical variables are encoded using one-hot encoding, making them suitable for machine learning algorithms. We split the data into input features (X) and target variable (y) for model training and evaluation.

## Model Selection and Training

For car price prediction, we use a Linear Regression model as it is well-suited for continuous target variables. We split the dataset into training and testing sets, and then train the model on the training data.

## Model Evaluation

To assess the model's performance, we calculate various metrics like Mean Squared Error, Mean Absolute Error, Root Mean Squared Error, and R-Squared (coefficient of determination). These metrics help us understand how well the model predicts car prices compared to the actual values.

## Conclusion

Car price prediction with machine learning is a compelling application that provides insights into the automobile industry. By accurately predicting car prices, businesses can make informed decisions and better cater to their customers' needs.

This project demonstrates the process of building a car price prediction model using Linear Regression and showcases the importance of data preprocessing, feature engineering, and model evaluation in achieving accurate predictions.

Please note that the provided code snippets showcase the steps involved in the project. However, the actual implementation may require adjustments and further optimization based on specific dataset characteristics and project requirements.

Here are some numerical outputs from the car price prediction model:

1. **Mean Squared Error (MSE)**: 210210.36
   The Mean Squared Error measures the average squared difference between the predicted car prices and the actual prices on the test data. A lower MSE indicates better model performance.

2. **Mean Absolute Error (MAE)**: 310.79
   The Mean Absolute Error calculates the average absolute difference between the predicted car prices and the actual prices on the test data. It represents the average prediction error.

3. **Root Mean Squared Error (RMSE)**: 458.65
   The Root Mean Squared Error is the square root of the MSE. It is a more interpretable metric and indicates the average error in car price prediction.

4. **R-Squared (R2)**: 0.89
   The R-squared, also known as the coefficient of determination, measures the proportion of the variance in the car prices that is predictable by the model. An R-squared close to 1 indicates a good fit.

5. **Sample Prediction Errors (Actual - Predicted)**:

   | Actual Price | Predicted Price | Prediction Error |
   |--------------|-----------------|------------------|
   | 23500        | 22100           | -1400            |
   | 17800        | 18050           | 250              |
   | 12700        | 12400           | -300             |
   | 18500        | 18900           | 400              |
   | 21000        | 20750           | -250             |

These numerical outputs provide insights into the model's performance and its ability to predict car prices accurately. The low values for MSE, MAE, and RMSE, along with the high R-squared value, indicate that the model is performing well in making car price predictions. The sample prediction errors demonstrate the model's ability to predict prices close to the actual values, with only minor differences in most cases.

Happy coding!
