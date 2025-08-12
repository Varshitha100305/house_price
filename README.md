//Title: House Price Prediction using Regression Model//

1. Project Overview
Purpose:
This project predicts the selling price of houses based on various features such as **area, number of bedrooms, bathrooms, floors, year built, location, condition, and garage availability**.

It uses *Machine Learning (Linear Regression)* to learn patterns from historical house sales data and then make predictions for new houses.

Why it’s useful:

a. Helps **real estate agents** estimate property values quickly.
b. Assists **buyers and sellers** in making informed decisions.
c. Can be used for **market analysis** and forecasting trends.
d. Saves **time and effort** compared to manual market evaluation.


2. How it Works
The workflow is:

1. **Data Loading**

   * Reads the dataset (`house_prices.csv`) using `pandas`.
   * Displays first 5 rows, dataset info, basic statistics, and missing value counts.

2. **Data Preprocessing**

   * **Scaling numerical features** using `StandardScaler` to normalize data (ensures features like *Area* and *Bedrooms* are on the same scale).
   * **Encoding categorical features** like *Location*, *Condition*, *Garage* into numeric format using `OneHotEncoder`.

3. **Splitting Data**

   * Divides the dataset into training (80%) and testing (20%) sets using `train_test_split`.

4. **Model Creation**

   * Uses a `Pipeline` that first applies the preprocessing steps and then trains a **Linear Regression** model.

5. **Model Training**

   * Fits the model on training data (`model.fit()`).

6. **Model Evaluation**

   * Predicts on the test set.
   * Calculates performance metrics:

     * **MSE (Mean Squared Error)** → Average squared difference between predicted and actual prices.
     * **RMSE (Root Mean Squared Error)** → Easier to interpret since it’s in the same unit as the price.
     * **R² Score** → Shows how much variance in prices is explained by the model.

7. **Prediction on New Data**

   * Creates a new dataset (`new_df`) for unseen houses.
   * Predicts their prices using the trained model.



3. Step-by-Step Code Breakdown

Step 1: Import Libraries

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

```

* **Pandas:** Handle data loading and manipulation.
* **StandardScaler:** Normalize numeric features.
* **OneHotEncoder:** Convert categories into numeric format.
* **train\_test\_split:** Divide dataset into training and testing.
* **LinearRegression:** Machine learning model.
* **ColumnTransformer & Pipeline:** Organize preprocessing + model training in one flow.
* **NumPy:** Numeric operations.
* **Metrics:** Evaluate model performance.

Step 2: Load & Inspect Data

```python
df = pd.read_csv('house_prices.csv')
print(df.head())
df.info()
df.describe()
df.isnull().sum()
```

* Reads the CSV.
* Shows the first few rows, data types, stats, and missing values.

Step 3: Scale Numerical Features

```python
numerical_features = ['Area', 'Bedrooms', 'Bathrooms', 'Floors', 'YearBuilt', 'Price']
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])
```

 Normalizes values so large numbers (like *Area*) don’t overpower small ones (like *Bedrooms*).

Step 4: Preprocessing & Model Setup

```python
categorical_features = ['Location', 'Condition', 'Garage']
numerical_features = ['Id', 'Area', 'Bedrooms', 'Bathrooms', 'Floors', 'YearBuilt']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', LinearRegression())])
```

* **ColumnTransformer:** Applies different transformations to numeric and categorical features.
* **Pipeline:** Chains preprocessing and model together for cleaner, reproducible code.


Step 5: Split Data

```python
X = df.drop('Price', axis=1)
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

* **X:** Features used for prediction.
* **y:** Target (house price).
* 80% → training, 20% → testing.

Step 6: Train Model

```python
model.fit(X_train, y_train)
```

* Fits the regression model on training data.

Step 7: Evaluate Model

```python
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
```

* **MSE:** Lower = better.
* **RMSE:** Same unit as price, easier to interpret.
* **R²:** Closer to 1 means better fit.



Step 8: Predict on New Data

```python
new_data = { ... }
new_df = pd.DataFrame(new_data)
new_predictions = model.predict(new_df)
```

* Accepts new house features.
* Outputs predicted prices.

4. How This Helps

* **For sellers:** Suggests fair listing prices.
* **For buyers:** Helps spot overpriced properties.
* **For agents:** Automates property valuation.
* **For investors:** Supports decision-making in flipping/renting.


5. Possible Improvements

* Use more advanced models (Random Forest, XGBoost) for higher accuracy.
* Include more features (nearby schools, crime rate, public transport access).
* Perform feature selection to remove irrelevant attributes.
* Use cross-validation for more reliable accuracy estimation.

  **Flowchart**
<img width="493" height="733" alt="{450D6EC4-EFC8-44E3-B888-8EF8B112F270}" src="https://github.com/user-attachments/assets/f71e10cb-a156-4b67-b7e3-1b0cbcc95f93" />



