# House price prediction machine learning project using a regression model.

import pandas as pd # type: ignore

df = pd.read_csv('house_prices.csv')
print("First 5 rows of the dataframe:")
print(df.head())

print("\nData Info:")
df.info()
print("\nDescriptive Statistics:")
print(df.describe())
print("\nMissing values per column:")
print(df.isnull().sum())

from sklearn.preprocessing import StandardScaler # type: ignore

numerical_features = ['Area', 'Bedrooms', 'Bathrooms', 'Floors', 'YearBuilt', 'Price']

scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

print("\nFirst 5 rows after scaling numerical features:")
print(df.head())

from sklearn.model_selection import train_test_split # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.preprocessing import OneHotEncoder # type: ignore
from sklearn.compose import ColumnTransformer # type: ignore
from sklearn.pipeline import Pipeline # type: ignore
import numpy as np # type: ignore


categorical_features = ['Location', 'Condition', 'Garage']
numerical_features = ['Id', 'Area', 'Bedrooms', 'Bathrooms', 'Floors', 'YearBuilt'] # Exclude 'Price' as it's the target

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough' # Keep any other columns (like 'Id')
)

model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', LinearRegression())])


X = df.drop('Price', axis=1)
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error, r2_score # type: ignore

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R2): {r2}")

y_pred = model.predict(X_test)

new_data = {
    'Id': [2001, 2002, 2003],
    'Area': [1500, 3000, 2000],
    'Bedrooms': [3, 4, 2],
    'Bathrooms': [2, 3, 2],
    'Floors': [1, 2, 3],
    'YearBuilt': [2010, 1995, 2020],
    'Location': ['Urban', 'Suburban', 'Downtown'],
    'Condition': ['Good', 'Excellent', 'Fair'],
    'Garage': ['Yes', 'No', 'Yes']
}

new_df = pd.DataFrame(new_data)


new_predictions = model.predict(new_df)

print("\nNew data for prediction:")
print(new_df)
print("\nPredictions on new data:")
print(new_predictions)


