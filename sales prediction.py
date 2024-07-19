import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# Create a sample dataset and save it to CSV
data = {
    'Item_Identifier': ['FDA15', 'DRC01', 'FDN15', 'FDX07', 'NCD19'],
    'Item_Weight': [9.3, 5.92, 17.5, 19.2, 8.93],
    'Item_Fat_Content': ['Low Fat', 'Regular', 'Low Fat', 'Regular', 'Low Fat'],
    'Item_Visibility': [0.016047301, 0.019278216, 0.016760075, 0.000, 0.000],
    'Item_Type': ['Dairy', 'Soft Drinks', 'Meat', 'Fruits and Vegetables', 'Household'],
    'Item_MRP': [249.8092, 48.2692, 141.618, 182.095, 53.8614],
    'Outlet_Identifier': ['OUT049', 'OUT018', 'OUT049', 'OUT010', 'OUT013'],
    'Outlet_Establishment_Year': [1999, 2009, 1999, 1998, 1987],
    'Outlet_Size': ['Medium', 'Medium', 'Medium', 'Small', 'High'],
    'Outlet_Location_Type': ['Tier 1', 'Tier 3', 'Tier 1', 'Tier 3', 'Tier 3'],
    'Outlet_Type': ['Supermarket Type1', 'Supermarket Type2', 'Supermarket Type1', 'Grocery Store', 'Supermarket Type1'],
    'Item_Outlet_Sales': [3735.138, 443.4228, 2097.270, 732.3800, 994.7052]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('BigMartSalesData.csv', index=False)

# Load the dataset from CSV
df = pd.read_csv('BigMartSalesData.csv')

# Data preprocessing
# Handle missing values
df['Item_Weight'] = df['Item_Weight'].fillna(df['Item_Weight'].mean())
df['Outlet_Size'] = df['Outlet_Size'].fillna(df['Outlet_Size'].mode()[0])

# Encode categorical variables
le = LabelEncoder()
df['Item_Fat_Content'] = le.fit_transform(df['Item_Fat_Content'])
df['Item_Type'] = le.fit_transform(df['Item_Type'])
df['Outlet_Size'] = le.fit_transform(df['Outlet_Size'])
df['Outlet_Location_Type'] = le.fit_transform(df['Outlet_Location_Type'])
df['Outlet_Type'] = le.fit_transform(df['Outlet_Type'])

# Feature engineering
df['Outlet_Age'] = 2024 - df['Outlet_Establishment_Year']
df.drop(['Item_Identifier', 'Outlet_Identifier', 'Outlet_Establishment_Year'], axis=1, inplace=True)

# Define features and target variable
X = df.drop('Item_Outlet_Sales', axis=1)
y = df['Item_Outlet_Sales']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model training and evaluation
# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print('Linear Regression:')
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred_lr)))
print('R^2 Score:', r2_score(y_test, y_pred_lr))

# Decision Tree Regressor
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print('Decision Tree Regressor:')
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred_dt)))
print('R^2 Score:', r2_score(y_test, y_pred_dt))

# Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print('Random Forest Regressor:')
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred_rf)))
print('R^2 Score:', r2_score(y_test, y_pred_rf))

# Support Vector Regressor
svr = SVR()
svr.fit(X_train, y_train)
y_pred_svr = svr.predict(X_test)
print('Support Vector Regressor:')
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred_svr)))
print('R^2 Score:', r2_score(y_test, y_pred_svr))

# Plot actual vs predicted values for Random Forest Regressor
plt.figure(figsize=(10, 7))
plt.scatter(y_test, y_pred_rf, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales (Random Forest)')
plt.show()
