#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Load the data
data = pd.read_csv('outputttt.csv')

# Remove special characters from the month column
data['month'] = data['month'].str.replace(r'\W', '', regex=True)

# Convert the cleaned month column to datetime format
data['month'] = pd.to_datetime(data['month'], format='%Y%m%d', errors='coerce')

# Handle missing values (if any)
data.dropna(inplace=True)

# Display the cleaned data
print(data.head())


# In[ ]:





# In[2]:


import matplotlib.pyplot as plt

# Plot sales over time
plt.figure(figsize=(10, 6))
plt.plot(data['month'], data['sales'])
plt.title('Sales Over Time')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.show()

# Summary statistics
print(data.describe())


# In[3]:


# Extract year, month, and day as separate features
data['year'] = data['month'].dt.year
data['month_num'] = data['month'].dt.month
data['day'] = data['month'].dt.day

# Drop the original date column
data = data.drop(columns=['month'])

# Display the data with new features
print(data.head())


# In[4]:


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np


# In[5]:


# Prepare the data
X = data.drop(columns=['sales'])
y = data['sales']



# In[6]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[7]:


# Initialize the Random Forest Regressor
rf = RandomForestRegressor(random_state=42)



# In[8]:


# Define parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}



# In[13]:


# Grid search
#grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)



# In[10]:


# Best model
best_rf = grid_search.best_estimator_



# In[11]:


# Predict on the test data
y_pred = best_rf.predict(X_test)



# In[12]:


# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'Best Model RMSE: {rmse}')



# In[14]:


# Feature Importance
importances = best_rf.feature_importances_
features = X.columns



# In[15]:


# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(features, importances)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances')
plt.show()


# In[12]:


from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.datasets import make_classification

# Generate sample data
X_train, y_train = make_classification(n_samples=100, n_features=20, random_state=42)

# Train the Random Forest model
best_rf = RandomForestClassifier()
best_rf.fit(X_train, y_train)

# Save the trained model
joblib.dump(best_rf, 'random_forest_model.joblib')

# Train the model
best_rf = RandomForestClassifier()
best_rf.fit(X_train, y_train)


# In[13]:


from fastapi import FastAPI
from pydantic import BaseModel
import joblib


# In[14]:


#!pip uninstall typing_extensions
#!pip install typing_extensions
 


# In[15]:


# Save the best model
joblib.dump(best_rf, 'random_forest_model.joblib')



# In[16]:


# Load the trained model
model = joblib.load('random_forest_model.joblib')


# In[17]:


# Initialize FastAPI app
app = FastAPI()


# In[20]:


uvicorn.run()


# In[7]:


# Define the request body
class PredictionRequest(BaseModel):
   year: int
   month_num: int
   day: int


# In[ ]:





# In[8]:


# Define the prediction endpoint
@app.post('/predict')
def predict(request: PredictionRequest):
 # Convert request data to DataFrame
    data = pd.DataFrame([request.dict().values()], columns=request.dict().keys())
 # Make prediction
    prediction = model.predict(data)
 # Return the prediction
    return {"prediction": prediction.tolist()}


# In[9]:


# Run the API with Uvicorn
if __name__ == "__main__":
   import uvicorn
uvicorn.run(app, host='0.0.0.0', port=8000)


# In[1]:


from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from IPython.display import HTML


app = FastAPI()

class PredictionRequest(BaseModel):
    feature1: float
    feature2: float
    # Add other features as needed

# Load the trained model
model = joblib.load('random_forest_model.joblib')

@app.post('/predict')
async def predict(request: PredictionRequest):
    # Convert request data to DataFrame
    data = pd.DataFrame([request.dict().values()], columns=request.dict().keys())
    
    # Make predictions
    prediction = model.predict(data)
    
    # Return the prediction
    return {"prediction": prediction.tolist()}

if __name__ == "__main__":
    display(HTML("<a href='http://localhost:8000/docs' target='_blank'>Open API Docs</a>"))
    uvicorn.run

   


# In[1]:


from flask import Flask
app = Flask(__name__)

@app.route('/')
def home():
    return "Hello, Flask!"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)


# In[1]:


from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

class PredictionRequest(BaseModel):
    feature1: float
    feature2: float
    # Add other features as needed

# Load the trained model
model = joblib.load('random_forest_model.joblib')

@app.post('/predict')
async def predict(request: PredictionRequest):
    # Convert request data to DataFrame
    data = pd.DataFrame([request.dict().values()], columns=request.dict().keys())
    
    # Make predictions
    prediction = model.predict(data)
    
    # Return the prediction
    return {"prediction": prediction.tolist()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

