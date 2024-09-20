#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import pandas as pd


# In[15]:


data = pd.read_csv('outputttt.csv')


# In[16]:


data.head()


# In[17]:


data.drop('accID',axis=1,inplace= True)


# In[19]:


# Remove special characters from the month column
data['month'] = data['month'].str.replace(r'\W', '', regex=True)

# Convert the cleaned month column to datetime format
data['month'] = pd.to_datetime(data['month'], format='%Y%m%d', errors='coerce')

# Handle missing values (if any)
data.dropna(inplace=True)



# Extract year, month, and day as separate features
data['year'] = data['month'].dt.year
data['month_num'] = data['month'].dt.month
data['day'] = data['month'].dt.day

# Drop the original date column
data = data.drop(columns=['month'])

# Display the data with new features
print(data.head())


# In[23]:


x = data.iloc[: , :-1]
y = data.iloc[:,-1]


# In[24]:


# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[25]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(x_train , y_train)


# In[27]:


y_predict = classifier.predict(x_test)


# In[29]:


from sklearn.metrics import accuracy_score
score = accuracy_score(y_test , y_predict)


# In[30]:


score


# In[32]:


# pickling the model 

import pickle
pickle_out = open('classifier.pkl','wb')
pickle.dump(classifier , pickle_out)
pickle_out.close()

#pickle is continution of the data it package everything with  pkl extension 


# In[ ]:


import streamlit as st
from PTL import image


#loading the model to predict on the data 

pickle_in = open('classifier.pkl','rb')
classifer = pickle.load(pickle_in)


def welcome():
    return 'welocme all '


# In[ ]:


#defining the function which will make predictions using
# The data which the users inputs

def prediction(sales,strategy1, strategy2, strategy3,salesVisit1_pct, salesVisit2_pct, salesVisit3_pct, salesVisit4_pct, 
               salesVisit5_pct,alesVisit1_dollar, salesVisit2_dollar, salesVisit3_dollar, 
               salesVisit4_dollar, salesVisit5_dollar):
    prediction = classifier.predict(
            [[sales,strategy1, strategy2, strategy3,salesVisit1_pct, salesVisit2_pct, salesVisit3_pct, salesVisit4_pct, 
               salesVisit5_pct,alesVisit1_dollar, salesVisit2_dollar, salesVisit3_dollar, 
               salesVisit4_dollar, salesVisit5_dollar]])
    print(prediction)
    return prediction
    


# In[42]:


import pandas as pd
import numpy as np
import pickle
import streamlit as st
from PIL import Image   
   
# this is the main function in which we define our webpage 
def main():
     # giving the webpage a title
   st.title("drug analysis")
     
   # here we define some of the front end elements of the web page like 
   # the font and background color, the padding and the text to be displayed
   html_temp = """
   <div style ="background-color:yellow;padding:13px">
   <h1 style ="color:black;text-align:center;">Streamlit Iris Flower Classifier ML App </h1>
   </div>
   """
     
   # this line allows us to display the front end aspects we have 
   # defined in the above code
   st.markdown(html_temp, unsafe_allow_html = True)
     
   # the following lines create text boxes in which the user can enter 
   # the data required to make the prediction
   sepal_length = st.text_input("sales", "Type Here")
   sepal_width = st.text_input("pre_competitor", "Type Here")
   petal_length = st.text_input("post_competitor", "Type Here")
   petal_width = st.text_input(" CompBrand", "Type Here")
   result =""
     
   # the below line ensures that when the button called 'Predict' is clicked, 
   # the prediction function defined above is called to make the prediction 
   # and store it in the variable result
   if st.button("Predict"):
       result = prediction(sales, pre_competitor, post_competitor, CompBrand)
   st.success('The output is {}'.format(result))
    
if __name__=='__main__':
   main()


# In[36]:


import logging
logging.getLogger('streamlit').setLevel(logging.ERROR)


# In[40]:


from streamlit.runtime.scriptrunner import add_script_run_ctx
import threading

def your_thread_function():
    add_script_run_ctx(threading.current_thread())
    # Your thread code here

thread = threading.Thread(target=your_thread_function)
thread.start()




# In[38]:


pip install --upgrade streamlit


# In[44]:


output= lambda x: 'sales impact' if x == '0' else 'post_competitior' if x == '1' else 'post_competitor'
output(str(3))


# In[ ]:





# In[47]:


# Initialize and train the classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(x_train, y_train)

# Make predictions
y_pred = clf.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Example prediction
sample_input = x_test.iloc[0].values.reshape(1, -1)
predicted_sales = clf.predict(sample_input)
print(f'Predicted Sales: {predicted_sales[0]}')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




