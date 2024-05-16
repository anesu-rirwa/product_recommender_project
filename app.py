import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import requests
from io import BytesIO

# Load test data for seeing current image
test_data = pickle.load(open('test_data.pkl','rb'))
test_data_ = pd.DataFrame(test_data)

# Load train data for seeing recommendations
train_data = pickle.load(open('img_data.pkl','rb'))
train_data_ = pd.DataFrame(train_data)

# Load model
knn = pickle.load(open('model_recommend.pkl','rb'))

# Load TF-IDF array for text
X_test = pickle.load(open('test_array.pkl','rb'))

st.title("Fashion Recommendation system")

st.header('Goal:')
st.markdown("To develop a fashion recommender system. By leveraging machine learning techniques, the system aims to offer personalized recommendations based on user preferences and budget constraints.")

st.header('Achievements:')
st.markdown("- Developed a Fashion Recommendation System tailored providing access to affordable and fashionable clothing.")
st.markdown("- Leveraged machine learning techniques to provide personalized recommendations based on user preferences and budget constraints.")
st.markdown("- Created an intuitive user interface using Streamlit, allowing users to easily search for products and receive recommendations.")

st.header('About Recommendation model:')
st.markdown("The model used is Nearest Neighbors, which provides similar products based on the input product's title, color, and brand. For a given data point it gives "
            "us similar points within the neighbourhood. Here, for a given women wear we get 10 more recommendations.")

st.header('Data:')
st.markdown("The data used for this project consists from Amazon.com and contains information about clothing products. The data includes the product's title, brand, color, price, and image URL. The data was used to train a machine learning model that provides recommendations based on the input product's title, color, and brand.")

st.header('How to use:')

title_current = st.selectbox('Search for the product you want here or Select a product from the dropdown list:',
                    list(test_data_['title']))
product = test_data_[(test_data_['title'] == title_current)]
s1 = product.index[0]
captions = [test_data_['brand'].values[s1],test_data_['formatted_price'].values[s1]]
c1,c2,c3 = st.columns(3)
with c1:
    st.image(test_data_['medium_image_url'].values[s1])
with c2:
    st.text(f'Brand:') 
    st.text('Color:')
    st.text('Price ($) ')
with c3:
    st.text(test_data_['brand'].values[s1])
    st.text(test_data_['color'].values[s1])
    st.text(test_data_['formatted_price'].values[s1])

distances, indices = knn.kneighbors([X_test.toarray()[s1]])
result1 = list(indices.flatten())[:5]
result2 = list(indices.flatten())[5:]

if st.button('Click here to get recommendations'):
    st.success('Hope you like the below recommendations :)')
    col1,col2,col3,col4,col5 = st.columns(5)
    lst1 = [col1,col2,col3,col4,col5]
    for i,j in zip(lst1,result1):
        with i:
            st.text(train_data_['brand'].values[j])
            st.text(train_data_['color'].values[j])
            st.image(train_data_['medium_image_url'].values[j])

    col6, col7, col8, col9, col10 = st.columns(5)
    lst2 = [col6, col7, col8, col9, col10]
    for k,l in zip(lst2,result2):
        with k:
            st.text(train_data_['brand'].values[l])
            st.text(train_data_['color'].values[l])
            st.image(train_data_['medium_image_url'].values[l])
