import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import requests
from io import BytesIO
from PIL import Image
import pandas as pd

# Load feature list, filenames, and dataset
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))
dataset = pd.read_csv('data.csv')  # Assuming the dataset contains 'id' and 'filename' columns

# Load pre-trained ResNet50 model + higher-level layers
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Function to extract features from an image URL
def extract_features_from_url(image_url, model):
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        features = model.predict(preprocessed_img).flatten()
        normalized_features = features / norm(features)
        return normalized_features
    except Exception as e:
        st.error(f"Error processing URL {image_url}: {e}")
        return None

# Function to recommend images
def recommend(image_url, model, feature_list, filenames):
    features = extract_features_from_url(image_url, model)
    if features is not None:
        neighbors = NearestNeighbors(n_neighbors=4, algorithm='brute', metric='euclidean')
        neighbors.fit(feature_list)
        distances, indices = neighbors.kneighbors([features])
        return indices, features
    else:
        return None, None

# Streamlit app
st.title('Fashion Recommender System')

image_url_input = st.text_input("Enter the URL of an image:")
if image_url_input:
    st.write("Processing the image...")

    # Display the uploaded image
    try:
        response = requests.get(image_url_input)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert('RGB')
        st.image(img, caption='Input Image', use_column_width=True)

        # Get recommendations
        indices, input_features = recommend(image_url_input, model, feature_list, filenames)

        if indices is not None:
            # Exclude the input image from recommendations
            recommended_indices = [idx for idx in indices[0] if not np.allclose(input_features, feature_list[idx])]

            # Display recommended images with IDs and filenames
            st.write("Here are 3 recommended images:")
            for idx in recommended_indices[:3]:
                recommended_filename = filenames[idx]
                recommended_id = dataset[dataset['filename'] == recommended_filename]['id'].values[0]
                recommended_url = f"http://100.117.130.24:8080/image/original/{recommended_filename}"
                st.image(recommended_url, caption=f"ID: {recommended_id}, Filename: {recommended_filename}", use_column_width=True)
        else:
            st.write("No recommendations could be generated.")
    except Exception as e:
        st.error(f"Error fetching or displaying the image: {e}")
