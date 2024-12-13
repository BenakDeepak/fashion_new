import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle
import pandas as pd
import requests
from io import BytesIO
from PIL import Image

# Load the pre-trained ResNet50 model and add a GlobalMaxPooling2D layer
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
        result = model.predict(preprocessed_img).flatten()
        normalized_result = result / norm(result)
        return normalized_result
    except Exception as e:
        print(f"Error processing URL {image_url}: {e}")
        return None

# Load the CSV file containing filenames
csv_file = 'data.csv'
data = pd.read_csv(csv_file)
base_url = 'http://100.117.130.24:8080/image/original/'

# Generate full URLs by appending filenames to the base URL
data['image_url'] = base_url + data['filename'].astype(str)

# Load existing embeddings and filenames if they exist
if os.path.exists('embeddings.pkl') and os.path.exists('filenames.pkl'):
    with open('embeddings.pkl', 'rb') as f:
        feature_list = pickle.load(f)
    with open('filenames.pkl', 'rb') as f:
        filenames = pickle.load(f)
else:
    feature_list = []
    filenames = []

# Find new URLs that are not already processed
new_files = data[~data['filename'].isin(filenames)]

# Extract features for new URLs and update the lists
for _, row in tqdm(new_files.iterrows(), total=new_files.shape[0]):
    image_url = row['image_url']
    features = extract_features_from_url(image_url, model)
    if features is not None:
        feature_list.append(features)
        filenames.append(row['filename'])

# Save the updated embeddings and filenames
with open('embeddings.pkl', 'wb') as f:
    pickle.dump(feature_list, f)
with open('filenames.pkl', 'wb') as f:
    pickle.dump(filenames, f)

print("Embeddings and filenames updated successfully.")
