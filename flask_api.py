#!/usr/bin/env python
# coding: utf-8

# In[9]:





# In[25]:





# In[3]:





# In[30]:


import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from keras.applications.vgg16 import preprocess_input, decode_predictions, VGG16
from flask import Flask, request, render_template

model = tf.keras.models.load_model('RiceBuddy_model.h5')
def decode_predict(predictions):
    # Daftar label kelas
    classes = ['BrownSpot', 'Healthy', 'Hispa', 'LeafBlast']
    
    # Mengambil indeks dengan probabilitas tertinggi
    predicted_index = np.argmax(predictions)
    
    # Mengambil label kelas yang sesuai
    predicted_class = classes[predicted_index]
    
    return predicted_class

app = Flask(__name__)

# Route untuk mendapatkan semua data
@app.route('/', methods=['GET'])
def hello_world():
  return render_template("index.html")

@app.route('/', methods=['POST'])
def predict():
  imagefile= request.files['imagefile']
  image_path = "./images/" + imagefile.filename
  imagefile.save(image_path)

  image = load_img(image_path, target_size=(100,100))
  image = img_to_array(image)
  image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
  image = preprocess_input(image)
  yhat = model.predict(image)
  label = decode_predict(yhat)
  

  return render_template("index.html", prediction=label)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port= 5000, debug= False)


# In[ ]:




