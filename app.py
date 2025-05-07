import os
from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Define class labels
categories = ['Normal', 'Glaucoma', 'Cataract', 'Diabetic Retinopathy']
IMG_SIZE = 128

# Load model
model = tf.keras.models.load_model("eye_disease_model.h5")

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def predict_image(image_path):
    image_array = preprocess_image(image_path)
    predictions = model.predict(image_array)
    predicted_index = np.argmax(predictions)
    predicted_class = categories[predicted_index]
    return predicted_class

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            predicted_class = predict_image(file_path)
            image_url = url_for('static', filename='uploads/' + file.filename)
            return render_template('index.html', prediction=predicted_class, image_url=image_url)
    return render_template('index.html', prediction=None, image_url=None)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
