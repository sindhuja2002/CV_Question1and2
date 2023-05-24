from flask import Flask, render_template, request
import onnxruntime
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load the ONNX model
onnx_model_path = 'model.onnx'
session = onnxruntime.InferenceSession(onnx_model_path)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
class_labels = ['Bean','Bitter_Gourd','Bottle_Gourd','Brinjal','Broccoli','Cabbage','Capsicum','Carrot','Cauliflower','Cucumber','Papaya','Potato','Pumpkin','Radish','Tomato']

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for image prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image from the request
    image = request.files['image']
    
    # Open and preprocess the image
    img = Image.open(image)
    img = img.resize((224, 224))  # Resize the image if needed
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array.astype('float32')
    img_array /= 255.0  # Normalize pixel values
    
    # Run the ONNX model to get predictions
    result = session.run([output_name], {input_name: img_array})
    predictions = result[0][0]
    
    # Get the predicted class label
    predicted_label = class_labels[np.argmax(predictions)]
    
    return render_template('index.html', prediction=predicted_label)

if __name__ == '__main__':
    app.run()
