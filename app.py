from flask import Flask, render_template, request, jsonify
import torch
from PIL import Image
from torchvision import transforms
import timm

app = Flask(__name__)

# Define the model architecture
def create_model():
    try:
        model = timm.create_model('resnet50', pretrained=True)
        model.fc = torch.nn.Sequential(
            torch.nn.Linear(2048, 256),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 4),
            torch.nn.Softmax(dim=1)
        )
        return model
    except Exception as e:
        print(f"Error in create_model: {e}")
        return None

# Load the saved model
def load_model(model_path):
    try:
        model = create_model()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading the model: {e}")
        return None

model_path = 'YourFlaskApp/model_ResNet50_best.h5'  # Update the model path
model = load_model(model_path)

# Image preprocessing function
def process_image(image):
    try:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(image).unsqueeze(0)
    except Exception as e:
        print(f"Error processing the image: {e}")
        return None

# Prediction function
def predict(image, model):
    if model is None:
        return None
    try:
        image = process_image(image)
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            return predicted.item()
    except Exception as e:
        print(f"Error making the prediction: {e}")
        return None

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        image = Image.open(file)
        label_index = predict(image, model)
        if label_index is not None:
            class_name = ['adenocarcinoma', 'large.cell.carcinoma', 'normal', 'squamous.cell.carcinoma']
            label = class_name[label_index]
            return jsonify({'prediction': label})
        else:
            return jsonify({'error': 'Error making the prediction'})

if __name__ == '__main__':
    app.run(debug=True)
