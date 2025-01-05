import logging
from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
from io import BytesIO

import timm
import torch.nn as nn
import torch.nn.functional as F

class PeopleClassifier(nn.Module):
    #Definiramo vse dele mreže (uporabimo že nareto mrežo)
    def __init__(self):
        super(PeopleClassifier, self).__init__()
         # Use EfficientNet-B0 as the backbone
        self.base_model = timm.create_model("efficientnet_b0", pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])  # Remove the final layer

        # Add a regression head to predict the scalar value (number of people)
        enet_out_size = 1280  # EfficientNet-B0 output size
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(enet_out_size, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 1)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x

app = Flask(__name__)

# Load the model once at startup to avoid reloading it for every request
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PeopleClassifier().to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.route('/PeopleRecognizer', methods=['POST'])
def main():
    logging.info('Received a request for PeopleRecognizer webhook.')

    try:
        # Get the image data from the request
        image_data = request.data

        # Log the first few bytes of the image data to check it's valid
        logging.info(f"Received image data: {image_data[:20]}...")

        # Try to open the image with PIL
        try:
            image = Image.open(BytesIO(image_data)).convert("RGB")
        except UnidentifiedImageError:
            logging.error('Cannot identify image file')
            return jsonify({"error": "Cannot identify image file."}), 400

        # Log image size and format
        logging.info(f"Image opened with size: {image.size}, format: {image.format}")

        # Apply the transformation to the image
        image = transform(image).unsqueeze(0).to(device)

        # Pass the image through the model
        with torch.no_grad():
            output = model(image)

        # Assuming the output is a single value (you can adjust this based on your model)
        return jsonify({"result": output.item()}), 200

    except Exception as e:
        logging.error(f"Error processing the request: {e}")
        return jsonify({"error": "Failed to process the uploaded file."}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5005)