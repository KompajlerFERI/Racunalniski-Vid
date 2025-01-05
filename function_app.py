import logging
import azure.functions as func
import torch
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
from io import BytesIO
#from PeopleClassifier import PeopleClassifier

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

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

@app.route(route="PeopleRecognizer")
def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PeopleClassifier().to(device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()

    # Define the transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    try:
        # Get the image from the request
        image_data = req.get_body()

        # Log the first few bytes of the image data to check it's valid
        logging.info(f"Received image data: {image_data[:20]}...")  # Log the first 20 bytes

        # Try to open the image with PIL
        try:
            image = Image.open(BytesIO(image_data)).convert("RGB")
        except UnidentifiedImageError:
            logging.error('Cannot identify image file')
            return func.HttpResponse(
                "Cannot identify image file.",
                status_code=400
            )

        # Log image size and format
        logging.info(f"Image opened with size: {image.size}, format: {image.format}")

        # Apply the transformation to the image
        image = transform(image).unsqueeze(0).to(device)

        # Pass the image through the model
        with torch.no_grad():
            output = model(image)
        
        # Return the result
        return func.HttpResponse(
            str(output.item()),
            status_code=200
        )
    except Exception as e:
        logging.error(f"Error processing the request: {e}")
        return func.HttpResponse(
            "Failed to process the uploaded file.",
            status_code=400
        )