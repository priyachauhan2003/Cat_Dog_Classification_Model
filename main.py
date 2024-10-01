from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

# Initialize the FastAPI app
app = FastAPI()

# Load the pre-trained model
model = load_model('cat_dog_classification_model.h5')

# Image preprocessing function (resize to 100x100, normalize to range 0-1)
def preprocess_image(image: Image.Image):
    image = image.resize((100, 100))  # Resize the image to 100x100 as required by the model
    image = np.array(image) / 255.0   # Normalize the pixel values to [0,1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension (1, 100, 100, 3)
    return image

# Define the prediction endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read the file and convert it into an image
    image = Image.open(io.BytesIO(await file.read()))

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Make a prediction using the loaded model
    prediction = model.predict(preprocessed_image)

    # Post-process the prediction (assuming binary classification: 0 for dog, 1 for cat)
    if prediction[0] > 0.5:
        return {"prediction": "Cat", "confidence": float(prediction[0])}
    else:
        return {"prediction": "Dog", "confidence": float(1 - prediction[0])}

# Run the app using uvicorn in terminal or command prompt
# Example: uvicorn main:app --reload
