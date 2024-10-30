import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

from evaluate_model import EmotionModel

# Load your model (make sure to adjust the path to your model)
model_path = 'models\\fer_model.pth'  # Change this to your model path
model = EmotionModel()  # Make sure your EmotionModel class is defined
model.load_state_dict(torch.load(model_path))
model.eval()

# Define the transformations
transform = transforms.Compose([
    transforms.Resize((48, 48)),  # Resize to the input size of the model
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.5,), (0.5,)),  # Normalize if needed
])

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Emotion labels (make sure these match your training labels)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Start webcam capture
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    # Convert the frame (NumPy array) to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    # Detect faces
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Prepare the face region for prediction
        face = gray_frame[y:y + h, x:x + w]
        pil_image = Image.fromarray(face)  # Convert to PIL Image
        input_tensor = transform(pil_image).unsqueeze(0)  # Add batch dimension

        # Make the prediction
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output.data, 1)
            emotion = emotion_labels[predicted.item()]  # Get the emotion label

        # Draw the emotion label above the bounding box
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

# Release the capture and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
