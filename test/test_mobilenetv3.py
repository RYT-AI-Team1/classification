import os
import torch
import torchvision
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms 
from torchvision.transforms import Resize

# Step 1: Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model')
MODEL1_PATH = os.path.join(MODEL_PATH, 'mobilenetv3')
BEST_MODEL_FILE = os.path.join(MODEL1_PATH, 'best.pt')
checkpoint = torch.load(BEST_MODEL_FILE)

# Model
model = torchvision.models.mobilenet_v3_small(pretrained=True)

# Replace the last layer with a new fully connected layer
num_classes = 6  # Replace this with the actual number of output classes
in_features = model.classifier[3].in_features
model.classifier[3] = torch.nn.Linear(in_features, num_classes)

model.load_state_dict(checkpoint['state_dict'])

# Step 2: Load a random test image from the datafolder folder

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
DATAFOLDER_PATH = os.path.join(DATA_PATH, 'datasets/datafolder')

# Get a list of all image files in the datafolder folder
image_files = [os.path.join(DATAFOLDER_PATH, f) for f in os.listdir(DATAFOLDER_PATH) if f.endswith('.jpg')]

# Choose a random image file from the list
rand_idx = np.random.randint(len(image_files))
image_path = image_files[rand_idx]
image_name = os.path.splitext(os.path.basename(image_path))[0]  # Extract the filename without extension

image = Image.open(image_path)

# Step 3: Apply the same transformations to the test image as used during training

transform = transforms.Compose([
    Resize((256, 256)),
    transforms.ToTensor()
])

# Apply the transformation to the image
image_tensor = transform(image).unsqueeze(0)  # Unsqueeze adds a batch dimension

# Send the tensor to the device (CPU or GPU)
device = torch.device("cpu")  # Replace this with your actual device name
image_tensor = image_tensor.to(device)

# Step 4: Get the predictions for the image using the trained model

with torch.no_grad():
    scores = model(image_tensor)
    _, predictions = scores.max(1)
    
# Print the predicted class
predicted_class = predictions.item()
score = scores[0][predicted_class].item()
print(f"Predicted class: {predicted_class}")
print(f"Prediction score: {score}")

if score < 5:
    print("False")

# Step 5: Draw a bounding box around the object in the image

# Convert the PIL image to OpenCV format
image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

# Get the screen resolution
screen_width, screen_height = 1920, 1080  # Replace these values with your actual screen resolution

# Calculate the new dimensions of the image
new_width = int(screen_width * 0.6)
new_height = int(screen_height * 0.6)

# Resize the image
image_cv2 = cv2.resize(image_cv2, (new_width, new_height))

# Define the colors for the bounding box and the label text
color = (0, 255, 0)  # Green
label_color = (125, 125, 125)  # Gray

# Load the classes file
with open('test/classes.txt') as f:
    classes = f.read().splitlines()

# Get the label text for the predicted class
label_text = classes[predicted_class]

# Add the prediction score to the label text
label_text += f" ({score:.2f})"

# Get the image dimensions
height, width, _ = image_cv2.shape

# Draw a rectangle around the object
cv2.rectangle(image_cv2, (0, 0), (width, height), color, 3)

# Write the label text on the image
label_size, baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
label_x = int((width - label_size[0]) / 2)
label_y = int(height - 0.1*height)
cv2.rectangle(image_cv2, (label_x - 10, label_y - label_size[1] - 10),
              (label_x + label_size[0] + 10, label_y + baseline + 10),
              label_color, cv2.FILLED)
cv2.putText(image_cv2, label_text, (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

# Save the resulting image with a filename that includes the original image name and the predicted class label
SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'saved')
OUTPUT_PATH = os.path.join(SAVE_PATH, 'outputs/mobilenetv3')
os.makedirs(OUTPUT_PATH, exist_ok=True)
output_file = os.path.join(OUTPUT_PATH, f"{image_name}_result{predicted_class}.jpg")
cv2.imwrite(output_file, image_cv2)

# Show the image with the bounding box and label
cv2.imshow('Image', image_cv2)
cv2.waitKey(0)
cv2.destroyAllWindows()
