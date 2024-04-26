import os
import cv2
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from flask import Flask, request, jsonify

# Step 1: Load annotated data
def load_annotations(json_file):
    with open(json_file, 'r') as f:
        annotations = json.load(f)
    return annotations

# Step 2: Extract features from images (using Histogram of Oriented Gradients - HOG)
def extract_features(image):
    if image is None:
        print("Error: Invalid image - image is None")
        return None
    
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize image to a fixed size (e.g., 64x128)
    resized_image = cv2.resize(gray, (64, 128))

    # Calculate HOG features
    hog = cv2.HOGDescriptor()
    features = hog.compute(resized_image).flatten()
    return features

# Step 3: Train KNN classifier
def train_classifier(data, labels):
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(data, labels)
    return clf

# Step 4: Perform object detection and memory classification
def detect_memory(image, clf):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Thresholding
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    memory_regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 10 and h > 10:  # Filter out small regions
            roi = image[y:y+h, x:x+w]
            features = extract_features(roi)
            prediction = clf.predict([features])
            if prediction == 1:  # Memory class
                memory_regions.append((x, y, x+w, y+h))

    return memory_regions

# Step 5: Assign captions to detected regions based on image labels
def assign_captions(image_path, memory_regions):
    if 'memory' in image_path:
        return ["Memory" for _ in range(len(memory_regions))]
    elif 'no_memory' in image_path:
        return ["No Memory" for _ in range(len(memory_regions))]
    else:
        return ["Unknown" for _ in range(len(memory_regions))]

app = Flask(__name__)

# Load annotated data
annotated_data_dir = "C:/Users/Michael.A_Sydani/Desktop/task3/annotated_data"
memory_annotations_file = "C:/Users/Michael.A_Sydani/Desktop/task3/annotated_data/labels_memory_2024-04-25-08-07-34.json"
no_memory_annotations_file = "C:/Users/Michael.A_Sydani/Desktop/task3/annotated_data/labels_no-memory_2024-04-25-08-28-24.json"

memory_annotations = load_annotations(memory_annotations_file)
no_memory_annotations = load_annotations(no_memory_annotations_file)

# Full paths to memory and no memory image directories
memory_images_dir = "C:/Users/Michael.A_Sydani/Desktop/task3/memory"
no_memory_images_dir = "C:/Users/Michael.A_Sydani/Desktop/task3/no_memory"

memory_data = [(cv2.imread(os.path.join(memory_images_dir, filename)), 1) for filename in os.listdir(memory_images_dir)]
no_memory_data = [(cv2.imread(os.path.join(no_memory_images_dir, filename)), 0) for filename in os.listdir(no_memory_images_dir)]

data = memory_data + no_memory_data
labels = [item[1] for item in data]
features = [extract_features(item[0]) for item in data]

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, random_state=42)
clf = train_classifier(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Classifier accuracy:", accuracy)

@app.route('/detect_memory', methods=['POST'])
def detect_memory_api():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})

    image = cv2.imdecode(np.frombuffer(request.files['image'].read(), np.uint8), cv2.IMREAD_COLOR)
    memory_regions = detect_memory(image, clf)

    # Get the filename from the request
    image_filename = request.files['image'].filename

    # Assign captions based on image path
    captions = assign_captions(image_filename, memory_regions)

    for region, caption in zip(memory_regions, captions):
        x, y, x2, y2 = region
        cv2.rectangle(image, (x, y), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, caption, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    _, encoded_image = cv2.imencode('.jpg', image)

    return encoded_image.tobytes(), 200, {'Content-Type': 'image/jpeg'}

if __name__ == '__main__':
    app.run(debug=True)
