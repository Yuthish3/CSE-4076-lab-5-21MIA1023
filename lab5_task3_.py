import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load images from directories or from a single input image
def load_images_from_folder_or_image(folder=None, image_path=None):
    images = []
    if folder:
        for filename in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
    elif image_path:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images

# Face detection using Haar Cascades
def detect_face(image, face_cascade):
    gray = image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None
    (x, y, w, h) = faces[0]
    face = gray[y:y+h, x:x+w]
    return face, (w, h)  # Return the detected face and its width and height

# Feature extraction: Geometric features
def extract_geometric_features(face, jaw_width, face_height):
    h, w = face.shape
    center_x, center_y = w // 2, h // 2
    
    # Estimate positions of facial landmarks (eyes, nose, mouth)
    eye_y = h // 4
    mouth_y = 3 * h // 4
    eye_distance = w // 4  # Roughly the distance between the eyes

    # Distances between features
    eye_to_nose = center_y - eye_y
    nose_to_mouth = mouth_y - center_y
    
    # Adding additional geometric features
    return {
        'jaw_width': jaw_width,
        'face_height': face_height,
        'eye_distance': eye_distance,
        'eye_to_nose': eye_to_nose,
        'nose_to_mouth': nose_to_mouth
    }

# Feature extraction: Texture-based features (e.g., LBP)
def extract_texture_features(face):
    lbp = np.zeros(face.shape, dtype=np.uint8)
    for y in range(1, face.shape[0] - 1):
        for x in range(1, face.shape[1] - 1):
            center = face[y, x]
            binary_string = ''
            for dy, dx in [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]:
                binary_string += '1' if face[y + dy, x + dx] > center else '0'
            lbp[y, x] = int(binary_string, 2)

    hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
    return hist

# Rule-based classification based on extracted features
def classify_gender(geometric_features):
    jaw_width = geometric_features['jaw_width']
    face_height = geometric_features['face_height']
    
    # Using a combination of geometric features for classification
    if jaw_width < 1200 and face_height < 200: 
        return 'Male'
    else:
        return 'Female'

# Main pipeline
def process_and_classify_images(female_dir=None, male_dir=None, single_image_path=None):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    images = []
    if female_dir:
        images += load_images_from_folder_or_image(folder=female_dir)
    if male_dir:
        images += load_images_from_folder_or_image(folder=male_dir)
    if single_image_path:
        images += load_images_from_folder_or_image(image_path=single_image_path)

    for i, image in enumerate(images):
        result = detect_face(image, face_cascade)
        if result is None:
            print(f"Image {i+1}: No face detected.")
            continue
        face, (jaw_width, face_height) = result
        
        face_resized = cv2.resize(face, (200, 200))
        
        geom_features = extract_geometric_features(face_resized, jaw_width, face_height)
        text_features = extract_texture_features(face_resized)
        
        gender = classify_gender(geom_features)
        
        print(f"Image {i+1}: Predicted Gender = {gender}, Jaw Width = {jaw_width}")
        plt.imshow(face_resized, cmap='gray')
        plt.title(f"Extracted Face - Gender: {gender} | Jaw Width: {jaw_width} ")
        plt.show()

# Example usage:

female_dir = '/Users/yuthishkumar/Downloads/archive (28)/train/women'
male_dir = '/Users/yuthishkumar/Downloads/archive (28)/test/men'

single_image_path = '/Users/yuthishkumar/Downloads/male_test.webp'  # Replace with your image path
process_and_classify_images(female_dir=female_dir,male_dir=male_dir,single_image_path=single_image_path)

