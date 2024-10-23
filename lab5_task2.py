import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
def load_image(image_path):
    image = cv2.imread(image_path)
    return image

# Skin color thresholding for face and hand detection
def detect_skin(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv_img, lower_skin, upper_skin)
    return mask

# Detect faces using Haar Cascades
def detect_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return faces

# Extract facial features and return emotions
def extract_facial_features(img, face_coords):
    emotions = []
    for (x, y, w, h) in face_coords:
        roi = img[y:y+h, x:x+w]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
        # Adjust scaleFactor and minNeighbors for better smile detection
        mouth = mouth_cascade.detectMultiScale(gray_roi, scaleFactor=1.1, minNeighbors=15, minSize=(25, 25))
        
        if len(mouth) > 0:
            # Smile detected (based on upward curvature)
            emotion = "Happy"
        else:
            # Further logic to detect sadness or neutral
            mouth_straight = detect_mouth_straightness(roi)
            if mouth_straight:
                emotion = "Neutral"
            else:
                emotion = "Sad"
        
        # Store the detected emotion
        emotions.append(emotion)
        
        # Draw rectangle and label emotion on the image
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(img, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    
    return emotions

# Function to detect if mouth is relatively straight (neutral)
def detect_mouth_straightness(roi):
    height, width = roi.shape[:2]
    mouth_area = roi[int(height * 0.7):int(height * 0.85), :]  # Extract lower part where mouth usually is
    edge_detection = cv2.Canny(mouth_area, 50, 150)
    contours, _ = cv2.findContours(edge_detection, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # If contours are minimal, the mouth is likely straight (neutral)
    return len(contours) < 5

# Overall sentiment classification based on majority
def classify_overall_sentiment(face_coords, img):
    # Get the emotions for all detected faces
    emotions = extract_facial_features(img, face_coords)
    
    happy_count = emotions.count("Happy")
    sad_count = emotions.count("Sad")
    neutral_count = emotions.count("Neutral")
    
    if happy_count > sad_count and happy_count > neutral_count:
        return "Overall Sentiment: Happy"
    elif sad_count > happy_count and sad_count > neutral_count:
        return "Overall Sentiment: Sad"
    elif neutral_count > happy_count and neutral_count > sad_count:
        return "Overall Sentiment: Neutral"
    else:
        return "Overall Sentiment: Mixed"

# Plot image with detected features and sentiments
def plot_image(img, title):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Main Function
if __name__ == "__main__":
    image_path = "/Users/yuthishkumar/Downloads/sad_test.jpg"  # Add your image path here
    img = load_image(image_path)
    
    # Detect skin regions
    skin_mask = detect_skin(img)
    
    # Detect faces
    faces = detect_faces(img)
    
    # Classify the overall sentiment
    overall_sentiment = classify_overall_sentiment(faces, img)
    
    # Display the image with facial feature analysis and sentiment
    plot_image(img, overall_sentiment)
