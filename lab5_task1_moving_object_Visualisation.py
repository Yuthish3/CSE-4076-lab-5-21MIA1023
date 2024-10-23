import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Helper function to load grayscale frames from a directory
def load_frames_from_directory(directory):
    frames = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(directory, filename)
            image = Image.open(image_path).convert('L')  # Convert image to grayscale
            frames.append(np.array(image))
    return frames

# Grayscale Thresholding Segmentation
def grayscale_threshold(frame, lower_thresh, upper_thresh):
    mask = (frame >= lower_thresh) & (frame <= upper_thresh)
    return mask.astype(np.uint8)

# Sobel Edge Detection
def sobel_edge_detection(frame):
    Kx = np.array([[ -1,  0,  1], [ -2,  0,  2], [ -1,  0,  1]])
    Ky = np.array([[ -1, -2, -1], [  0,  0,  0], [  1,  2,  1]])
    
    height, width = frame.shape
    edges_x = np.zeros_like(frame)
    edges_y = np.zeros_like(frame)
    
    for i in range(1, height-1):
        for j in range(1, width-1):
            Gx = np.sum(np.multiply(Kx, frame[i-1:i+2, j-1:j+2]))
            edges_x[i, j] = abs(Gx)
            
            Gy = np.sum(np.multiply(Ky, frame[i-1:i+2, j-1:j+2]))
            edges_y[i, j] = abs(Gy)
    
    edges = np.sqrt(edges_x**2 + edges_y**2)
    return (edges / np.max(edges) * 255).astype(np.uint8)

# Calculate the centroid of segmented regions
def calculate_centroid(binary_mask):
    coords = np.argwhere(binary_mask == 1)
    if len(coords) == 0:
        return None
    centroid = np.mean(coords, axis=0)
    return centroid

# Background Subtraction to identify foreground regions
def background_subtraction(frame1, frame2, threshold=30):
    diff = np.abs(frame1.astype(np.int16) - frame2.astype(np.int16))
    foreground_mask = diff > threshold
    return foreground_mask.astype(np.uint8)

# Find static regions (background) across multiple frames
def find_static_regions(frame_sequence, threshold=10):
    static_regions = np.ones_like(frame_sequence[0])
    for i in range(1, len(frame_sequence)):
        diff = np.abs(frame_sequence[i].astype(np.int16) - frame_sequence[0].astype(np.int16))
        static_regions &= (diff < threshold)
    return static_regions.astype(np.uint8)

# Main Processing Function
def spatio_temporal_segmentation(directory):
    frames = load_frames_from_directory(directory)
    
    if len(frames) < 2:
        print("Not enough frames for analysis.")
        return
    
    lower_threshold = 100  # For grayscale images, just one threshold value
    upper_threshold = 200
    
    # Process each frame
    for i in range(165, len(frames)):
        frame_1 = frames[i-1]
        frame_2 = frames[i]

        # Apply grayscale thresholding
        segmented_frame_1 = grayscale_threshold(frame_1, lower_threshold, upper_threshold)
        segmented_frame_2 = grayscale_threshold(frame_2, lower_threshold, upper_threshold)

        # Track object motion
        centroid_frame_1 = calculate_centroid(segmented_frame_1)
        centroid_frame_2 = calculate_centroid(segmented_frame_2)
        
        if centroid_frame_1 is not None and centroid_frame_2 is not None:
            motion_vector = centroid_frame_2 - centroid_frame_1
            print(f"Frame {i}: Object moved by {motion_vector}")

        # Foreground Detection
        foreground_mask = background_subtraction(frame_1, frame_2)

        # Visualize the segmentation and motion
        plt.figure(figsize=(10, 5))
        
        # Original Frame
        plt.subplot(1, 3, 1)
        plt.imshow(frame_2)
        plt.title("Original Frame")



        # Foreground (Moving Regions)
        plt.subplot(1, 3, 3)
        plt.imshow(foreground_mask * 255, cmap='gray')
        plt.title("Foreground (Moving)")

        plt.show()
    
    # Analyze static regions (background)
    background_mask = find_static_regions(frames[:3])
    plt.figure()
    plt.imshow(background_mask * 255, cmap='gray')
    plt.title("Static Background")
    plt.show()

# Example usage
directory_path = "/Users/yuthishkumar/Desktop/python project/drinkad"  # Provide the path to the directory containing grayscale frames
spatio_temporal_segmentation(directory_path)
