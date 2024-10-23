import cv2
import numpy as np

# Load the video
video_path = '/Users/yuthishkumar/Downloads/drinkad_iva_lab4.mp4'  # Change this to your actual video path
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
frame_count = 0
event_frames = []
previous_frame = None  # Initialize previous_frame here

# Define function to calculate manual histogram
def calculate_manual_histogram(image, num_bins=256):
    hist = np.zeros(num_bins)
    for pixel_value in image.flatten():
        hist[pixel_value] += 1
    return hist

# Define function to normalize histogram
def normalize_histogram(hist, num_pixels):
    return hist / num_pixels

# Define function to compare histograms (intersection)
def compare_histograms(hist1, hist2):
    return np.sum(np.minimum(hist1, hist2))

# Define threshold for motion detection
motion_threshold = 0.5  # Adjust based on sensitivity

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate the histogram manually
    frame_hist = calculate_manual_histogram(frame_gray)

    if previous_frame is not None:
        # Calculate the histogram for the previous frame
        previous_hist = calculate_manual_histogram(previous_frame)

        # Normalize the histograms
        num_pixels = frame_gray.size
        frame_hist = normalize_histogram(frame_hist, num_pixels)
        previous_hist = normalize_histogram(previous_hist, num_pixels)

        # Compare histograms using intersection
        hist_diff = compare_histograms(previous_hist, frame_hist)

        # Detect motion if the difference is below the threshold
        if hist_diff < motion_threshold:
            event_frames.append((frame_count, hist_diff))
            timestamp = frame_count / fps  # Calculate timestamp in seconds
            # Highlight the moving region
            cv2.putText(frame, 'Motion Detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # Show the frame with the motion event
            cv2.imshow('Motion Detection', frame)

            # Print frame information with timestamp
            print(f"Motion detected at Frame {frame_count}, Time: {timestamp:.2f}s, Histogram Difference: {hist_diff:.4f}")

        # Wait a little before displaying the next frame
        if cv2.waitKey(2) & 0xFF == ord('q'):
            break

    previous_frame = frame_gray
    frame_count += 1

cap.release()
cv2.destroyAllWindows()

# Print final event frame information
print("Motion detected in the following frames:")
for event_frame in event_frames:
    timestamp = event_frame[0] / fps  # Calculate timestamp in seconds
    print(f"Frame {event_frame[0]}, Time: {timestamp:.2f}s, Histogram Difference: {event_frame[1]:.4f}")
