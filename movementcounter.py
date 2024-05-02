import cv2
import time

# Function to detect repetitive movement
def detect_repetitive_movement(frame1, frame2, slow_threshold, fast_threshold, min_repetitions, delay_time):
    # Convert frames to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Compute absolute difference between frames
    diff = cv2.absdiff(gray1, gray2)

    # Apply a threshold to get a binary image
    _, threshold = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables
    repetitive_movement = False
    movement_count = len(contours)

    # Classify movement speed based on thresholds
    if movement_count <= slow_threshold:
        speed = "Too Slow"
    elif movement_count <= fast_threshold:
        speed = "Medium (Good)"
    else:
        speed = "Too Fast"

    # Check if movement is repetitive
    if movement_count >= min_repetitions:
        repetitive_movement = True

    # Draw movement count and speed meter
    cv2.putText(frame1, f"Movement Count: {movement_count}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame1, f"Movement Speed: {speed}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Draw repetitive movement indicator
    if repetitive_movement:
        cv2.putText(frame1, "Repetitive Movement Detected", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw contours
    cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)

    return frame1, repetitive_movement

# Initialize webcam with DirectShow backend
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Read two frames
ret, frame1 = cap.read()
ret, frame2 = cap.read()

# Set thresholds for movement speed classification
slow_threshold = 500  # Adjust according to your preference
fast_threshold = 1000  # Adjust according to your preference

# Set parameters for repetitive movement detection
min_repetitions = 3  # Minimum number of repetitions to consider movement repetitive
delay_time = 10  # Delay time in seconds before changing from one movement type to another

# Initialize variables for delay
current_movement_type = ""
last_movement_change_time = time.time()

while True:
    # Detect repetitive movement and display meter
    processed_frame, repetitive_movement = detect_repetitive_movement(frame1, frame2, slow_threshold, fast_threshold, min_repetitions, delay_time)

    # Display the resulting frame
    cv2.imshow('Movement Detection', processed_frame)

    # Check if movement type has changed and if the delay time has passed
    if repetitive_movement and current_movement_type != "Repetitive":
        current_movement_type = "Repetitive"
        last_movement_change_time = time.time()
    elif not repetitive_movement and current_movement_type != "Non-Repetitive":
        current_movement_type = "Non-Repetitive"
        last_movement_change_time = time.time()

    # Update frames
    frame1 = frame2
    ret, frame2 = cap.read()

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()

cv2.destroyAllWindows()
