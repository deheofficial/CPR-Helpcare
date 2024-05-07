import cv2
import numpy as np
import time

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize background subtractor
background_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, detectShadows=False)

# Initialize movement counter and CPR rate counter
movement_count = 0
cpr_rate_counter = 0

# Initialize variables for smooth transition
target_color = [0, 255, 255]  # Initial color (Yellow)
current_color = target_color.copy()
transition_speed = 4  # Speed of color transition

# Initialize time for delaying rate updates and message persistence
last_rate_update_time = time.time()
rate_update_delay = 1  # Delay in seconds for updating rate
message_persistence_time = 5  # Time in seconds to persist message and color
last_message_change_time = time.time()

# Initialize overlay_color and message outside the loop
overlay_color = tuple(current_color)
message = ""

# Initialize variables for counting time and total CPR count
start_time = time.time()
total_cpr_count = 0

while True:
    # Calculate elapsed time
    elapsed_time = int(time.time() - start_time)
    seconds_remaining = 60 - elapsed_time

    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply background subtraction
    mask = background_subtractor.apply(gray)

    # Apply thresholding to obtain binary image
    _, thresh = cv2.threshold(mask, 240, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Count movements
    movement_count = len(contours)

    # Perform rate calculation based on movement count
    rate = movement_count  # Simplified for demonstration

    # Check if it's time to update the rate
    if time.time() - last_rate_update_time > rate_update_delay:
        # Define messages based on rate value
        if rate < 600:
            message = "Compress Harder"
            target_color = [0, 255, 255]  # Yellow
        elif 600 <= rate <= 1900:
            message = "Good Compress"
            target_color = [0, 255, 0]  # Green
        else:
            message = "Too Hard"
            target_color = [0, 0, 255]  # Red

        # Update the time of last rate update
        last_rate_update_time = time.time()

        # Update the time of last message change
        last_message_change_time = time.time()

        # Check if the CPR rate exceeds 500 and increment counter
        if rate >= 500:
            # Increment CPR counter by 1
            cpr_rate_counter += 1

            # Increment total CPR count by the actual rate value divided by 500
            total_cpr_count += rate / 500

    # Smooth transition for overlay color change
    for i in range(3):
        if current_color[i] < target_color[i]:
            current_color[i] = min(current_color[i] + transition_speed, target_color[i])
        elif current_color[i] > target_color[i]:
            current_color[i] = max(current_color[i] - transition_speed, target_color[i])

    overlay_color = tuple(current_color)

    # Check if the message and color should persist
    if time.time() - last_message_change_time < message_persistence_time:
        persist_message = message
        persist_color = overlay_color
    else:
        persist_message = ""
        persist_color = (0, 0, 0)  # Black color for no message

    # Create overlay with the current color
    overlay = np.zeros_like(frame)
    overlay[:] = persist_color

    # Add weighted overlay to frame
    opacity = 0.5
    cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)

    # Display the resulting frame with message, counter, and time remaining
    cv2.putText(frame, "Rate: {}".format(rate), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, persist_message, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, "Time Remaining: {} seconds".format(seconds_remaining), (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('CPR Rate Monitor', frame)

    # Check if time is up
    if elapsed_time >= 60:
        # Determine CPR efficiency based on total CPR count
        if total_cpr_count < 81:
            efficiency = "Inefficient CPR"
        elif 81 <= total_cpr_count <= 90:
            efficiency = "Best CPR"
        else:
            efficiency = "Vigorous CPR"

        print("Total CPR Count:", total_cpr_count)
        print("CPR Efficiency:", efficiency)
        break

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
