from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

# Function to detect movement
def detect_movement(frame1, frame2, slow_threshold, fast_threshold):
    # Convert frames to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Compute absolute difference between frames
    diff = cv2.absdiff(gray1, gray2)

    # Apply a threshold to get a binary image
    _, threshold = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Count movement repetitions
    movement_count = len(contours)

    # Classify movement speed based on thresholds
    if movement_count <= slow_threshold:
        speed = "Too Slow"
    elif movement_count <= fast_threshold:
        speed = "Medium (Good)"
    else:
        speed = "Too Fast"

    # Draw movement count and speed meter
    cv2.putText(frame1, f"Movement Count: {movement_count}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame1, f"Movement Speed: {speed}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Draw contours
    cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)

    return frame1

def generate_frames():
    # Initialize webcam with DirectShow backend
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Read two frames
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    # Set thresholds for movement speed classification
    slow_threshold = 500  # Adjust according to your preference
    fast_threshold = 1500  # Adjust according to your preference

    while True:
        # Detect movement and display meter
        processed_frame = detect_movement(frame1, frame2, slow_threshold, fast_threshold)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # Update frames
        frame1 = frame2
        ret, frame2 = cap.read()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/movement_count')
def movement_count():
    # Perform movement detection and return the count as a string
    slow_threshold = 500  # Adjust according to your preference
    fast_threshold = 1500  # Adjust according to your preference
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    _, frame1 = cap.read()
    _, frame2 = cap.read()
    count = len(cv2.findContours(cv2.absdiff(cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY),
                                               cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)),
                                  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])
    speed = "Too Slow" if count <= slow_threshold else "Medium (Good)" if count <= fast_threshold else "Too Fast"
    return str(count) + " (" + speed + ")"

if __name__ == "__main__":
    app.run(debug=True)
