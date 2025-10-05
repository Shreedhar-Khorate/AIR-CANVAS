from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import os
from datetime import datetime

app = Flask(__name__)

# Initialize color points arrays
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]
wpoints = [deque(maxlen=1024)]  # White color for eraser

# Initialize indexes for each color
blue_index = green_index = red_index = yellow_index = white_index = 0

# Colors in BGR format (added white for eraser)
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 255, 255)]
color_names = ["Blue", "Green", "Red", "Yellow", "Eraser"]
colorIndex = 0
brush_size = 5
eraser_size = 20
canvas_color = (255, 255, 255)  # White background

# Create a canvas
paintWindow = np.ones((480, 640, 3), dtype=np.uint8) * 255

# Initialize Mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Error: Webcam could not be opened.")

def update_paint_window():
   
    global paintWindow
    
    # Apply canvas color
    paintWindow[:] = canvas_color

    # --- Draw Kolam-style dots (white background = black dots) ---
    rows, cols = 7, 7   # grid size, you can change (e.g., 5x5, 9x9)
    spacing = 60        # distance between dots
    start_x, start_y = 80, 80  # starting position (margin)
    dot_radius = 5
    
    for i in range(rows):
        for j in range(cols):
            cx = start_x + j * spacing
            cy = start_y + i * spacing
            cv2.circle(paintWindow, (cx, cy), dot_radius, (0, 0, 0), -1)  # black filled dot

    # Draw all the points with their respective sizes (original code)
    points = [bpoints, gpoints, rpoints, ypoints, wpoints]
    for color_points, color in zip(points, colors):
        for i in range(len(color_points)):
            for j in range(1, len(color_points[i])):
                if color_points[i][j-1] is None or color_points[i][j] is None:
                    continue
                thickness = eraser_size if color == (255, 255, 255) else brush_size
                # For eraser, draw with the canvas color to "erase"
                if color == (255, 255, 255):
                    cv2.line(paintWindow, color_points[i][j-1], color_points[i][j], canvas_color, thickness)
                else:
                    cv2.line(paintWindow, color_points[i][j-1], color_points[i][j], color, thickness)

def generate_webcam_frames():
    global blue_index, green_index, red_index, yellow_index, white_index, colorIndex, brush_size, eraser_size

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Clear the canvas when 'c' is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            bpoints.clear()
            gpoints.clear()
            rpoints.clear()
            ypoints.clear()
            wpoints.clear()
            bpoints.append(deque(maxlen=1024))
            gpoints.append(deque(maxlen=1024))
            rpoints.append(deque(maxlen=1024))
            ypoints.append(deque(maxlen=1024))
            wpoints.append(deque(maxlen=1024))
            blue_index = green_index = red_index = yellow_index = white_index = 0

        # Process hand landmarks
        result = hands.process(framergb)
        
        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
                
                # Get landmark positions
                for lm in handslms.landmark:
                    lmx = int(lm.x * frame.shape[1])
                    lmy = int(lm.y * frame.shape[0])
                    landmarks.append([lmx, lmy])

                # Get index finger (landmark 8) and thumb (landmark 4)
                if len(landmarks) >= 9:  # Ensure we have enough landmarks
                    index_finger = (landmarks[8][0], landmarks[8][1])
                    thumb = (landmarks[4][0], landmarks[4][1])
                    
                    # Draw circle on index finger with current brush size
                    current_color = (255, 255, 255) if colorIndex == 4 else colors[colorIndex]
                    current_size = eraser_size if colorIndex == 4 else brush_size
                    cv2.circle(frame, index_finger, current_size//2, current_color, -1)
                    
                    # Check if thumb and index finger are close (pinch gesture)
                    distance = ((thumb[0] - index_finger[0])**2 + (thumb[1] - index_finger[1])**2)**0.5
                    
                    if distance < 30:  # Pinch detected
                        if colorIndex == 0:
                            bpoints.append(deque(maxlen=1024))
                            blue_index += 1
                        elif colorIndex == 1:
                            gpoints.append(deque(maxlen=1024))
                            green_index += 1
                        elif colorIndex == 2:
                            rpoints.append(deque(maxlen=1024))
                            red_index += 1
                        elif colorIndex == 3:
                            ypoints.append(deque(maxlen=1024))
                            yellow_index += 1
                        elif colorIndex == 4:
                            wpoints.append(deque(maxlen=1024))
                            white_index += 1
                    else:
                        # Append the current position to the current color's points
                        if colorIndex == 0:
                            bpoints[blue_index].appendleft(index_finger)
                        elif colorIndex == 1:
                            gpoints[green_index].appendleft(index_finger)
                        elif colorIndex == 2:
                            rpoints[red_index].appendleft(index_finger)
                        elif colorIndex == 3:
                            ypoints[yellow_index].appendleft(index_finger)
                        elif colorIndex == 4:
                            wpoints[white_index].appendleft(index_finger)
                else:
                    # Append None when hand is not detected to break the line
                    if colorIndex == 0:
                        bpoints[blue_index].appendleft(None)
                    elif colorIndex == 1:
                        gpoints[green_index].appendleft(None)
                    elif colorIndex == 2:
                        rpoints[red_index].appendleft(None)
                    elif colorIndex == 3:
                        ypoints[yellow_index].appendleft(None)
                    elif colorIndex == 4:
                        wpoints[white_index].appendleft(None)

        # Update the paint window
        update_paint_window()
        
        # Combine the webcam feed and the paint window
        combined = np.hstack([frame, paintWindow])
        
        ret, buffer = cv2.imencode('.jpg', combined)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/save_canvas', methods=['POST'])
def save_canvas():
    update_paint_window()
    # Save the canvas as an image
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sketch_{timestamp}.png"
    filepath = os.path.join('static', 'saved_sketches', filename)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save the image
    cv2.imwrite(filepath, paintWindow)
    
    # Return the URL of the saved image
    sketch_url = f"/static/saved_sketches/{filename}"
    return jsonify({"status": "success", "sketch_url": sketch_url})

@app.route('/set_color', methods=['POST'])
def set_color():
    global colorIndex
    data = request.json
    colorIndex = int(data.get('colorIndex', 0))
    return {"status": "success"}

@app.route('/set_canvas_color', methods=['POST'])
def set_canvas_color():
    global canvas_color
    data = request.json
    canvas_color = tuple(map(int, data.get('canvasColor', [255, 255, 255])))
    return {"status": "success"}

@app.route('/set_brush_size', methods=['POST'])
def set_brush_size():
    global brush_size, eraser_size
    data = request.json
    size = int(data.get('brushSize', 5))
    brush_size = size
    eraser_size = size * 3  # Eraser is larger than brush
    return {"status": "success"}

@app.route('/clear_canvas', methods=['POST'])
def clear_canvas():
    global bpoints, gpoints, rpoints, ypoints, wpoints, blue_index, green_index, red_index, yellow_index, white_index
    bpoints = [deque(maxlen=1024)]
    gpoints = [deque(maxlen=1024)]
    rpoints = [deque(maxlen=1024)]
    ypoints = [deque(maxlen=1024)]
    wpoints = [deque(maxlen=1024)]
    blue_index = green_index = red_index = yellow_index = white_index = 0
    return {"status": "success"}

@app.route('/video_feed')
def video_feed():
    return Response(generate_webcam_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)