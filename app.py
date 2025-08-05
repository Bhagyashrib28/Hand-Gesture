from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import pyttsx3
import threading
import math
import queue
import time

# Initialize Flask App
app = Flask(__name__)

# Initialize Mediapipe Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
video_running = False

# Create a queue to store speech commands
speech_queue = queue.Queue(maxsize=5)

# Cooldown mechanism
last_speech_time = 0
speech_cooldown = 2  # 2 seconds between each speech output

def speak(text):
    """ Reinitialize pyttsx3 every time to ensure speech works consistently. """
    global last_speech_time
    current_time = time.time()
    
    if current_time - last_speech_time > speech_cooldown:  # Ensure cooldown between commands
        if not speech_queue.full():  # Avoid queue overflow
            speech_queue.put(text)  # Add text to queue for processing
            last_speech_time = current_time

# Function to process speech commands

def process_speech():
    while True:
        text = speech_queue.get()
        if text is None:
            break  # Stop if None is received
        try:
            engine = pyttsx3.init()  # Reinitialize each time
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except RuntimeError:
            pass  # Handle errors gracefully
        speech_queue.task_done()

# Start the speech processing thread
speech_thread = threading.Thread(target=process_speech, daemon=True)
speech_thread.start()

def calculate_distance(landmark1, landmark2):
    return math.sqrt((landmark1.x - landmark2.x) ** 2 + (landmark1.y - landmark2.y) ** 2)

def check_gesture(landmarks):
    thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip, palm_base = \
        landmarks[4], landmarks[8], landmarks[12], landmarks[16], landmarks[20], landmarks[0]

    dist_thumb, dist_index, dist_middle, dist_ring, dist_pinky = \
        calculate_distance(palm_base, thumb_tip), calculate_distance(palm_base, index_tip), \
        calculate_distance(palm_base, middle_tip), calculate_distance(palm_base, ring_tip), \
        calculate_distance(palm_base, pinky_tip)

    if dist_thumb > 0.2 and dist_index > 0.2 and dist_middle > 0.2 and dist_ring > 0.2 and dist_pinky > 0.2:
        return "Hello!"
    if dist_thumb < 0.2 and dist_index < 0.2 and dist_middle < 0.2 and dist_ring < 0.2 and dist_pinky < 0.2:
        return "Punch!"
    if dist_thumb > 0.2 and dist_index < 0.2 and dist_middle < 0.2 and dist_ring < 0.2 and dist_pinky < 0.2:
        return "Good Job!"
    if dist_index > 0.2 and dist_middle > 0.2 and dist_ring < 0.2 and dist_pinky < 0.2 and dist_thumb < 0.2:
        return "Peace!"
    if dist_thumb < 0.2 and dist_index < 0.2 and dist_middle > 0.2 and dist_ring > 0.2 and dist_pinky > 0.2:
        return "Okay!"
    if dist_index > 0.2 and dist_pinky > 0.2 and dist_middle < 0.2 and dist_ring < 0.2 and dist_thumb < 0.2:
        return "Rock On!"
    if dist_thumb > 0.2 and dist_pinky > 0.2 and dist_index < 0.2 and dist_middle < 0.2 and dist_ring < 0.2:
        return "Call Me!"
    if dist_index > 0.2 and dist_thumb > 0.2 and dist_middle < 0.2 and dist_ring < 0.2 and dist_pinky < 0.2:
        return "Smile!"
    if dist_index > 0.2 and dist_middle > 0.2 and dist_pinky > 0.2 and dist_ring < 0.2 and dist_thumb < 0.2:
        return "I love you"
    return ""

def generate_frames():
    global video_running
    previous_gesture = ""
    while video_running:
        success, frame = cap.read()
        if not success:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        gesture = ""
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = hand_landmarks.landmark
                gesture = check_gesture(landmarks)

        if gesture and gesture != previous_gesture:
            previous_gesture = gesture
            speak(gesture)

        cv2.putText(frame, gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
# Routes for Web Pages
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/gestures')
def gestures():
    return render_template('gestures.html')

@app.route('/features')
def features():
    return render_template('features.html')

@app.route('/demo')
def demo():
    return render_template('demo.html')

@app.route('/live-testing')
def live_testing():
    return render_template('live-testing.html')


@app.route('/video_feed')
def video_feed():
    global video_running, cap
    if cap is None or not cap.isOpened():  # Reinitialize camera if needed
        cap = cv2.VideoCapture(0)

    video_running = True
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_video')
def stop_video():
    global video_running, cap
    video_running = False  # Stop video feed

    time.sleep(1)  # Wait for the video loop to stop

    # Release the camera safely
    if cap is not None:
        cap.release()
        cap = None  # Reset camera

    return jsonify({"message": "Video feed stopped and camera released"})

if __name__ == "__main__":
    app.run(debug=True)
    