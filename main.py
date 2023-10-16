from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import random
import json
import pickle
import numpy as np
# import wikipedia
import nltk
import cv2
import math
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse
from ultralytics import YOLO
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow import keras
from threading import Thread
import atexit  # Add this line
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from tkinter import *
import tkintermapview
from PIL import ImageGrab



nltk.download('punkt')
nltk.download('wordnet')

app = Flask(__name__)
CORS(app)

lemmatizer = WordNetLemmatizer()

# Load the chatbot model and intents data
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
chatbot_model = keras.models.load_model('chatbotmodel.h5')

# Initialize the Twilio client
TWILIO_ACCOUNT_SID = 'AC811e94c5589ae7b41288c411eac25787'
TWILIO_AUTH_TOKEN = 'b53d2aac24ba8127a504bbeb15f34944'
TWILIO_PHONE_NUMBER = '+12406963888'
YOUR_PHONE_NUMBER = '+918056457791'
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Initialize the YOLO model
fire_model = YOLO('fire.pt')

# Running real-time from webcam
cap = cv2.VideoCapture('fire2.mp4')

# Reading the classes
classnames = ['fire']

# Initialize fire frame count
fire_frame_count = 0

# Initialize the fire detection thread and cap
fire_thread = None

# Function to make an emergency call
def make_emergency_call():
    # Create a TwiML response that specifies the action to take during the call
    response = VoiceResponse()
    response.say("This is an emergency call. Please stay on the line for assistance.")

    # Make the call
    call = client.calls.create(
        to=YOUR_PHONE_NUMBER,
        from_=TWILIO_PHONE_NUMBER,
        twiml=response.to_xml()
    )

    print(f'Emergency call initiated.')

def clean_up():
    cap.release()
    cv2.destroyAllWindows()

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = chatbot_model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def fire_detection():
    global fire_frame_count
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (640, 480))
        result = fire_model(frame, stream=True)

        # Getting bbox, confidence, and class names information to work with
        fire_percentage = 0

        for info in result:
            boxes = info.boxes
            for box in boxes:
                confidence = box.conf[0]
                confidence = math.ceil(confidence * 100)
                Class = int(box.cls[0])
                if Class == 0:  # Check if the detected class is "fire"
                    if confidence > 50:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                        fire_percentage += 1

        cv2.imshow('frame', frame)

        # Increment fire frame count if fire is detected
        if fire_percentage > 0:
            fire_frame_count += 1

        # Check if the fire frame count exceeds 10 frames
        if fire_frame_count >= 10:
            # Send an SMS using Twilio
            message = client.messages.create(
                body=f'Fire percentage exceeds 10%: Alert',
                from_=TWILIO_PHONE_NUMBER,
                to=YOUR_PHONE_NUMBER
            )
            print('SMS sent!')

            # Make an emergency call
            make_emergency_call()

            # Reset the fire frame count
            fire_frame_count = 0

        # Exit the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            clean_up()
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Flask route to start fire detection
@app.route('/start_fire_detection', methods=['GET'])
def start_fire_detection():
    global fire_thread, cap

    if not fire_thread or not fire_thread.is_alive():
        # Initialize the webcam and start the fire detection thread
        cap = cv2.VideoCapture(1)
        fire_thread = Thread(target=fire_detection)
        fire_thread.start()
        return jsonify({'status': 'Fire detection started.'})
    else:
        return jsonify({'status': 'Fire detection is already running.'})

# Flask route to stop fire detection
@app.route('/stop_fire_detection', methods=['GET'])
def stop_fire_detection():
    global fire_thread, cap

    if fire_thread and fire_thread.is_alive():
        # Release the webcam and stop the fire detection thread
        cap.release()
        fire_thread.join()
        cv2.destroyAllWindows()  # Close all OpenCV windows
        return jsonify({'status': 'Fire detection stopped.'})
    else:
        return jsonify({'status': 'Fire detection is not running.'})

# Flask route for chat
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data['message']
    ints = predict_class(user_input)
    res = get_response(ints, intents)

    # Checking for fire in the user input
    if 'fire' in user_input.lower():
        # Make an emergency call
        make_emergency_call()
        res += " Emergency call initiated."

    return jsonify({'response': res})

# Flask route to get detection status
@app.route('/detection_status', methods=['GET'])
def get_detection_status():
    global fire_thread

    if fire_thread and fire_thread.is_alive():
        status = 'Fire detection is running.'
    else:
        status = 'Fire detection is not running.'

    return jsonify({'status': status})



@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')




# Define global variables for the map window and its status
map_window = None
map_window_open = False

def count_trees(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for the green color (adjust these values if needed)
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])

    # Threshold the image to extract the green regions
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Find contours in the binary image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Count the number of detected contours (trees)
    tree_count = len(contours)

    # Draw contours on the original image
    result_image = image.copy()
    cv2.drawContours(result_image, contours, -1, (0, 255, 0), 2)

    # Display the result
    cv2.imshow("Tree Detection", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return tree_count

def open_map_window():
    global map_window, map_window_open

    map_window = Tk()
    map_window.title("Map")
    map_window.geometry("1000x800")

    label = LabelFrame(map_window)
    label.pack(pady=20)

    map_widget = tkintermapview.TkinterMapView(label, width=800, height=600)
    map_widget.set_tile_server("https://mt0.google.com/vt/lyrs=s&hl=en&x={x}&y={y}&z={z}&s=Ga", max_zoom=22)
    map_widget.pack()

    capture_button = Button(map_window, text="Capture and Save Image", command=open_map_capture_image)
    capture_button.pack()

    map_widget.set_position(8.7244041, 77.7350118)

    map_window_open = True
    map_window.mainloop()

# Route to open the map, capture the image, and count trees
@app.route('/open_map_capture_image', methods=['POST'])
def open_map_capture_image():
    global map_window_open

    x = request.json.get('x')
    y = request.json.get('y')
    width = request.json.get('width')
    height = request.json.get('height')

    # Open the map window only if it's not already open
    if not map_window_open:
        open_map_window()

    # Capture the image
    screenshot = ImageGrab.grab(bbox=(x, y, x + width, y + height))
    screenshot.save("captured_image.jpg")

    # Count trees
    trees_detected = count_trees("captured_image.jpg")
    print(f"Number of trees detected: {trees_detected}")

    return jsonify({'message': 'Map opened, image captured, and trees counted successfully'})

 


# Run the Flask app
if __name__ == '__main__':
    atexit.register(clean_up)  # Register cleanup function
    app.run(host='0.0.0.0', port=5000)
