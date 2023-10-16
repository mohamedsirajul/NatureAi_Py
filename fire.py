from ultralytics import YOLO
import cv2
import math
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse

# Twilio credentials (replace with your own)
TWILIO_ACCOUNT_SID = 'AC811e94c5589ae7b41288c411eac25787'
TWILIO_AUTH_TOKEN = 'b53d2aac24ba8127a504bbeb15f34944'
TWILIO_PHONE_NUMBER = '+12406963888'
YOUR_PHONE_NUMBER = '+918056457791'

# Initialize the Twilio client
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Initialize the YOLO model
# model = YOLO('fire.pt')
model = YOLO('fire.pt')


# Running real-time from webcam
cap = cv2.VideoCapture('fire2.mp4')

# Reading the classes
classnames = ['fire']

# Initialize fire frame count
fire_frame_count = 0

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

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))
    result = model(frame, stream=True)

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
            body=f'Fire percentage exceeds 10%: Alart',
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

cap.release()
cv2.destroyAllWindows()
