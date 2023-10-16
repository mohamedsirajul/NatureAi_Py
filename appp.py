from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from tkinter import *
import tkintermapview
from PIL import ImageGrab

app = Flask(__name__)
CORS(app)

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

# Your existing GUI setup code

if __name__ == '__main__':
    app.run(debug=True)
