import cv2
import numpy as np
from tkinter import *
import tkintermapview
from PIL import ImageGrab

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

def capture_and_save_image():
    x = window.winfo_rootx()
    y = window.winfo_rooty()
    width = window.winfo_width()
    height = 630
    
    screenshot = ImageGrab.grab(bbox=(x, y, x + width, y + height))
    screenshot.save("captured_image.jpg")

    trees_detected = count_trees("captured_image.jpg")
    print(f"Number of trees detected: {trees_detected}")

    # Additional code if needed to perform actions based on the tree count

# GUI setup
window = Tk()
window.title("Map")
window.geometry("1000x800")

label = LabelFrame(window)
label.pack(pady=20)

map_widget = tkintermapview.TkinterMapView(label , width =800 , height = 600)
map_widget.set_tile_server("https://mt0.google.com/vt/lyrs=s&hl=en&x={x}&y={y}&z={z}&s=Ga", max_zoom=22)
map_widget.pack()

map_widget.set_position(8.7244041, 77.7350118)

capture_button = Button(window, text="Capture and Save Image", command=capture_and_save_image)
capture_button.pack()

window.mainloop()
 