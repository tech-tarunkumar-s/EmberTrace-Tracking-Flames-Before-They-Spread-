import cv2
import numpy as np

# Fire detection function
def detect_fire(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_fire = np.array([0, 50, 250])
    upper_fire = np.array([35, 255, 255])
    fire_mask = cv2.inRange(hsv, lower_fire, upper_fire)
    
    contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fire_detected = False
    for contour in contours:
        if cv2.contourArea(contour) > 1000:
            fire_detected = True
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Draw bounding box

    return frame, fire_detected

# Smoke detection function
def detect_smoke(frame):
    # Simple method: convert to grayscale and threshold to detect smoke-like features
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, smoke_mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    
    smoke_detected = np.any(smoke_mask)  # Check if smoke is detected (any white pixels)
    return frame, smoke_detected
