import streamlit as st
import cv2
import numpy as np
from fire_detect import detect_fire, detect_smoke  # Assuming you have smoke detection implemented

# Set page configuration and title
st.set_page_config(page_title="🔥 Fire & Smoke Detection", layout="wide")
st.title("🔥 AI Fire and Smoke Detection System")

# Add a custom style for the UI
st.markdown("""
    <style>
        .title {
            color: #FF4500;
            font-size: 40px;
            font-family: 'Arial', sans-serif;
            text-align: center;
            margin-bottom: 20px;
        }
        .button {
            background-color: #FF4500;
            color: white;
            font-size: 20px;
            padding: 12px 40px;
            border-radius: 5px;
            border: none;
            cursor: pointer;
        }
        .button:hover {
            background-color: #FF6347;
        }
        .status {
            font-size: 25px;
            font-weight: bold;
            text-align: center;
            margin-top: 20px;
        }
        .fire {
            color: red;
        }
        .no-fire {
            color: green;
        }
        .smoke {
            color: gray;
        }
        .detection-info {
            font-size: 18px;
            text-align: center;
            color: #666;
        }
    </style>
""", unsafe_allow_html=True)

# Button for starting detection
start_cam = st.button("Start Detection", key="start")
stop_cam = st.button("Stop Detection", key="stop")

# Containers for webcam feed and status updates
frame_placeholder = st.empty()
mask_placeholder = st.empty()
status_placeholder = st.empty()

if start_cam:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Failed to open webcam.")
    else:
        st.success("Webcam started. Click 'Stop' to end.")
        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("Can't read frame. Exiting.")
                break

            # Fire and smoke detection
            fire_output, fire_detected = detect_fire(frame)
            smoke_output, smoke_detected = detect_smoke(frame)

            # Fire detection status
            fire_status = "🔥 FIRE DETECTED!" if fire_detected else "✅ No Fire"
            fire_status_color = "red" if fire_detected else "green"

            # Smoke detection status
            smoke_status = "🌫️ SMOKE DETECTED!" if smoke_detected else "✅ No Smoke"
            smoke_status_color = "gray" if smoke_detected else "green"

            status_placeholder.markdown(f"### <span style='color:{fire_status_color}'>{fire_status}</span>", unsafe_allow_html=True)
            status_placeholder.markdown(f"### <span style='color:{smoke_status_color}'>{smoke_status}</span>", unsafe_allow_html=True)

            # Draw bounding boxes around detected fire (if any)
            if fire_detected:
                # Assuming you have fire bounding box coordinates
                fire_box = fire_output  # Draw bounding box in detect_fire or modify it here

            # Display webcam feed
            frame_rgb = cv2.cvtColor(fire_output, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            # Optional: Fire Mask (for debugging)
            hsv = cv2.cvtColor(fire_output, cv2.COLOR_BGR2HSV)
            lower_fire = np.array([0, 50, 250])
            upper_fire = np.array([35, 255, 255])
            fire_mask = cv2.inRange(hsv, lower_fire, upper_fire)
            mask_placeholder.image(fire_mask, caption="🔥 Fire Mask Debug", use_container_width=True)

            # Check if stop button is clicked
            if stop_cam:
                st.warning("Stopping webcam feed.")
                cap.release()
                break

    cap.release()
    st.success("Webcam feed stopped.")
