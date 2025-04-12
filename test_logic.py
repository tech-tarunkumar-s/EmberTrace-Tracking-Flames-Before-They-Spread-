# test_logic.py
import cv2
from fire_detect import detect_fire

import numpy as np



cap = cv2.VideoCapture(0)



if not cap.isOpened():
    print("❌ Failed to open webcam.")
    exit()

print("✅ Webcam opened successfully.")

while True:
   # This part stays the same
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))  # optional
    output, fire_detected = detect_fire(frame)

    label = "🔥 FIRE DETECTED!" if fire_detected else "✅ No fire"
    cv2.putText(output, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Fire Detection", output)

    # Add to test_logic.py
    fire_mask = cv2.inRange(cv2.cvtColor(output, cv2.COLOR_BGR2HSV),
                        np.array([0, 50, 250]),
                        np.array([35, 255, 255]))

    cv2.imshow("Fire Mask Debug", fire_mask)  # Shows detected fire region



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
