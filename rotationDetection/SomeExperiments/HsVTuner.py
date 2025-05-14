import cv2
import numpy as np
import random

# --- Parameters ---
VIDEO_SOURCE = 'fronti.mp4'  # Your video file

# Initial loose defaults
h_min, s_min, v_min = 0, 0, 100
h_max, s_max, v_max = 180, 100, 255

# --- Callback for trackbars ---
def nothing(x):
    pass

# --- Setup ---
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print("Error opening video stream or file")
    exit()

cv2.namedWindow("HSV Tuner", cv2.WINDOW_NORMAL)
cv2.resizeWindow("HSV Tuner", 500, 400)

# Create trackbars
cv2.createTrackbar("Hue Min", "HSV Tuner", h_min, 180, nothing)
cv2.createTrackbar("Hue Max", "HSV Tuner", h_max, 180, nothing)
cv2.createTrackbar("Sat Min", "HSV Tuner", s_min, 255, nothing)
cv2.createTrackbar("Sat Max", "HSV Tuner", s_max, 255, nothing)
cv2.createTrackbar("Val Min", "HSV Tuner", v_min, 255, nothing)
cv2.createTrackbar("Val Max", "HSV Tuner", v_max, 255, nothing)

paused = False

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("End of video reached — restarting...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to first frame
            continue

        # --- Trackbar read ---
        h_min = cv2.getTrackbarPos("Hue Min", "HSV Tuner")
        h_max = cv2.getTrackbarPos("Hue Max", "HSV Tuner")
        s_min = cv2.getTrackbarPos("Sat Min", "HSV Tuner")
        s_max = cv2.getTrackbarPos("Sat Max", "HSV Tuner")
        v_min = cv2.getTrackbarPos("Val Min", "HSV Tuner")
        v_max = cv2.getTrackbarPos("Val Max", "HSV Tuner")

        lower_bound = np.array([h_min, s_min, v_min])
        upper_bound = np.array([h_max, s_max, v_max])

        # --- Process frame ---
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        result = cv2.bitwise_and(frame, frame, mask=mask)

        # --- Visualization merge ---
        frame_resized = cv2.resize(frame, (500, 400))
        mask_resized = cv2.resize(mask, (500, 400))
        result_resized = cv2.resize(result, (500, 400))

        # Convert mask to BGR to merge properly
        mask_bgr = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)

        # Horizontal stack
        combined = np.hstack((frame_resized, mask_bgr, result_resized))

        # --- Final display ---
        cv2.namedWindow('Original | Mask | Result', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Original | Mask | Result', 1600, 500)
        cv2.imshow('Original | Mask | Result', combined)

    # --- Keyboard Interaction ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Quitting...")
        break
    elif key == ord('p'):
        paused = not paused
        print("Paused." if paused else "Resumed.")
    elif key == ord('r'):
        # --- Randomize HSV values ---
        random_h_min = random.randint(0, 90)
        random_h_max = random.randint(random_h_min+10, 180)
        random_s_min = random.randint(0, 50)
        random_s_max = random.randint(random_s_min+20, 255)
        random_v_min = random.randint(100, 200)
        random_v_max = random.randint(random_v_min+20, 255)

        # Set to trackbars
        cv2.setTrackbarPos("Hue Min", "HSV Tuner", random_h_min)
        cv2.setTrackbarPos("Hue Max", "HSV Tuner", random_h_max)
        cv2.setTrackbarPos("Sat Min", "HSV Tuner", random_s_min)
        cv2.setTrackbarPos("Sat Max", "HSV Tuner", random_s_max)
        cv2.setTrackbarPos("Val Min", "HSV Tuner", random_v_min)
        cv2.setTrackbarPos("Val Max", "HSV Tuner", random_v_max)

        print(f"Random HSV Set: H=({random_h_min}-{random_h_max}) S=({random_s_min}-{random_s_max}) V=({random_v_min}-{random_v_max})")

cap.release()
cv2.destroyAllWindows()
