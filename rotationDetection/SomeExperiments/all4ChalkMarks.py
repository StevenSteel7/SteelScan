import cv2
import numpy as np
import math
import sys

# --- Parameters ---
VIDEO_SOURCE = 'fronti.mp4'

ROI_X, ROI_Y, ROI_W, ROI_H = 630, 460, 73, 65
CENTER_X_ROI, CENTER_Y_ROI = 35, 40

CHALK_HSV_LOWER = np.array([0, 0, 148])
CHALK_HSV_UPPER = np.array([180, 253, 255])

MIN_CHALK_AREA = 6
MAX_CHALK_AREA = 500

DISTANCE_THRESHOLD = 15

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
LINE_THICKNESS = 2
MARK_COLOR = (255, 0, 0)
CENTER_COLOR = (0, 0, 255)
ROI_COLOR = (0, 255, 0)
INFO_COLOR = (255, 255, 255)

# --- Helper Functions ---
def safe_divide(numerator, denominator):
    return 0 if denominator == 0 else numerator / denominator

def calculate_angular_difference(angle1_deg, angle2_deg):
    diff = angle1_deg - angle2_deg
    while diff <= -180:
        diff += 360
    while diff > 180:
        diff -= 360
    return diff

# --- Main Processing ---
cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    print(f"Error: Could not open video file: {VIDEO_SOURCE}")
    sys.exit()

# --- Read FPS for Real-Time Playback ---
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30  # fallback
wait_time = int(1000 / fps)

previous_detected_marks = []
frame_count = 0
paused = False
processed_display_frame = None # Declare this variable

ret, first_frame = cap.read()
if not ret:
    print("Error: Could not read the first frame.")
    cap.release()
    sys.exit()

frame_h, frame_w = first_frame.shape[:2]
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

print(f"Video Frame Size: {frame_w}x{frame_h}, FPS: {fps}")

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("End of video reached. Restarting...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_count += 1
        display_frame = frame.copy()
    else:
        display_frame = processed_display_frame.copy() # Use the stored frame

    if not paused:
        roi = frame[ROI_Y:ROI_Y+ROI_H, ROI_X:ROI_X+ROI_W]
        if roi.size == 0:
            print(f"Warning: ROI empty at frame {frame_count}.")
            continue

        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, CHALK_HSV_LOWER, CHALK_HSV_UPPER)

        # --- Morphological Cleaning ---
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Contour Detection
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sort contours by area (largest first) and select top 4
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:4]

        detected_marks_roi = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if MIN_CHALK_AREA < area < MAX_CHALK_AREA:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    mark_x_roi = int(safe_divide(M["m10"], M["m00"]))
                    mark_y_roi = int(safe_divide(M["m01"], M["m00"]))
                    detected_marks_roi.append((mark_x_roi, mark_y_roi))

        # --- Matching by Distance ---
        average_delta = 0.0
        num_matches = 0

        if previous_detected_marks and detected_marks_roi:
            matched = []
            for curr_mark in detected_marks_roi:
                best_distance = float('inf')
                best_prev = None
                for prev_mark in previous_detected_marks:
                    dist = math.hypot(curr_mark[0] - prev_mark[0], curr_mark[1] - prev_mark[1])
                    if dist < best_distance:
                        best_distance = dist
                        best_prev = prev_mark

                if best_distance < DISTANCE_THRESHOLD:
                    matched.append((curr_mark, best_prev))

            if matched:
                angle_deltas = []
                for (curr, prev) in matched:
                    curr_angle = math.degrees(math.atan2(curr[1] - CENTER_Y_ROI, curr[0] - CENTER_X_ROI))
                    prev_angle = math.degrees(math.atan2(prev[1] - CENTER_Y_ROI, prev[0] - CENTER_X_ROI))
                    delta = calculate_angular_difference((curr_angle + 360) % 360, (prev_angle + 360) % 360)
                    angle_deltas.append(delta)

                if angle_deltas:
                    average_delta = sum(angle_deltas) / len(angle_deltas)
                    num_matches = len(angle_deltas)

        previous_detected_marks = detected_marks_roi.copy()

        # --- Visualization ---
        center_full_x = ROI_X + CENTER_X_ROI
        center_full_y = ROI_Y + CENTER_Y_ROI
        cv2.rectangle(display_frame, (ROI_X, ROI_Y), (ROI_X+ROI_W, ROI_Y+ROI_H), ROI_COLOR, LINE_THICKNESS)
        cv2.circle(display_frame, (center_full_x, center_full_y), 5, CENTER_COLOR, -1)

        for mark_x_roi, mark_y_roi in detected_marks_roi:
            mark_full_x = ROI_X + mark_x_roi
            mark_full_y = ROI_Y + mark_y_roi
            cv2.circle(display_frame, (mark_full_x, mark_full_y), 4, MARK_COLOR, -1)

        # Status
        num_detected = len(detected_marks_roi)
        status_line1 = f"Frame: {frame_count}"
        status_line2 = f"Marks Detected: {num_detected}"
        status_line3 = f"Avg Rot: {average_delta:.1f} deg/fr ({num_matches} matched)"

        cv2.putText(display_frame, status_line1, (10, frame_h - 60), FONT, FONT_SCALE, INFO_COLOR, LINE_THICKNESS)
        cv2.putText(display_frame, status_line2, (10, frame_h - 40), FONT, FONT_SCALE, INFO_COLOR, LINE_THICKNESS)
        cv2.putText(display_frame, status_line3, (10, frame_h - 20), FONT, FONT_SCALE, INFO_COLOR, LINE_THICKNESS)

        processed_display_frame = display_frame.copy() # store the display frame

    # --- Show ---
    cv2.imshow('Frame', display_frame)

    if not paused:
        roi_display = roi.copy()
        mask_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        stacked = np.hstack((cv2.resize(roi_display, (400, 300)), cv2.resize(mask_display, (400, 300))))
        cv2.imshow('ROI + Mask', stacked)

    key = cv2.waitKey(wait_time) & 0xFF
    if key == ord('q'):
        print("Exit requested.")
        break
    elif key == ord('p'):
        paused = not paused
        print("Paused." if paused else "Resumed.")

cap.release()
cv2.destroyAllWindows()
print("Processing finished.")
