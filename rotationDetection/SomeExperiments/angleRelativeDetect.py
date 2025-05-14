import cv2
import numpy as np
import time
import os

# --- Parameters You MUST Adjust ---

# 1. Video Source
VIDEO_SOURCE = "fronti.mp4"  # Change to your video file

# 2. Region of Interest (ROI) [x, y, width, height]
ROI = [600, 432, 135, 122]  # Tune if needed

# 3. Motion Threshold (pixels)
MOTION_THRESHOLD = 0.5

# 4. Log File Name
LOG_FILENAME = "motion_log.txt"

# --- Feature Detection and Tracking Parameters ---
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
REDETECT_INTERVAL = 30


motion_stop_time = None  # Timestamp when motion last stopped
IDLE_RESET_SECONDS = 3   # Configurable idle duration


# --- Initialization ---

cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print(f"Error: Could not open video source: {VIDEO_SOURCE}")
    exit()

# --- Get Video Properties ---
original_fps = cap.get(cv2.CAP_PROP_FPS)
wait_time_ms = 1 if original_fps <= 0 else int(1000 / original_fps)
is_camera = isinstance(VIDEO_SOURCE, int)

# --- Read First Frame ---
ret, first_frame = cap.read()
if not ret:
    print("Error: Could not read first frame.")
    cap.release()
    exit()

frame_height, frame_width = first_frame.shape[:2]
x, y, w, h = ROI
if x < 0 or y < 0 or x + w > frame_width or y + h > frame_height or w <= 0 or h <= 0:
    print(f"Error: Invalid ROI {ROI} for frame size {frame_width}x{frame_height}")
    cap.release()
    exit()

prev_roi = first_frame[y:y+h, x:x+w]
prev_gray = cv2.cvtColor(prev_roi, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
if p0 is None:
    p0 = np.array([], dtype=np.float32).reshape(0, 1, 2)

mask = np.zeros_like(prev_roi)
frame_count = 0
last_time = time.time()
previous_motion_status = False

# --- New for Rotation Tracking ---
cumulative_rotation = 0.0

# --- Open Log File ---
try:
    with open(LOG_FILENAME, 'w') as log_file:
        log_file.write("Timestamp(s),Event,Rotation(deg)\n")

        while True:
            loop_start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                print("End of video or cannot read frame.")
                break

            current_timestamp_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 if not is_camera else time.time()
            frame_count += 1

            frame_roi = frame[y:y+h, x:x+w].copy() # Create a copy for drawing on
            gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
            motion_detected_this_frame = False
            avg_magnitude = 0.0
            rotation_angle_this_frame = 0.0

            # Re-detect features
            if p0 is None or len(p0) < 5 or frame_count % REDETECT_INTERVAL == 0:
                p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
                mask = np.zeros_like(prev_roi)
                if p0 is None:
                    p0 = np.array([], dtype=np.float32).reshape(0, 1, 2)

            # Optical Flow
            if p0 is not None and len(p0) > 0:
                p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk_params)

                if p1 is not None and st is not None:
                    good_new = p1[st == 1]
                    good_old = p0[st == 1]

                    if len(good_new) > 0:
                        displacement = good_new - good_old
                        magnitudes = np.linalg.norm(displacement, axis=1)
                        avg_magnitude = np.mean(magnitudes)

                        if avg_magnitude > MOTION_THRESHOLD:
                            motion_detected_this_frame = True

                        # Draw tracks on the ROI copy
                        for i, (new, old) in enumerate(zip(good_new, good_old)):
                            a, b = new.ravel().astype(int)
                            c, d = old.ravel().astype(int)
                            mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 1)
                            cv2.circle(frame_roi, (a, b), 3, (0, 0, 255), -1)

                        # 🔵 --- ROTATION EXTRACTION BLOCK ---
                        if len(good_old) >= 3:
                            matrix, inliers = cv2.estimateAffinePartial2D(good_old, good_new)
                            if matrix is not None:
                                rotation_radians = np.arctan2(matrix[1, 0], matrix[0, 0])
                                rotation_angle_this_frame = np.degrees(rotation_radians)
                                cumulative_rotation += rotation_angle_this_frame
                                # Log this rotation
                                print(f"Rotation this frame: {rotation_angle_this_frame:.2f} deg | Cumulative: {cumulative_rotation:.2f} deg")
                        # 🔵 --- ROTATION EXTRACTION BLOCK END ---

                        p0 = good_new.reshape(-1, 1, 2)
                    else:
                        p0 = np.array([], dtype=np.float32).reshape(0, 1, 2)
                else:
                    p0 = np.array([], dtype=np.float32).reshape(0, 1, 2)
            else:
                p0 = np.array([], dtype=np.float32).reshape(0, 1, 2)

            # Log motion changes
            if motion_detected_this_frame and not previous_motion_status:
                log_message = f"{current_timestamp_sec:.3f},Motion Started,{rotation_angle_this_frame:.2f}\n"
                print(log_message.strip())
                log_file.write(log_message)
                log_file.flush()
            elif not motion_detected_this_frame and previous_motion_status:
                log_message = f"{current_timestamp_sec:.3f},Motion Stopped,{rotation_angle_this_frame:.2f}\n"
                print(log_message.strip())
                log_file.write(log_message)
                log_file.flush()

            previous_motion_status = motion_detected_this_frame


                        # --- Rotation Reset if No Motion for 3 seconds ---
            if not motion_detected_this_frame:
                if motion_stop_time is None:
                    motion_stop_time = time.time()  # Mark when motion first stops
                elif time.time() - motion_stop_time > IDLE_RESET_SECONDS:
                    print(f"Machine idle for {IDLE_RESET_SECONDS} sec, resetting rotation counter.")
                    cumulative_rotation = 0.0
                    motion_stop_time = None  # Reset the stop timer
            else:
                motion_stop_time = None  # Reset if motion is ongoing
            # --- End Rotation Reset ---


            # Display on the full frame
            img_display_roi = cv2.add(frame_roi, mask)
            status_text = "MOTION DETECTED" if motion_detected_this_frame else "No Motion"
            status_color = (0, 255, 0) if motion_detected_this_frame else (0, 0, 255)
            cv2.putText(img_display_roi, status_text, (5, h-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
            cv2.putText(img_display_roi, f"Avg Mag: {avg_magnitude:.2f} px", (5, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            cv2.putText(img_display_roi, f"Rot: {cumulative_rotation:.2f} deg", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # Incorporate the ROI display into the full frame
            frame[y:y+h, x:x+w] = img_display_roi

            # Optional Full Frame Info
            proc_time_ms = (time.time() - loop_start_time) * 1000
            actual_wait_time = max(1, wait_time_ms - int(proc_time_ms))
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Video Time: {current_timestamp_sec:.2f}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow('Motion and Rotation Detection', frame) # Show the combined frame

            prev_gray = gray.copy()

            if cv2.waitKey(actual_wait_time) & 0xFF == ord('q'):
                break

except IOError:
    print(f"Error: Could not open or write to log file: {LOG_FILENAME}")
except Exception as e:
    print(f"Unexpected error: {e}")
finally:
    cap.release()
    cv2.destroyAllWindows()
    print(f"Log saved to {LOG_FILENAME}")