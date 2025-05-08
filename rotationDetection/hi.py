import cv2
import numpy as np
import time
import os # Added for checking file source type

# --- Parameters You MUST Adjust ---

# 1. Video Source:
# Use 0, 1, etc. for cameras, or "your_video.mp4" for a file
VIDEO_SOURCE = "video.mp4"  # Change to your video file path if needed

# 2. Region of Interest (ROI) [x, y, width, height]
# Manually determine this rectangle around the hub with the markings
ROI = [600, 432, 135, 122] # Using your provided ROI

# 3. Motion Threshold (pixels)
MOTION_THRESHOLD = 0.5  # Adjust as needed

# 4. Log File Name
LOG_FILENAME = "motion_log.txt"

# --- Feature Detection and Tracking Parameters ---
feature_params = dict( maxCorners = 100, qualityLevel = 0.3, minDistance = 7, blockSize = 7 )
lk_params = dict( winSize  = (15, 15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
REDETECT_INTERVAL = 30 # Re-detect features every N frames

# --- Initialization ---

cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print(f"Error: Could not open video source: {VIDEO_SOURCE}")
    exit()

# --- Get Video Properties for Playback Speed Control ---
original_fps = cap.get(cv2.CAP_PROP_FPS)
wait_time_ms = 1 # Default wait time if FPS is not available
is_camera = isinstance(VIDEO_SOURCE, int) # Check if source is camera or file

if original_fps > 0:
    print(f"Original Video FPS: {original_fps}")
    wait_time_ms = int(1000 / original_fps) # Time per frame in milliseconds
else:
    print("Warning: Could not determine video FPS. Playback speed might be incorrect.")
    # For cameras, FPS might be 0 initially, or for some file formats.
    # Use a default assumption if needed, e.g., wait_time_ms = 33 for ~30 FPS
    if is_camera:
        print("Assuming camera, will use minimal wait time.")
        wait_time_ms = 1 # Process camera frames as fast as possible
    else:
         print("Using default wait time of 33ms (~30 FPS assumption).")
         wait_time_ms = 33 # Assume ~30 FPS for files if unknown


# --- Read First Frame ---
ret, first_frame = cap.read()
if not ret:
    print("Error: Could not read first frame.")
    cap.release()
    exit()

# --- ROI Setup ---
frame_height, frame_width = first_frame.shape[:2]
x, y, w, h = ROI
if x < 0 or y < 0 or x + w > frame_width or y + h > frame_height or w <= 0 or h <= 0:
     print(f"Error: Invalid ROI {ROI} for frame size {frame_width}x{frame_height}")
     cap.release()
     exit()

prev_roi = first_frame[y:y+h, x:x+w]
prev_gray = cv2.cvtColor(prev_roi, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(prev_gray, mask = None, **feature_params)
if p0 is None:
    print("Warning: No initial features found in the ROI.")
    p0 = np.array([], dtype=np.float32).reshape(0, 1, 2)

mask = np.zeros_like(prev_roi)
frame_count = 0
last_time = time.time() # For FPS calculation of processing loop
previous_motion_status = False # Keep track of motion status in the *previous* frame

# --- Open Log File ---
try:
    with open(LOG_FILENAME, 'w') as log_file:
        log_file.write("Timestamp(s),Event\n") # Write header

        # --- Main Loop ---
        while True:
            loop_start_time = time.time() # For calculating processing time

            ret, frame = cap.read()
            if not ret:
                print("End of video or cannot read frame.")
                break

            # --- Get Accurate Timestamp ---
            if not is_camera:
                # Use video's internal millisecond counter for files
                current_timestamp_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            else:
                # Use system time for live camera feeds
                current_timestamp_sec = time.time()

            frame_count += 1

            # --- Processing ---
            frame_roi = frame[y:y+h, x:x+w]
            gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
            motion_detected_this_frame = False
            avg_magnitude = 0.0

            # Re-detect features periodically or if lost
            if p0 is None or len(p0) < 5 or frame_count % REDETECT_INTERVAL == 0:
                p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
                mask = np.zeros_like(prev_roi)
                if p0 is None:
                    p0 = np.array([], dtype=np.float32).reshape(0, 1, 2)
                    # print("Warning: No features found during re-detection.") # Optional: can be noisy

            # Calculate optical flow
            if p0 is not None and len(p0) > 0:
                p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk_params)

                if p1 is not None and st is not None:
                    good_new = p1[st == 1]
                    good_old = p0[st == 1]

                    if len(good_new) > 0:
                        displacement = good_new - good_old
                        magnitudes = np.linalg.norm(displacement, axis=1)
                        avg_magnitude = np.mean(magnitudes)

                        # --- Motion Detection Logic ---
                        if avg_magnitude > MOTION_THRESHOLD:
                            motion_detected_this_frame = True

                        # Draw tracks
                        for i, (new, old) in enumerate(zip(good_new, good_old)):
                            a, b = new.ravel().astype(int)
                            c, d = old.ravel().astype(int)
                            mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 1)
                            frame_roi = cv2.circle(frame_roi, (a, b), 3, (0, 0, 255), -1)

                        p0 = good_new.reshape(-1, 1, 2)
                    else: p0 = np.array([], dtype=np.float32).reshape(0, 1, 2) # Force re-detect
                else: p0 = np.array([], dtype=np.float32).reshape(0, 1, 2) # Force re-detect
            else: p0 = np.array([], dtype=np.float32).reshape(0, 1, 2) # Force re-detect

            # --- Logging Motion Changes ---
            if motion_detected_this_frame and not previous_motion_status:
                log_message = f"{current_timestamp_sec:.3f},Motion Started\n"
                print(log_message.strip()) # Also print to console
                log_file.write(log_message)
                log_file.flush() # Ensure it's written immediately
            elif not motion_detected_this_frame and previous_motion_status:
                log_message = f"{current_timestamp_sec:.3f},Motion Stopped\n"
                print(log_message.strip()) # Also print to console
                log_file.write(log_message)
                log_file.flush() # Ensure it's written immediately

            # Update status for the next frame's comparison
            previous_motion_status = motion_detected_this_frame

            # --- Display ---
            img_display = cv2.add(frame_roi, mask)
            status_text = "MOTION DETECTED" if motion_detected_this_frame else "No Motion"
            status_color = (0, 255, 0) if motion_detected_this_frame else (0, 0, 255)
            cv2.putText(img_display, status_text, (5, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1, cv2.LINE_AA)
            cv2.putText(img_display, f"Avg Mag: {avg_magnitude:.2f} px", (5, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow('ROI Motion Detection', img_display)

            # Optional: Display full frame
            current_loop_time = time.time()
            processing_fps = 1.0 / (current_loop_time - last_time) if (current_loop_time - last_time) > 0 else 0
            last_time = current_loop_time

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Processing FPS: {processing_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if not is_camera:
                 cv2.putText(frame, f"Video Time: {current_timestamp_sec:.2f}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow('Full Frame', frame)

            # --- Update previous frame ---
            prev_gray = gray.copy()

            # --- Wait Key Logic for Playback Speed Control ---
            proc_time_ms = (time.time() - loop_start_time) * 1000
            actual_wait_time = max(1, wait_time_ms - int(proc_time_ms)) # Ensure wait is at least 1ms

            if cv2.waitKey(actual_wait_time) & 0xFF == ord('q'):
                break

except IOError:
    print(f"Error: Could not open or write to log file: {LOG_FILENAME}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
finally:
    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    print(f"Log saved to {LOG_FILENAME}")