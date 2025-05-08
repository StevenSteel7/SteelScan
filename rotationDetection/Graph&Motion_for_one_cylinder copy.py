import cv2
import numpy as np
import time
import os # Make sure os is imported
import collections
import matplotlib.pyplot as plt

# --- Parameters You MUST Adjust ---
VIDEO_SOURCE = "fronti.mp4"
ROI = [600, 432, 125, 165]
MOTION_THRESHOLD = 0.5
OUTPUT_FOLDER = "SingleBboxAnylysis" # Define the output folder name

# --- Filenames (will be placed inside OUTPUT_FOLDER) ---
LOG_FILENAME_BASE = "motion_log.txt"
FINAL_GRAPH_FILENAME_BASE = "motion_summary_plot.png"

# --- Create Output Folder ---
try:
    os.makedirs(OUTPUT_FOLDER, exist_ok=True) # exist_ok=True prevents error if folder exists
    print(f"Output folder: '{OUTPUT_FOLDER}'")
except OSError as e:
    print(f"Error creating directory {OUTPUT_FOLDER}: {e}")
    exit() # Exit if we can't create the folder

# --- Construct Full Paths ---
LOG_FILENAME = os.path.join(OUTPUT_FOLDER, LOG_FILENAME_BASE)
FINAL_GRAPH_FILENAME = os.path.join(OUTPUT_FOLDER, FINAL_GRAPH_FILENAME_BASE)

# --- Graphing Parameters ---
GRAPH_HISTORY_LENGTH = 200
GRAPH_WIDTH = 500
GRAPH_HEIGHT = 200
GRAPH_MAX_MAGNITUDE = 5.0 # Adjust for LIVE graph
GRAPH_MARGIN = 10

# --- Feature Detection and Tracking Parameters ---
feature_params = dict( maxCorners = 100, qualityLevel = 0.3, minDistance = 7, blockSize = 7 )
lk_params = dict( winSize  = (15, 15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
REDETECT_INTERVAL = 30

# --- Initialization ---
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print(f"Error: Could not open video source: {VIDEO_SOURCE}")
    exit()

original_fps = cap.get(cv2.CAP_PROP_FPS)
wait_time_ms = 1
is_camera = isinstance(VIDEO_SOURCE, int)

if original_fps > 0:
    print(f"Original Video FPS: {original_fps}")
    wait_time_ms = int(1000 / original_fps)
else:
    print("Warning: Could not determine video FPS.")
    if is_camera: wait_time_ms = 1
    else: wait_time_ms = 33

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
p0 = cv2.goodFeaturesToTrack(prev_gray, mask = None, **feature_params)
if p0 is None:
    print("Warning: No initial features found in the ROI.")
    p0 = np.array([], dtype=np.float32).reshape(0, 1, 2)

mask = np.zeros_like(prev_roi)
frame_count = 0
last_time = time.time()
previous_motion_status = False

# --- Initialize LIVE Graph ---
motion_history = collections.deque(maxlen=GRAPH_HISTORY_LENGTH)
graph_img = np.zeros((GRAPH_HEIGHT, GRAPH_WIDTH, 3), dtype=np.uint8)

# --- Initialize Data Storage for FINAL Graph ---
all_timestamps = []
all_magnitudes = []

# --- Open Log File ---
try:
    # Use the full path LOG_FILENAME defined earlier
    with open(LOG_FILENAME, 'w') as log_file:
        log_file.write("Timestamp(s),Event,AvgMagnitude\n")

        # --- Main Loop ---
        # (The entire main loop remains exactly the same as the previous version)
        # ...
        while True:
            loop_start_time = time.time()
            ret, frame = cap.read()
            if not ret: break
            # Timestamp calculation...
            # Optical flow calculation...
            # Storing data (all_timestamps, all_magnitudes)...
            # Update Live graph data...
            # Logging motion changes...
            # Display ROI...
            # Draw LIVE graph...
            # Display Full Frame...
            # Update prev_gray...
            # Wait key logic...

            # --- Get Timestamp ---
            if not is_camera and original_fps > 0: # Use video timestamp if reliable
                current_timestamp_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                if frame_count > 0 and all_timestamps and current_timestamp_sec <= all_timestamps[-1]:
                    current_timestamp_sec = all_timestamps[-1] + (1.0 / original_fps) # Estimate
            elif all_timestamps: # If camera or unreliable video time, use system time delta
                 # Calculate time relative to the first frame's system time
                 if frame_count == 0: start_system_time = time.time() # Store start time only once
                 current_timestamp_sec = time.time() - start_system_time
            else: # First frame time
                current_timestamp_sec = 0 # Start relative time at 0
                start_system_time = time.time() # Store actual start time

            frame_count += 1

            # --- Processing (Optical Flow) ---
            frame_roi = frame[y:y+h, x:x+w]
            gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
            motion_detected_this_frame = False
            avg_magnitude = 0.0
            if p0 is None or len(p0) < 5 or frame_count % REDETECT_INTERVAL == 0:
                p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
                mask = np.zeros_like(prev_roi)
                if p0 is None: p0 = np.array([], dtype=np.float32).reshape(0, 1, 2)
            if p0 is not None and len(p0) > 0:
                p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk_params)
                if p1 is not None and st is not None:
                    good_new = p1[st == 1]
                    good_old = p0[st == 1]
                    if len(good_new) > 0:
                        displacement = good_new - good_old
                        magnitudes = np.linalg.norm(displacement, axis=1)
                        avg_magnitude = np.mean(magnitudes)
                        if avg_magnitude > MOTION_THRESHOLD: motion_detected_this_frame = True
                        for new in good_new:
                            a, b = new.ravel().astype(int)
                            frame_roi = cv2.circle(frame_roi, (a, b), 3, (0, 0, 255), -1)
                        p0 = good_new.reshape(-1, 1, 2)
                    else: p0 = np.array([], dtype=np.float32).reshape(0, 1, 2)
                else: p0 = np.array([], dtype=np.float32).reshape(0, 1, 2)
            else: p0 = np.array([], dtype=np.float32).reshape(0, 1, 2)

            # --- Store Data ---
            all_timestamps.append(current_timestamp_sec)
            all_magnitudes.append(avg_magnitude)
            motion_history.append(avg_magnitude)

            # --- Logging ---
            if motion_detected_this_frame != previous_motion_status:
                event = "Motion Started" if motion_detected_this_frame else "Motion Stopped"
                log_message = f"{current_timestamp_sec:.3f},{event},{avg_magnitude:.4f}\n"
                print(log_message.strip())
                log_file.write(log_message)
                log_file.flush()
            previous_motion_status = motion_detected_this_frame

            # --- Display ---
            img_display = frame_roi
            status_text = "MOTION DETECTED" if motion_detected_this_frame else "No Motion"
            status_color = (0, 255, 0) if motion_detected_this_frame else (0, 0, 255)
            cv2.putText(img_display, status_text, (5, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1, cv2.LINE_AA)
            cv2.putText(img_display, f"Avg Mag: {avg_magnitude:.2f} px", (5, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow('ROI Motion Detection', img_display)

            # --- Draw LIVE Graph ---
            graph_img.fill(0)
            plot_area_height = GRAPH_HEIGHT - 2 * GRAPH_MARGIN
            plot_area_width = GRAPH_WIDTH - 2 * GRAPH_MARGIN
            if GRAPH_MAX_MAGNITUDE > 0:
                thresh_y = GRAPH_HEIGHT - GRAPH_MARGIN - int(np.clip(MOTION_THRESHOLD / GRAPH_MAX_MAGNITUDE, 0, 1) * plot_area_height)
                cv2.line(graph_img, (GRAPH_MARGIN, thresh_y), (GRAPH_WIDTH - GRAPH_MARGIN, thresh_y), (0, 100, 100), 1)
            points = []
            for i, mag in enumerate(motion_history):
                y_coord = GRAPH_HEIGHT - GRAPH_MARGIN - int(np.clip(mag / GRAPH_MAX_MAGNITUDE, 0, 1) * plot_area_height)
                x_coord = GRAPH_MARGIN + int((i / (GRAPH_HISTORY_LENGTH - 1 if GRAPH_HISTORY_LENGTH > 1 else 1)) * plot_area_width)
                points.append((x_coord, y_coord))
            if len(points) > 1: cv2.polylines(graph_img, [np.array(points)], isClosed=False, color=(100, 255, 100), thickness=1)
            if points:
                last_point = points[-1]
                marker_color = (0, 255, 0) if avg_magnitude > MOTION_THRESHOLD else (0, 0, 255)
                cv2.circle(graph_img, last_point, 3, marker_color, -1)
            cv2.putText(graph_img, "Live Motion Magnitude", (GRAPH_MARGIN, GRAPH_MARGIN + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(graph_img, f"Threshold: {MOTION_THRESHOLD:.2f}", (GRAPH_MARGIN + 150, GRAPH_MARGIN + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 100), 1)
            cv2.imshow('Live Motion Graph', graph_img)

            # --- Display Full Frame ---
            current_loop_time = time.time()
            processing_fps = 1.0 / (current_loop_time - last_time) if (current_loop_time - last_time) > 0 else 0
            last_time = current_loop_time
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Processing FPS: {processing_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Video Time: {current_timestamp_sec:.2f}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow('Full Frame', frame)

            # --- Update previous frame ---
            prev_gray = gray.copy()

            # --- Wait Key Logic ---
            proc_time_ms = (time.time() - loop_start_time) * 1000
            actual_wait_time = max(1, wait_time_ms - int(proc_time_ms))
            key = cv2.waitKey(actual_wait_time)
            if key != -1 and key & 0xFF == ord('q'):
                print("Exit requested by user.")
                break

except IOError as e:
    # Catch potential file opening errors
    print(f"Error opening log file {LOG_FILENAME}: {e}")
except Exception as e:
    print(f"An unexpected error occurred during processing: {e}")

finally:
    # --- Cleanup OpenCV ---
    print("Releasing video capture...")
    cap.release()
    print("Destroying OpenCV windows...")
    cv2.destroyAllWindows()

    # --- Generate and Save FINAL Matplotlib Graph ---
    print("Generating final summary graph...")
    try:
        if all_timestamps and all_magnitudes:
            PLOT_YLIM_TOP = 10.0

            fig, ax = plt.subplots(figsize=(15, 5))
            ax.plot(all_timestamps, all_magnitudes, label='Avg Motion Magnitude', linewidth=1)
            ax.axhline(MOTION_THRESHOLD, color='r', linestyle='--', label=f'Threshold={MOTION_THRESHOLD}')
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Avg Motion Magnitude (pixels)")
            plot_title = "Motion Magnitude Over Time (Full Session)"
            if PLOT_YLIM_TOP is not None:
                plot_title += f" [Y-Axis Clipped at {PLOT_YLIM_TOP}]"
            ax.set_title(plot_title)
            ax.legend()
            ax.grid(True)
            ax.set_ylim(bottom=-0.5)
            if PLOT_YLIM_TOP is not None:
                 ax.set_ylim(top=PLOT_YLIM_TOP)
            if all_timestamps:
                ax.set_xlim(left=min(all_timestamps)- (max(all_timestamps)*0.01) if all_timestamps else 0,
                            right=max(all_timestamps)*1.02 if all_timestamps else 1)

            # Use the full path FINAL_GRAPH_FILENAME defined earlier
            plt.savefig(FINAL_GRAPH_FILENAME, dpi=150, bbox_inches='tight')
            plt.close(fig)
            # Update print statement to show full path
            print(f"Final summary graph saved to {FINAL_GRAPH_FILENAME}")

        else:
            print("Warning: No motion data collected, cannot generate final graph.")

    except Exception as e:
        print(f"An error occurred while generating or saving the final graph: {e}")

    # Update print statement to show full path
    print(f"Log file saved to {LOG_FILENAME}")
    print("Script finished.")