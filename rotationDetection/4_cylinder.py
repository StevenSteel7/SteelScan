import cv2
import numpy as np
import time
import os
import collections
import matplotlib.pyplot as plt

# --- Parameters You MUST Adjust ---

# 1. Video Source:
VIDEO_SOURCE = "fronti.mp4"

# 2. Regions of Interest (ROIs)
ROIS = [
    [600, 432, 125, 120], # ROI 0: Hub
    [470, 432, 125, 120], # ROI 1: Cylinder 1
    [765, 432, 127, 120], # ROI 2: Cylinder 2
    [900, 432, 127, 120]  # ROI 3: Cylinder 3
]
NUM_ROIS = len(ROIS)
if NUM_ROIS != 4:
    print(f"Warning: Code designed for 4 ROIs, but found {NUM_ROIS}. Display might be incorrect.")

# 3. Motion Threshold (Global for now)
MOTION_THRESHOLD = 0.5

# 4. Output Folder and File Names
OUTPUT_FOLDER = "4BboxAnylysis"
LOG_FILENAME_BASE = "motion_log_multi.txt"
FINAL_GRAPH_FILENAME_BASE = "motion_summary_plots.png"

# --- Create Output Folder ---
try:
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"Output will be saved in folder: '{OUTPUT_FOLDER}'")
except OSError as e:
    print(f"Error creating directory {OUTPUT_FOLDER}: {e}"); exit()

# --- Construct Full Paths ---
LOG_FILENAME = os.path.join(OUTPUT_FOLDER, LOG_FILENAME_BASE)
FINAL_GRAPH_FILENAME = os.path.join(OUTPUT_FOLDER, FINAL_GRAPH_FILENAME_BASE)

# 5. Graphing Parameters
GRAPH_HISTORY_LENGTH = 150 # Adjust history length if needed
# Parameters for the LIVE graphs shown in the combined window
LIVE_GRAPH_WIDTH_PER_ROI = 250 # Width of each small live graph
LIVE_GRAPH_PADDING = 10      # Padding around/between graphs
LIVE_GRAPH_MAX_MAGNITUDE = 5.0 # Y-axis limit for LIVE graphs (adjust!)
LIVE_GRAPH_MARGIN = 5        # Internal margin within each small graph
# Limit for the FINAL Matplotlib plots
PLOT_YLIM_TOP = 10.0

# --- Feature Detection and Tracking Parameters ---
feature_params = dict( maxCorners = 50, qualityLevel = 0.3, minDistance = 7, blockSize = 7 )
lk_params = dict( winSize  = (15, 15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
REDETECT_INTERVAL = 30

# --- Helper Function to Draw Live Graph ---
def draw_live_graph(canvas, history, roi_index, motion_status, x_offset, y_offset, width, height, max_mag, threshold):
    """Draws a single live graph onto the canvas."""
    # Define graph area within the given bounds
    graph_x = x_offset + LIVE_GRAPH_MARGIN
    graph_y = y_offset + LIVE_GRAPH_MARGIN
    graph_w = width - 2 * LIVE_GRAPH_MARGIN
    graph_h = height - 2 * LIVE_GRAPH_MARGIN

    if graph_w <= 0 or graph_h <= 0: return # Avoid drawing if area is too small

    # Draw background rectangle (optional)
    cv2.rectangle(canvas, (x_offset, y_offset), (x_offset + width, y_offset + height), (40, 40, 40), -1)

    # Draw Threshold line
    if max_mag > 0:
        thresh_line_y = graph_y + graph_h - int(np.clip(threshold / max_mag, 0, 1) * graph_h)
        cv2.line(canvas, (graph_x, thresh_line_y), (graph_x + graph_w, thresh_line_y), (0, 100, 100), 1)

    # Prepare points for plotting
    points = []
    hist_len = len(history)
    for i, mag in enumerate(history):
        y_coord = graph_y + graph_h - int(np.clip(mag / max_mag, 0, 1) * graph_h)
        x_coord = graph_x + int((i / (GRAPH_HISTORY_LENGTH - 1 if GRAPH_HISTORY_LENGTH > 1 else 1)) * graph_w)
        points.append((x_coord, y_coord))

    # Draw motion history line
    if len(points) > 1:
        cv2.polylines(canvas, [np.array(points)], isClosed=False, color=(100, 255, 100), thickness=1)

    # Draw current magnitude marker
    if points:
        last_point = points[-1]
        marker_color = (0, 255, 0) if motion_status else (0, 0, 255)
        cv2.circle(canvas, last_point, 3, marker_color, -1)

    # Add Text Label
    cv2.putText(canvas, f"ROI {roi_index}", (x_offset + 5, y_offset + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

# --- Initialization ---
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened(): print(f"Error: Could not open video source: {VIDEO_SOURCE}"); exit()

original_fps = cap.get(cv2.CAP_PROP_FPS)
wait_time_ms = 1
is_camera = isinstance(VIDEO_SOURCE, int)
if original_fps > 0: wait_time_ms = int(1000 / original_fps)
else: print("Warning: Could not determine video FPS."); wait_time_ms = 33 if not is_camera else 1

ret, first_frame = cap.read()
if not ret: print("Error: Could not read first frame."); cap.release(); exit()

frame_height, frame_width = first_frame.shape[:2]

# --- Initialize Per-ROI Data Structures ---
prev_grays = {}
prev_pts = {}
motion_histories = {i: collections.deque(maxlen=GRAPH_HISTORY_LENGTH) for i in range(NUM_ROIS)}
previous_motion_statuses = {i: False for i in range(NUM_ROIS)}
all_timestamps = []
all_magnitudes_per_roi = {i: [] for i in range(NUM_ROIS)}

# --- Initialize features for each ROI ---
print("Initializing features for each ROI...")
for i in range(NUM_ROIS):
    x, y, w, h = ROIS[i]
    if x < 0 or y < 0 or x + w > frame_width or y + h > frame_height or w <= 0 or h <= 0:
        print(f"Error: Invalid ROI {i}: {ROIS[i]}"); cap.release(); exit()
    prev_roi = first_frame[y:y+h, x:x+w]
    prev_grays[i] = cv2.cvtColor(prev_roi, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(prev_grays[i], mask=None, **feature_params)
    prev_pts[i] = p0 if p0 is not None else np.array([], dtype=np.float32).reshape(0, 1, 2)
    if prev_pts[i].size == 0: print(f"Warning: No initial features found in ROI {i}.")
print("Initialization complete.")

frame_count = 0
start_system_time = time.time()

# --- Calculate dimensions for the combined window and graphs ---
live_graph_area_width = LIVE_GRAPH_WIDTH_PER_ROI + LIVE_GRAPH_PADDING
combined_width = frame_width + live_graph_area_width
# Calculate height needed for graphs to fit next to the video frame
total_graph_area_height_needed = (NUM_ROIS * (LIVE_GRAPH_PADDING)) + LIVE_GRAPH_PADDING # Sum of heights + padding
graph_height_per_roi = 0
if NUM_ROIS > 0:
    available_height_for_graphs = frame_height - (NUM_ROIS + 1) * LIVE_GRAPH_PADDING
    if available_height_for_graphs > 0:
         graph_height_per_roi = available_height_for_graphs // NUM_ROIS
    else:
        print("Warning: Frame height too small to display live graphs effectively.")
        graph_height_per_roi = 20 # Assign a small default height

combined_height = frame_height # Match combined window height to frame height

# --- Open Log File ---
try:
    with open(LOG_FILENAME, 'w') as log_file:
        log_file.write("Timestamp(s),ROI_Index,Event,AvgMagnitude\n")

        # --- Main Loop ---
        while True:
            loop_start_time = time.time()
            ret, frame = cap.read()
            if not ret: break

            # --- Get Timestamp ---
            if not is_camera and original_fps > 0:
                current_timestamp_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                if frame_count > 0 and len(all_timestamps) > 0 and current_timestamp_sec <= all_timestamps[-1]:
                    current_timestamp_sec = all_timestamps[-1] + (1.0 / original_fps)
            else: current_timestamp_sec = time.time() - start_system_time
            frame_count += 1
            all_timestamps.append(current_timestamp_sec)

            # --- Process Each ROI ---
            frame_with_rois = frame.copy() # For drawing ROIs on video feed part
            for i in range(NUM_ROIS):
                # (Optical flow calculation is the same as previous version)
                # ...
                x, y, w, h = ROIS[i]; motion_detected = False; avg_magnitude = 0.0
                frame_roi = frame[y:y+h, x:x+w]; gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
                p0 = prev_pts[i]; prev_gray = prev_grays[i]
                if p0 is None or len(p0) < 5 or frame_count % REDETECT_INTERVAL == 0:
                    p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
                    if p0 is None: p0 = np.array([], dtype=np.float32).reshape(0, 1, 2)
                    prev_pts[i] = p0
                if p0 is not None and len(p0) > 0:
                    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk_params)
                    if p1 is not None and st is not None:
                        good_new = p1[st == 1]; good_old = p0[st == 1]
                        if len(good_new) > 0:
                            displacement = good_new - good_old; magnitudes = np.linalg.norm(displacement, axis=1)
                            avg_magnitude = np.mean(magnitudes)
                            if avg_magnitude > MOTION_THRESHOLD: motion_detected = True
                            prev_pts[i] = good_new.reshape(-1, 1, 2)
                        else: prev_pts[i] = np.array([], dtype=np.float32).reshape(0, 1, 2)
                    else: prev_pts[i] = np.array([], dtype=np.float32).reshape(0, 1, 2)
                else: prev_pts[i] = np.array([], dtype=np.float32).reshape(0, 1, 2)

                # --- Store data ---
                all_magnitudes_per_roi[i].append(avg_magnitude)
                motion_histories[i].append(avg_magnitude)

                # --- Logging ---
                if motion_detected != previous_motion_statuses[i]:
                    event = "Motion Started" if motion_detected else "Motion Stopped"
                    log_message = f"{current_timestamp_sec:.3f},{i},{event},{avg_magnitude:.4f}\n"
                    print(f"ROI {i}: {log_message.strip()}")
                    log_file.write(log_message); log_file.flush()
                previous_motion_statuses[i] = motion_detected

                # --- Update prev gray ---
                prev_grays[i] = gray.copy()

                # --- Draw ROI on video feed part ---
                status_color = (0, 255, 0) if motion_detected else (0, 0, 255)
                cv2.rectangle(frame_with_rois, (x, y), (x + w, y + h), status_color, 2)
                cv2.putText(frame_with_rois, f"ROI {i}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1, cv2.LINE_AA)
            #--- End ROI Processing Loop ---

            # --- Create Combined Window Canvas ---
            combined_window = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)

            # --- Place Video Frame onto Canvas ---
            combined_window[0:frame_height, 0:frame_width] = frame_with_rois

            # --- Draw Live Graphs onto Canvas ---
            if graph_height_per_roi > 0: # Only draw if height is valid
                for i in range(NUM_ROIS):
                    graph_x_start = frame_width + LIVE_GRAPH_PADDING // 2
                    graph_y_start = LIVE_GRAPH_PADDING + i * (graph_height_per_roi + LIVE_GRAPH_PADDING)
                    draw_live_graph(
                        combined_window,
                        motion_histories[i],
                        i,
                        previous_motion_statuses[i], # Pass current status
                        graph_x_start,
                        graph_y_start,
                        LIVE_GRAPH_WIDTH_PER_ROI,
                        graph_height_per_roi,
                        LIVE_GRAPH_MAX_MAGNITUDE,
                        MOTION_THRESHOLD
                    )
            else: # Add text if graphs can't be drawn
                 cv2.putText(combined_window, "Graphs disabled", (frame_width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)


            # --- Add general info text ---
            processing_fps = 1.0 / (time.time() - loop_start_time) if (time.time() - loop_start_time) > 0 else 0
            cv2.putText(combined_window, f"Processing FPS: {processing_fps:.1f}", (10, frame_height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(combined_window, f"Video Time: {current_timestamp_sec:.2f}s", (10, frame_height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # --- Display Combined Window ---
            cv2.imshow('Multi-ROI Motion Detection & Live Graphs', combined_window)

            # --- Wait Key Logic ---
            proc_time_ms = (time.time() - loop_start_time) * 1000
            actual_wait_time = max(1, wait_time_ms - int(proc_time_ms))
            key = cv2.waitKey(actual_wait_time)
            if key != -1 and key & 0xFF == ord('q'):
                print("Exit requested by user."); break

except IOError as e: print(f"Error opening log file {LOG_FILENAME}: {e}")
except Exception as e: print(f"An unexpected error occurred: {e}")

finally:
    # --- Cleanup OpenCV ---
    print("Releasing video capture..."); cap.release()
    print("Destroying OpenCV windows..."); cv2.destroyAllWindows()

    # --- Generate and Save FINAL Matplotlib Graph (Multi-plot) ---
    # (This part remains the same as the previous version)
    print("Generating final summary graph for all ROIs...")
    try:
        if all_timestamps and any(val for roi_list in all_magnitudes_per_roi.values() for val in roi_list): # Check for any non-empty list
            fig, axes = plt.subplots(NUM_ROIS, 1, figsize=(15, 4 * NUM_ROIS), sharex=True)
            if NUM_ROIS == 1: axes = [axes]

            max_mag_overall = 0
            if PLOT_YLIM_TOP is None:
                 for i in range(NUM_ROIS):
                     # Check if list exists and is not empty before finding max
                     if i in all_magnitudes_per_roi and all_magnitudes_per_roi[i]:
                          max_mag_overall = max(max_mag_overall, max(all_magnitudes_per_roi[i]))
                 effective_ylim_top = max(max_mag_overall * 1.1, MOTION_THRESHOLD * 1.5) if max_mag_overall > 0 else MOTION_THRESHOLD * 1.5
            else:
                 effective_ylim_top = PLOT_YLIM_TOP

            for i in range(NUM_ROIS):
                ax = axes[i]
                # Check if data exists for this ROI before plotting
                if i in all_magnitudes_per_roi and all_magnitudes_per_roi[i]:
                    ax.plot(all_timestamps, all_magnitudes_per_roi[i], label=f'ROI {i} Avg Magnitude', linewidth=1)
                else:
                    ax.text(0.5, 0.5, 'No data for this ROI', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

                ax.axhline(MOTION_THRESHOLD, color='r', linestyle='--', label=f'Threshold={MOTION_THRESHOLD}')
                ax.set_ylabel("Avg Motion Magnitude (px)")
                plot_title = f"ROI {i} Motion Magnitude"
                if PLOT_YLIM_TOP is not None:
                     plot_title += f" [Y-Axis Clipped at {effective_ylim_top:.1f}]"
                ax.set_title(plot_title)
                ax.legend()
                ax.grid(True)
                ax.set_ylim(bottom=-0.5, top=effective_ylim_top)

            axes[-1].set_xlabel("Time (s)")
            plt.tight_layout(rect=[0, 0.03, 1, 0.98])
            plt.savefig(FINAL_GRAPH_FILENAME, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Final summary graph saved to {FINAL_GRAPH_FILENAME}")

        else:
            print("Warning: No motion data collected for any ROI, cannot generate final graph.")

    except Exception as e: print(f"An error occurred while generating/saving the final graph: {e}")

    print(f"Log file saved to {LOG_FILENAME}")
    print("Script finished.")