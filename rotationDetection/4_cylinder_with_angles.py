import cv2
import numpy as np
import time
import os
import collections
import matplotlib.pyplot as plt
import math
import colorsys # To help find HSV values

# --- Parameters You MUST Adjust ---
# 1. Video Source:
VIDEO_SOURCE = "fronti.mp4"

# 2. Regions of Interest (ROIs) - Format: [x, y, width, height]
# Adjust these to precisely box the wheels.
ROIS = [
    [600, 432, 125, 120],  # ROI 0: Hub (Maybe use this for frame reference?)
    [470, 432, 125, 120],  # ROI 1: Wheel 1
    [765, 432, 127, 120],  # ROI 2: Wheel 2
    [900, 432, 127, 120]   # ROI 3: Wheel 3
]

graph_height_per_roi = 0

NUM_ROIS = len(ROIS)
if NUM_ROIS != 4:
    print(f"Warning: Code designed for 4 ROIs, but found {NUM_ROIS}. Display might be incorrect.")

# 3. Motion Threshold (Global for now) - Adjust based on pixel movement noise vs actual motion
# This threshold determines WHEN angle changes are accumulated.
MOTION_THRESHOLD = 0.5

# 4. Output Folder and File Names
OUTPUT_FOLDER = "4BboxAnalysis" # Corrected typo in folder name
LOG_FILENAME_BASE = "motion_log_multi.txt"
ANGLE_LOG_FILENAME_BASE = "angle_log_multi.txt"
FINAL_GRAPH_FILENAME_BASE = "motion_summary_plots.png"
ANGLE_GRAPH_FILENAME_BASE = "angle_summary_plots.png"

# 5. Angle Detection Parameters
USE_CHALK_MARK_DETECTION = True  # Set to True to use color-based chalk mark detection
# Parameters for HSV color thresholding for a LIGHT/WHITE mark
# You might need to adjust these based on your video.
# To find values: Open your video frame in paint/photoshop or use a color picker tool, get the RGB/HEX
# of the mark, convert to HSV. Or use a small script:
# color_bgr = np.uint8([[[255,255,255]]]) # Example white BGR
# color_hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
# print(color_hsv) # Gives approx HSV for that color
# For white/light colors, Hue (H) is less important, Saturation (S) should be low, Value (V) should be high.
CHALK_MARK_COLOR_LOWER = np.array([0, 0, 200])   # Example: Low Saturation, High Value
CHALK_MARK_COLOR_UPPER = np.array([180, 50, 255]) # Example: Low Saturation, High Value

# Minimum pixel area for a detected contour to be considered a chalk mark
CHALK_MARK_MIN_AREA = 10
# Maximum pixel area (optional, helps filter out large bright reflections)
CHALK_MARK_MAX_AREA = 2000 # Adjust based on ROI size and mark size

# 6. Graphing Parameters
GRAPH_HISTORY_LENGTH = 150  # Adjust history length for live graphs

# Parameters for the LIVE graphs shown in the combined window
LIVE_GRAPH_WIDTH_PER_ROI = 250  # Width of each small live graph
LIVE_GRAPH_PADDING = 10         # Padding around/between graphs
LIVE_GRAPH_MAX_MAGNITUDE = 5.0  # Y-axis limit for LIVE graphs (adjust!)
LIVE_GRAPH_MARGIN = 5           # Internal margin within each small graph

# Limit for the FINAL Matplotlib plots (Y-axis top limit for motion magnitude)
PLOT_YLIM_TOP = 10.0 # Set to None to auto-scale based on data

# --- Feature Detection and Tracking Parameters (for motion magnitude) ---
feature_params = dict(maxCorners=50, qualityLevel=0.3, minDistance=7, blockSize=7)
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
REDETECT_INTERVAL = 30

# --- Create Output Folder ---
try:
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"Output will be saved in folder: '{OUTPUT_FOLDER}'")
except OSError as e:
    print(f"Error creating directory {OUTPUT_FOLDER}: {e}")
    exit()

# --- Construct Full Paths ---
LOG_FILENAME = os.path.join(OUTPUT_FOLDER, LOG_FILENAME_BASE)
ANGLE_LOG_FILENAME = os.path.join(OUTPUT_FOLDER, ANGLE_LOG_FILENAME_BASE)
FINAL_GRAPH_FILENAME = os.path.join(OUTPUT_FOLDER, FINAL_GRAPH_FILENAME_BASE)
ANGLE_GRAPH_FILENAME = os.path.join(OUTPUT_FOLDER, ANGLE_GRAPH_FILENAME_BASE)

# --- Helper Function to Draw Live Graph (Motion) ---
def draw_live_graph(canvas, history, roi_index, motion_status, x_offset, y_offset, width, height, max_mag, threshold):
    """Draws a single live graph onto the canvas."""
    # Define graph area within the given bounds
    graph_x = x_offset + LIVE_GRAPH_MARGIN
    graph_y = y_offset + LIVE_GRAPH_MARGIN
    graph_w = width - 2 * LIVE_GRAPH_MARGIN
    graph_h = height - 2 * LIVE_GRAPH_MARGIN

    if graph_w <= 0 or graph_h <= 0:
        return  # Avoid drawing if area is too small

    # Draw background rectangle
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
    cv2.putText(canvas, f"ROI {roi_index} Motion", (x_offset + 5, y_offset + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    # Add current value
    if history:
        cv2.putText(canvas, f"{history[-1]:.2f}", (x_offset + 5, y_offset + height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


# --- Helper Function for Angle Detection ---
def detect_angle(roi_image):
    """
    Detects the angle of a wheel mark using color thresholding in HSV.
    
    Returns angle in degrees (0-360) from center to detected mark, or None if no mark found.
    Also returns the ROI image with visualization (for debugging/display).
    """
    h, w = roi_image.shape[:2]
    center = (w // 2, h // 2)
    
    # Convert to HSV
    hsv = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
    
    # Create mask for the chalk mark color
    mask = cv2.inRange(hsv, CHALK_MARK_COLOR_LOWER, CHALK_MARK_COLOR_UPPER)
    
    # Apply morphological operations to clean up the mask (optional, can help with noise)
    # kernel = np.ones((3,3), np.uint8)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours of potential chalk marks
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_contour = None
    largest_area = 0
    
    # Find the largest contour within area limits
    for contour in contours:
        area = cv2.contourArea(contour)
        if CHALK_MARK_MIN_AREA <= area <= CHALK_MARK_MAX_AREA:
            if area > largest_area:
                largest_area = area
                best_contour = contour
                
    roi_image_vis = roi_image.copy() # Create a copy for drawing visualization

    if best_contour is not None:
        # Find the center of mass of the contour
        M = cv2.moments(best_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Calculate angle from center to the chalk mark
            # Use atan2(dy, dx) where dy = cy - center[1], dx = cx - center[0]
            # Note: In OpenCV images, Y increases downwards, so dy should be positive downwards
            # atan2 returns radians in (-pi, pi]. math.degrees converts to (-180, 180].
            # We want 0-360, where 0 is pointing right, 90 downwards, 180 left, 270 upwards.
            # atan2(y, x) in math usually means angle from positive x-axis counter-clockwise.
            # In image coords with y down: atan2(dy, dx) gives angle where +x is right, +y is down.
            # 0 = right, 90 = down, 180 = left, -90 (or 270) = up. This matches standard image/atan2 conventions.
            angle = math.degrees(math.atan2(cy - center[1], cx - center[0]))
            
            # Convert to 0-360 range (if needed, atan2 output is often sufficient)
            # Let's keep it in (-180, 180] for easier diff calculation, but convert for display
            display_angle = (angle + 360) % 360
            
            # For visualization (debug)
            cv2.circle(roi_image_vis, (cx, cy), 5, (0, 255, 255), -1)  # Yellow dot at chalk mark
            # Draw line from center to mark
            center_abs = (center[0], center[1]) # Relative to ROI
            end_x = int(center_abs[0] + 40 * math.cos(math.radians(angle))) # Use raw angle for math
            end_y = int(center_abs[1] + 40 * math.sin(math.radians(angle)))
            cv2.line(roi_image_vis, center_abs, (end_x, end_y), (255, 0, 255), 2) # Magenta line
            
            return angle, roi_image_vis # Return angle in (-180, 180] and visualization

    # If no plausible mark found
    return None, roi_image_vis # Return None and the original ROI image (or blank if preferred)

def normalize_angle_diff(angle_diff):
    """Normalize angle difference to be between -180 and 180 degrees"""
    # Example: 350 deg difference -> -10 deg; -350 deg difference -> 10 deg
    angle_diff = (angle_diff + 180) % 360 - 180
    return angle_diff

# --- Initialization ---
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print(f"Error: Could not open video source: {VIDEO_SOURCE}")
    exit()

original_fps = cap.get(cv2.CAP_PROP_FPS)
wait_time_ms = 1
is_camera = isinstance(VIDEO_SOURCE, int)
if original_fps > 0:
    wait_time_ms = int(1000 / original_fps)
else:
    print("Warning: Could not determine video FPS. Using default wait time.")
    wait_time_ms = 33 if not is_camera else 1 # Approx 30 fps for video, fast for camera

ret, first_frame = cap.read()
if not ret:
    print("Error: Could not read first frame.")
    cap.release()
    exit()

frame_height, frame_width = first_frame.shape[:2]

# --- Initialize Per-ROI Data Structures ---
prev_grays = {} # Stores previous grayscale ROI image for optical flow
prev_pts = {}   # Stores previous feature points for optical flow

# Motion tracking history and status
motion_histories = {i: collections.deque(maxlen=GRAPH_HISTORY_LENGTH) for i in range(NUM_ROIS)}
previous_motion_statuses = {i: False for i in range(NUM_ROIS)} # Was motion detected in the last frame?

# Angle tracking data
# Stores the last successfully detected *absolute* angle (in -180 to 180 range)
last_successful_angle = {i: None for i in range(NUM_ROIS)}
# Stores the cumulative sum of angle *changes*
cumulative_angles = {i: 0.0 for i in range(NUM_ROIS)}

# Histories for plotting (store raw detection and cumulative)
# Raw angle history: stores the detected angle (-180 to 180) or None
raw_angle_histories = {i: collections.deque(maxlen=GRAPH_HISTORY_LENGTH) for i in range(NUM_ROIS)}
# Cumulative angle history: stores the cumulative angle
cumulative_angle_histories = {i: collections.deque(maxlen=GRAPH_HISTORY_LENGTH) for i in range(NUM_ROIS)}


# Full history for final plot
all_timestamps = []
all_magnitudes_per_roi = {i: [] for i in range(NUM_ROIS)}
all_raw_angles_per_roi = {i: [] for i in range(NUM_ROIS)} # Stores raw detected angles or None
all_cumulative_angles_per_roi = {i: [] for i in range(NUM_ROIS)}


# --- Initialize features and angles for each ROI ---
print("Initializing features and angle detection for each ROI...")
for i in range(NUM_ROIS):
    x, y, w, h = ROIS[i]
    # Validate ROI coordinates
    if x < 0 or y < 0 or x + w > frame_width or y + h > frame_height or w <= 0 or h <= 0:
        print(f"Error: Invalid ROI {i}: {ROIS[i]}. Dimensions are {frame_width}x{frame_height}.")
        cap.release()
        exit()

    prev_roi = first_frame[y:y+h, x:x+w].copy() # Use copy to avoid issues if ROI overlaps
    prev_grays[i] = cv2.cvtColor(prev_roi, cv2.COLOR_BGR2GRAY)

    # Initialize features for optical flow motion tracking
    p0 = cv2.goodFeaturesToTrack(prev_grays[i], mask=None, **feature_params)
    prev_pts[i] = p0 if p0 is not None else np.array([], dtype=np.float32).reshape(0, 1, 2)
    if prev_pts[i].size == 0:
        print(f"Warning: No initial features found in ROI {i}. Motion tracking might be unreliable.")

    # Initialize angle detection - find the first angle if possible
    detected_angle, _ = detect_angle(prev_roi)
    last_successful_angle[i] = detected_angle
    # Note: cumulative_angles[i] starts at 0, and will only increment if a change is detected
    # after the first successful angle detection.

print("Initialization complete. Starting video processing.")

frame_count = 0
start_system_time = time.time()

# --- Calculate dimensions for the combined window and graphs ---
live_graph_area_width = LIVE_GRAPH_WIDTH_PER_ROI + LIVE_GRAPH_PADDING
combined_width = frame_width + live_graph_area_width

# Calculate height needed for graphs to fit next to the video frame
# Assuming we stack NUM_ROIS motion graphs
graph_area_total_height_needed = NUM_ROIS * (graph_height_per_roi + LIVE_GRAPH_PADDING) + LIVE_GRAPH_PADDING
graph_height_per_roi = 0
if NUM_ROIS > 0:
    # Distribute available height among ROIs with padding
    available_height_for_graphs = frame_height - (NUM_ROIS + 1) * LIVE_GRAPH_PADDING
    if available_height_for_graphs > 0:
        graph_height_per_roi = available_height_for_graphs // NUM_ROIS
    else:
         # Fallback if frame is too short
        graph_height_per_roi = 50 # Minimum height
        print(f"Warning: Frame height ({frame_height}) too small for desired graph layout. Using minimum height {graph_height_per_roi}.")


combined_height = max(frame_height, graph_area_total_height_needed) # Ensure canvas is tall enough

# --- Open Log Files ---
try:
    with open(LOG_FILENAME, 'w') as log_file, open(ANGLE_LOG_FILENAME, 'w') as angle_log_file:
        log_file.write("Timestamp(s),ROI_Index,Event,AvgMagnitude\n")
        angle_log_file.write("Timestamp(s),ROI_Index,DetectedAngle(deg),AngleChange(deg),CumulativeAngle(deg)\n") # Log raw detected, change, and cumulative

        # --- Main Loop ---
        while True:
            loop_start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                print("End of video or cannot read frame.")
                break

            # --- Get Timestamp ---
            current_timestamp_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 if not is_camera and original_fps > 0 else time.time() - start_system_time
            # Ensure timestamps are increasing (can be an issue with cap.get(POS_MSEC))
            if frame_count > 0 and len(all_timestamps) > 0 and current_timestamp_sec <= all_timestamps[-1]:
                 # Estimate timestamp based on previous timestamp and FPS
                 current_timestamp_sec = all_timestamps[-1] + (1.0 / original_fps if original_fps > 0 else (time.time() - all_timestamps[-1]))

            all_timestamps.append(current_timestamp_sec)
            frame_count += 1


            # --- Create Combined Window Canvas ---
            # Dynamically adjust height if needed based on actual graph height
            current_combined_height = frame_height # Start with frame height
            if graph_height_per_roi > 0:
                 current_graph_area_height = NUM_ROIS * (graph_height_per_roi + LIVE_GRAPH_PADDING) + LIVE_GRAPH_PADDING
                 current_combined_height = max(frame_height, current_graph_area_height)

            combined_window = np.zeros((current_combined_height, combined_width, 3), dtype=np.uint8)

            # Place Video Frame onto Canvas (top-left corner)
            combined_window[0:frame_height, 0:frame_width] = frame

            frame_with_rois = frame.copy() # Copy for drawing ROIs/lines

            # --- Process Each ROI ---
            for i in range(NUM_ROIS):
                x, y, w, h = ROIS[i]

                # --- Motion detection (optical flow) ---
                frame_roi = frame[y:y+h, x:x+w] # Get the ROI frame
                gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)

                p0 = prev_pts[i]
                prev_gray = prev_grays[i]

                avg_magnitude = 0.0
                motion_detected = False

                # Redetect features periodically or if too few features are tracked
                if p0 is None or len(p0) < 5 or frame_count % REDETECT_INTERVAL == 0:
                     p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
                     if p0 is None:
                         p0 = np.array([], dtype=np.float32).reshape(0, 1, 2)
                     prev_pts[i] = p0

                if p0 is not None and len(p0) > 0:
                    # Calculate optical flow
                    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk_params)

                    if p1 is not None and st is not None:
                        # Select good points
                        good_new = p1[st == 1]
                        good_old = p0[st == 1]

                        if len(good_new) > 0:
                            # Calculate displacement magnitude for good points
                            displacement = good_new - good_old
                            magnitudes = np.linalg.norm(displacement, axis=1)
                            avg_magnitude = np.mean(magnitudes)

                            if avg_magnitude > MOTION_THRESHOLD:
                                motion_detected = True

                            # Update points for the next frame
                            prev_pts[i] = good_new.reshape(-1, 1, 2)
                        else:
                             # No good points found
                             prev_pts[i] = np.array([], dtype=np.float32).reshape(0, 1, 2)
                    else:
                         # Optical flow calculation failed
                         prev_pts[i] = np.array([], dtype=np.float32).reshape(0, 1, 2)
                else:
                    # No points to track
                    prev_pts[i] = np.array([], dtype=np.float32).reshape(0, 1, 2)


                # --- Angle Detection and Tracking ---
                current_raw_detected_angle = None # Angle returned directly by detection function (-180 to 180)
                angle_change_this_frame = 0.0    # Change added to cumulative angle this frame

                # Call the detection function
                detected_angle_result, frame_roi_with_angle_vis = detect_angle(frame_roi.copy())

                if detected_angle_result is not None:
                    # We successfully detected the marker in the current frame
                    current_raw_detected_angle = detected_angle_result # Store the detected angle

                    if last_successful_angle[i] is not None:
                        # We have a previous successful detection to compare against
                        angle_diff = current_raw_detected_angle - last_successful_angle[i]

                        # Normalize the difference (e.g., 10 -> -350 becomes 10)
                        angle_diff = normalize_angle_diff(angle_diff)

                        # Only add the change to cumulative if motion was detected
                        # This prevents angle drift when the wheel is stationary but mark wiggles
                        if motion_detected:
                             angle_change_this_frame = angle_diff

                    # Update the last successful angle tracker *regardless* of whether motion was detected
                    # We always want to know the latest valid angle measurement.
                    last_successful_angle[i] = current_raw_detected_angle

                # Add the calculated change (which might be 0) to the cumulative angle
                cumulative_angles[i] += angle_change_this_frame


                # --- Store data for histories and final plots ---
                motion_histories[i].append(avg_magnitude)
                all_magnitudes_per_roi[i].append(avg_magnitude)

                # Store the raw detected angle (None if not found)
                raw_angle_histories[i].append(current_raw_detected_angle) # For live plot (if implemented)
                all_raw_angles_per_roi[i].append(current_raw_detected_angle) # For final plot

                # Store the cumulative angle
                cumulative_angle_histories[i].append(cumulative_angles[i]) # For live plot (if implemented)
                all_cumulative_angles_per_roi[i].append(cumulative_angles[i])


                # --- Logging ---
                # Log motion status change
                if motion_detected != previous_motion_statuses[i]:
                    event = "Motion Started" if motion_detected else "Motion Stopped"
                    log_message = f"{current_timestamp_sec:.3f},{i},{event},{avg_magnitude:.4f}\n"
                    print(f"ROI {i}: {log_message.strip()}")
                    log_file.write(log_message)
                    log_file.flush()
                previous_motion_statuses[i] = motion_detected

                # Log angle change *only if* a change was actually calculated and added
                if abs(angle_change_this_frame) > 0:
                    # Log timestamp, ROI, the raw detected angle, the change, and the new cumulative total
                    # Convert raw angle to 0-360 for logging if desired, or keep -180-180
                    log_detected_angle = (current_raw_detected_angle + 360) % 360 if current_raw_detected_angle is not None else 'None'
                    angle_log_message = f"{current_timestamp_sec:.3f},{i},{log_detected_angle},{angle_change_this_frame:.2f},{cumulative_angles[i]:.2f}\n"
                    angle_log_file.write(angle_log_message)
                    angle_log_file.flush()


                # --- Update prev gray for optical flow ---
                prev_grays[i] = gray.copy()

                # --- Draw ROI on video frame part ---
                status_color = (0, 255, 0) if motion_detected else (0, 0, 255)
                cv2.rectangle(frame_with_rois, (x, y), (x + w, y + h), status_color, 2)

                # Overlay the ROI with angle visualization onto the main frame copy
                frame_with_rois[y:y+h, x:x+w] = frame_roi_with_angle_vis

                # Put text with ROI number
                cv2.putText(frame_with_rois, f"ROI {i}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1, cv2.LINE_AA)

                # Add current raw detected angle (converted to 0-360 for display)
                display_raw_angle = (current_raw_detected_angle + 360) % 360 if current_raw_detected_angle is not None else 'N/A'
                cv2.putText(frame_with_rois, f"Raw: {display_raw_angle}°",
                            (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)

                # Add cumulative angle
                cv2.putText(frame_with_rois, f"Total: {cumulative_angles[i]:.1f}°",
                            (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 165, 0), 1, cv2.LINE_AA)


            # --- End ROI Processing Loop ---

            # --- Place the frame with ROIs onto the combined canvas ---
            combined_window[0:frame_height, 0:frame_width] = frame_with_rois


            # --- Draw Live Graphs onto Canvas (Motion Magnitude) ---
            if graph_height_per_roi > 0:
                 for i in range(NUM_ROIS):
                     graph_x_start = frame_width + LIVE_GRAPH_PADDING // 2
                     graph_y_start = LIVE_GRAPH_PADDING + i * (graph_height_per_roi + LIVE_GRAPH_PADDING)
                     draw_live_graph(
                         combined_window,
                         motion_histories[i],
                         i,
                         previous_motion_statuses[i],
                         graph_x_start,
                         graph_y_start,
                         LIVE_GRAPH_WIDTH_PER_ROI,
                         graph_height_per_roi,
                         LIVE_GRAPH_MAX_MAGNITUDE,
                         MOTION_THRESHOLD
                     )
            else:
                 cv2.putText(combined_window, "Graphs disabled", (frame_width + 10, 30),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)


            # --- Add general info text ---
            processing_fps = 1.0 / (time.time() - loop_start_time) if (time.time() - loop_start_time) > 0 else 0
            cv2.putText(combined_window, f"Processing FPS: {processing_fps:.1f}",
                        (10, current_combined_height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(combined_window, f"Video Time: {current_timestamp_sec:.2f}s",
                        (10, current_combined_height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)


            # --- Display Combined Window ---
            cv2.imshow('Multi-ROI Motion & Angle Detection', combined_window)

            # --- Wait Key Logic ---
            proc_time_ms = (time.time() - loop_start_time) * 1000
            # Calculate time to wait to match original FPS, accounting for processing time
            actual_wait_time = max(1, wait_time_ms - int(proc_time_ms))
            key = cv2.waitKey(actual_wait_time)
            if key != -1 and key & 0xFF == ord('q'):
                print("Exit requested by user.")
                break

except IOError as e:
    print(f"Error accessing files: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    import traceback
    traceback.print_exc() # Print full traceback for unexpected errors

finally:
    # --- Cleanup OpenCV ---
    print("Releasing video capture...")
    cap.release()
    print("Destroying OpenCV windows...")
    cv2.destroyAllWindows()

    # --- Generate and Save FINAL Matplotlib Graphs ---
    print("Generating final plots...")
    try:
        # Motion Graph
        if all_timestamps and any(len(roi_list) > 0 for roi_list in all_magnitudes_per_roi.values()):
            fig, axes = plt.subplots(NUM_ROIS, 1, figsize=(15, 4 * NUM_ROIS), sharex=True)
            if NUM_ROIS == 1:
                axes = [axes]

            # Determine Y-axis limit
            effective_ylim_top = PLOT_YLIM_TOP
            if effective_ylim_top is None:
                 max_mag_overall = 0
                 for i in range(NUM_ROIS):
                    if i in all_magnitudes_per_roi and all_magnitudes_per_roi[i]:
                        max_mag_overall = max(max_mag_overall, max(all_magnitudes_per_roi[i]))
                 effective_ylim_top = max(max_mag_overall * 1.1, MOTION_THRESHOLD * 1.5) if max_mag_overall > 0 else MOTION_THRESHOLD * 1.5
                 effective_ylim_top = max(effective_ylim_top, 1.0) # Ensure a minimum y-range


            for i in range(NUM_ROIS):
                ax = axes[i]
                # Plot motion magnitude history
                if i in all_magnitudes_per_roi and all_magnitudes_per_roi[i]:
                    ax.plot(all_timestamps, all_magnitudes_per_roi[i], label=f'ROI {i} Avg Magnitude', linewidth=1)
                else:
                    ax.text(0.5, 0.5, 'No data for this ROI', horizontalalignment='center',
                            verticalalignment='center', transform=ax.transAxes)

                # Draw motion threshold
                ax.axhline(MOTION_THRESHOLD, color='r', linestyle='--', label=f'Threshold={MOTION_THRESHOLD}')

                ax.set_ylabel("Avg Motion Magnitude (px)")
                plot_title = f"ROI {i} Motion Magnitude"
                if PLOT_YLIM_TOP is not None:
                   plot_title += f" [Y-Axis Clipped at {effective_ylim_top:.1f}]"
                ax.set_title(plot_title)
                ax.legend(loc='upper right')
                ax.grid(True)
                ax.set_ylim(bottom=-0.1, top=effective_ylim_top) # Ensure bottom is slightly below 0

            axes[-1].set_xlabel("Time (s)")
            plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout to make room for suptitle if added
            plt.savefig(FINAL_GRAPH_FILENAME, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Final motion summary graph saved to {FINAL_GRAPH_FILENAME}")
        else:
            print("Warning: No motion data collected for any ROI, cannot generate final motion graph.")

        # Angle Graph (Cumulative and Raw)
        if all_timestamps and any(len(roi_list) > 0 for roi_list in all_cumulative_angles_per_roi.values()):
            fig, axes = plt.subplots(NUM_ROIS, 1, figsize=(15, 5 * NUM_ROIS), sharex=True) # Make height slightly larger
            if NUM_ROIS == 1:
                axes = [axes]

            for i in range(NUM_ROIS):
                ax = axes[i]
                # Plot Cumulative Angle
                if i in all_cumulative_angles_per_roi and all_cumulative_angles_per_roi[i]:
                    ax.plot(all_timestamps, all_cumulative_angles_per_roi[i],
                            label=f'ROI {i} Cumulative Angle', linewidth=1.5, color='blue')

                # Plot Raw Detected Angle on a secondary axis
                if i in all_raw_angles_per_roi and all_raw_angles_per_roi[i]:
                    # Filter out None values for plotting raw angles
                    valid_raw_angles_idx = [j for j, val in enumerate(all_raw_angles_per_roi[i]) if val is not None]
                    if valid_raw_angles_idx:
                        valid_timestamps_raw = [all_timestamps[j] for j in valid_raw_angles_idx]
                        valid_raw_angles = [(all_raw_angles_per_roi[i][j] + 360) % 360 for j in valid_raw_angles_idx] # Convert to 0-360 for plotting

                        ax2 = ax.twinx() # Create a secondary y-axis
                        # Use markers only if data is sparse, otherwise line is better
                        marker_style = 'o' if len(valid_raw_angles) < 100 else None
                        ax2.plot(valid_timestamps_raw, valid_raw_angles,
                                 label=f'ROI {i} Raw Detected Angle (0-360°)',
                                 linewidth=0.5, color='gray', alpha=0.5, linestyle='None', marker=marker_style, markersize=3) # Plot as points
                        ax2.set_ylabel("Raw Angle (°)", color='gray')
                        ax2.tick_params(axis='y', labelcolor='gray')
                        ax2.set_ylim(-10, 370) # Slightly outside 0-360 for clarity
                        ax2.legend(loc='upper right')


                else:
                    ax.text(0.5, 0.5, 'No angle data for this ROI', horizontalalignment='center',
                            verticalalignment='center', transform=ax.transAxes)


                ax.set_ylabel("Cumulative Rotation (°)")
                ax.set_title(f"ROI {i} Angle Tracking")
                ax.legend(loc='upper left')
                ax.grid(True)

            axes[-1].set_xlabel("Time (s)")
            plt.tight_layout(rect=[0, 0.03, 1, 0.98])
            plt.savefig(ANGLE_GRAPH_FILENAME, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Angle tracking graph saved to {ANGLE_GRAPH_FILENAME}")
        else:
            print("Warning: No angle data collected for any ROI, cannot generate angle graph.")

    except Exception as e:
        print(f"An error occurred while generating/saving the graphs: {e}")
        import traceback
        traceback.print_exc()

    print(f"Motion log file saved to {LOG_FILENAME}")
    print(f"Angle log file saved to {ANGLE_LOG_FILENAME}")
    print("Script finished.")