import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import time
import collections
from scipy.signal import savgol_filter

# --- General Configuration ---
OUTPUT_FOLDER = "DualStreamAnalysis"
LOG_FILENAME_BASE = "combined_motion_log.csv"
FINAL_GRAPH_FILENAME_BASE = "combined_motion_summary.png"
SHOW_VIDEO_OUTPUT = True
GRAPH_HISTORY_LENGTH = 150  # Number of data points for live graph

# Display layout: Two video panes on top, then two rows of 4 graphs each below.
VIDEO_PANE_W = 960  # Width for each displayed video pane
VIDEO_PANE_H = 540  # Height for each displayed video pane

# For live graphs (8 total, arranged in columns with V1 on top of corresponding V2)
NUM_GRAPH_COLUMNS = 4 # Corresponds to number of ROIs per video stream
NUM_GRAPH_ROWS_PER_COLUMN = 2 # V1 graph, then V2 graph
LIVE_GRAPH_PADDING = 10
LIVE_GRAPH_MARGIN = 5

TOTAL_GRAPH_AREA_W = 2 * VIDEO_PANE_W
INDIVIDUAL_GRAPH_W = (TOTAL_GRAPH_AREA_W // NUM_GRAPH_COLUMNS) - LIVE_GRAPH_PADDING
INDIVIDUAL_GRAPH_H = (VIDEO_PANE_H // NUM_GRAPH_ROWS_PER_COLUMN) - LIVE_GRAPH_PADDING

MAX_DISPLAY_W = 1920
MAX_DISPLAY_H = 1080

# --- Interactive Controls Configuration ---
KEY_SPACEBAR = 32
KEY_Q = ord('q')
# Arrow key codes from cv2.waitKeyEx() can vary.
# Common values:
# Windows: LEFT=2424832, RIGHT=2555904
# Linux (GTK backend): LEFT=65361, RIGHT=65363
# macOS (Cocoa backend): LEFT=63234, RIGHT=63235 (or f_keys if VT100, e.g. Left: 27, 91, 68 -> ESC [ D)
# Using common integer values, will print a warning if they don't match
KEY_LEFT_ARROW_OPTIONS = [2424832, 65361, 63234, ord('j')] # Add ord('j') as a fallback
KEY_RIGHT_ARROW_OPTIONS = [2555904, 65363, 63235, ord('l')]# Add ord('l') as a fallback
KEY_LEFT_ARROW = -1 # Will be determined at runtime
KEY_RIGHT_ARROW = -1 # Will be determined at runtime

MAX_FPS_FAST_MODE = 25.0 # Max FPS when holding arrow keys for fast fwd/bwd

# --- Create Output Folder ---
try:
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"Output will be saved in folder: '{OUTPUT_FOLDER}'")
except OSError as e:
    print(f"Error creating directory {OUTPUT_FOLDER}: {e}"); exit()

LOG_FILENAME = os.path.join(OUTPUT_FOLDER, LOG_FILENAME_BASE)
FINAL_GRAPH_FILENAME = os.path.join(OUTPUT_FOLDER, FINAL_GRAPH_FILENAME_BASE)

# --- Configuration for Video 1 (Adapted from front.py) ---
VIDEO_SOURCE_1 = "fronti.mp4"
ROI_COORDS_1 = [
    [470, 432, 125, 120], [600, 432, 125, 120],
    [765, 432, 127, 120], [900, 432, 127, 120]
]
NUM_ROIS_1 = len(ROI_COORDS_1)
if NUM_ROIS_1 != NUM_GRAPH_COLUMNS:
    print(f"Warning: NUM_ROIS_1 ({NUM_ROIS_1}) does not match NUM_GRAPH_COLUMNS ({NUM_GRAPH_COLUMNS}).")
MOTION_THRESHOLD_1 = 1.0
FEATURE_PARAMS_1 = dict(maxCorners=50, qualityLevel=0.3, minDistance=7, blockSize=7)
LK_PARAMS_1 = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
REDETECT_INTERVAL_1 = 30
LIVE_GRAPH_MAX_MAGNITUDE_1 = 5.0
PLOT_YLIM_TOP_1 = 5.0
VIDEO_1_NAME = "FrontView"

# --- Configuration for Video 2 (Adapted from back.py) ---
ASSETS_DIR_2 = Path("./assets")
CLIPS_DIR_2 = ASSETS_DIR_2 / "clips"
VIDEO_FILENAME_2 = "output_video2_synced.mp4"
VIDEO_SOURCE_2 = str(CLIPS_DIR_2 / VIDEO_FILENAME_2)
ROI_COORDS_2 = [
    (954, 534, 156, 160), (1143, 510, 174, 187),
    (1363, 543, 182, 176), (1605, 532, 129, 168),
]
NUM_ROIS_2 = len(ROI_COORDS_2)
if NUM_ROIS_2 != NUM_GRAPH_COLUMNS:
    print(f"Warning: NUM_ROIS_2 ({NUM_ROIS_2}) does not match NUM_GRAPH_COLUMNS ({NUM_GRAPH_COLUMNS}).")
CHANGE_THRESHOLD_2 = 3.0
PERSISTENCE_FRAMES_2 = 1
LIVE_GRAPH_MAX_MAGNITUDE_2 = 12.0
PLOT_YLIM_TOP_2 = 12.0
VIDEO_2_NAME = "BackView"
V2_ROI_MAPPING_FOR_DISPLAY = [3, 2, 1, 0]

def get_timestamp_from_video_seconds(cap, start_time_sec=0.0, frame_num=0, fps_fallback=30):
    # This function is called AFTER cap.set(POS_FRAMES, frame_num) and BEFORE cap.read()
    # So, POS_MSEC should correspond to 'frame_num'
    msec = cap.get(cv2.CAP_PROP_POS_MSEC)
    if msec > 0: # POS_MSEC is often more accurate if available
        return msec / 1000.0
    else: # Fallback to frame_num / fps
        current_fps = cap.get(cv2.CAP_PROP_FPS)
        if not current_fps or current_fps <= 0: current_fps = fps_fallback
        if current_fps > 0: return float(frame_num) / current_fps
        else: # Absolute fallback if FPS is unknown (should not happen with video files)
            return time.time() - start_time_sec # This is not ideal for seeking

def draw_live_graph(canvas, history, roi_label, motion_status_text, x_offset, y_offset, width, height, max_mag, threshold):
    graph_x = x_offset + LIVE_GRAPH_MARGIN
    graph_y = y_offset + LIVE_GRAPH_MARGIN
    graph_w = width - 2 * LIVE_GRAPH_MARGIN
    graph_h = height - 2 * LIVE_GRAPH_MARGIN

    if graph_w <= 0 or graph_h <= 0: return
    cv2.rectangle(canvas, (x_offset, y_offset), (x_offset + width, y_offset + height), (40, 40, 40), -1)

    if max_mag > 0:
        thresh_line_y = graph_y + graph_h - int(np.clip(threshold / max_mag, 0, 1) * graph_h)
        cv2.line(canvas, (graph_x, thresh_line_y), (graph_x + graph_w, thresh_line_y), (0, 100, 100), 1)

    points = []
    for i, mag_val in enumerate(history):
        y_coord = graph_y + graph_h - int(np.clip(mag_val / max_mag, 0, 1) * graph_h)
        x_coord = graph_x + int((i / (GRAPH_HISTORY_LENGTH - 1 if GRAPH_HISTORY_LENGTH > 1 else 1)) * graph_w)
        points.append((x_coord, y_coord))

    if len(points) > 1: cv2.polylines(canvas, [np.array(points)], isClosed=False, color=(100, 255, 100), thickness=1)
    if points:
        marker_color = (0, 255, 0) if motion_status_text == 'MOVING' else (0, 0, 255)
        cv2.circle(canvas, points[-1], 3, marker_color, -1)

    cv2.putText(canvas, f"{roi_label}: {motion_status_text}", (x_offset + 5, y_offset + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

def apply_savgol_filter(signal, window_length=11, polyorder=2):
    if len(signal) < window_length: return signal
    wl = min(window_length, len(signal))
    if wl % 2 == 0: wl -= 1
    if wl < 3 : return signal
    return savgol_filter(signal, window_length=wl, polyorder=polyorder)

def process_frame_video1(frame, frame_timestamp_sec, config, state, frame_idx_1_based): # Renamed frame_count_v1
    new_events = []
    if frame is None: return state, new_events, [0.0] * config['NUM_ROIS'], ['STOPPED'] * config['NUM_ROIS']

    current_roi_magnitudes = [0.0] * config['NUM_ROIS']
    current_motion_statuses = [False] * config['NUM_ROIS']

    for i in range(config['NUM_ROIS']):
        x, y, w, h = config['ROI_COORDS'][i]
        fh, fw = frame.shape[:2]
        if not (0 <= y < fh and 0 <= x < fw and y+h <= fh and x+w <= fw and w > 0 and h > 0) :
            current_roi_magnitudes[i] = 0.0
            current_motion_statuses[i] = False
            continue

        frame_roi = frame[y:y+h, x:x+w]
        gray_roi = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)

        p0 = state['prev_pts'][i]
        prev_gray_roi = state['prev_grays'][i]

        if p0 is None or len(p0) < 5 or frame_idx_1_based % config['REDETECT_INTERVAL'] == 0 or prev_gray_roi is None:
            p0 = cv2.goodFeaturesToTrack(prev_gray_roi if prev_gray_roi is not None else gray_roi, mask=None, **config['FEATURE_PARAMS'])
            if p0 is None: p0 = np.array([], dtype=np.float32).reshape(0, 1, 2)
            state['prev_pts'][i] = p0

        avg_magnitude = 0.0
        motion_detected_roi = False
        if p0 is not None and len(p0) > 0 and prev_gray_roi is not None:
            if prev_gray_roi.shape != gray_roi.shape:
                state['prev_grays'][i] = gray_roi.copy()
                state['prev_pts'][i] = cv2.goodFeaturesToTrack(gray_roi, mask=None, **config['FEATURE_PARAMS'])
                if state['prev_pts'][i] is None: state['prev_pts'][i] = np.array([], dtype=np.float32).reshape(0,1,2)
                p0 = state['prev_pts'][i]

            if p0 is not None and len(p0)>0:
                p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray_roi, gray_roi, p0, None, **config['LK_PARAMS'])
                if p1 is not None and st is not None:
                    good_new = p1[st == 1]
                    good_old = p0[st == 1]
                    if len(good_new) > 0 and len(good_old) > 0:
                        displacement = good_new - good_old
                        magnitudes = np.linalg.norm(displacement, axis=1)
                        avg_magnitude = np.mean(magnitudes)
                        if avg_magnitude > config['MOTION_THRESHOLD']:
                            motion_detected_roi = True
                        state['prev_pts'][i] = good_new.reshape(-1, 1, 2)
                    else:
                        state['prev_pts'][i] = np.array([], dtype=np.float32).reshape(0, 1, 2)
                else:
                    state['prev_pts'][i] = np.array([], dtype=np.float32).reshape(0, 1, 2)

        current_roi_magnitudes[i] = avg_magnitude
        current_motion_statuses[i] = motion_detected_roi

        if motion_detected_roi != state['previous_motion_statuses'][i]:
            event_type = 'START' if motion_detected_roi else 'STOP'
            duration_sec = None
            if event_type == 'START':
                state['roi_start_times_sec'][i] = frame_timestamp_sec
            elif state['roi_start_times_sec'][i] is not None:
                duration_sec = frame_timestamp_sec - state['roi_start_times_sec'][i]
                state['roi_start_times_sec'][i] = None

            event = (i, event_type, frame_timestamp_sec, duration_sec, avg_magnitude)
            new_events.append(event)
            state['previous_motion_statuses'][i] = motion_detected_roi

        state['prev_grays'][i] = gray_roi.copy()

    status_texts = ['MOVING' if status else 'STOPPED' for status in current_motion_statuses]
    return state, new_events, current_roi_magnitudes, status_texts

def process_frame_video2(frame, frame_timestamp_sec, config, state):
    new_events = []
    if frame is None:
        return state, new_events, [0.0] * config['NUM_ROIS'], ['STOPPED'] * config['NUM_ROIS']

    roi_coords_list = config['ROI_COORDS']
    change_threshold_val = config['CHANGE_THRESHOLD']
    persistence_frames_val = config['PERSISTENCE_FRAMES']
    fps = config.get('FPS', 30)

    previous_gray_rois = state['previous_gray_rois']
    roi_states_text = state['roi_states_text']
    roi_start_times_sec = state['roi_start_times_sec']
    roi_change_counters = state['roi_change_counters']
    roi_static_counters = state['roi_static_counters']
    roi_change_history = state.get('roi_change_history', [collections.deque(maxlen=30) for _ in range(len(roi_coords_list))])

    current_roi_change_scores = [0.0] * len(roi_coords_list)

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    for i, (x, y, w, h) in enumerate(roi_coords_list):
        h_frame, w_frame = gray_frame.shape
        y_end, x_end = min(y + h, h_frame), min(x + w, w_frame)
        if y >= y_end or x >= x_end or w <=0 or h <=0 :
             current_roi_change_scores[i] = 0.0
             continue

        current_roi_gray = gray_frame[y:y_end, x:x_end]

        if previous_gray_rois[i] is None or previous_gray_rois[i].shape != current_roi_gray.shape:
            previous_gray_rois[i] = current_roi_gray # Store copy for next frame
            current_roi_change_scores[i] = 0.0
            roi_change_history[i].clear() # Clear history if ROI shape changes or first time
            roi_change_history[i].append(0.0) # Add a zero to start
            continue

        diff_roi = cv2.absdiff(previous_gray_rois[i], current_roi_gray)
        change_score = np.mean(diff_roi)

        roi_change_history[i].append(change_score)
        filtered_score = change_score
        if len(roi_change_history[i]) >= 5: # Need enough points for filter
            # Ensure window_length is odd and <= number of points
            wl = min(11, len(roi_change_history[i]))
            if wl % 2 == 0: wl -=1
            if wl >=3: # Savgol filter requires window_length >= polyorder + 1, and polyorder must be < window_length
                 filtered_values = apply_savgol_filter(list(roi_change_history[i]), wl, 2)
                 if filtered_values is not None and len(filtered_values) > 0:
                    filtered_score = filtered_values[-1]
                 else: # Fallback if filter returns empty or None
                    filtered_score = change_score
            else: # Not enough points for meaningful filter after adjustment
                filtered_score = change_score

        current_roi_change_scores[i] = filtered_score

        if filtered_score > change_threshold_val:
            roi_change_counters[i] += 1
            roi_static_counters[i] = 0
        else:
            roi_static_counters[i] += 1
            roi_change_counters[i] = 0

        current_roi_state_text = roi_states_text[i]
        if current_roi_state_text == 'STOPPED' and roi_change_counters[i] >= persistence_frames_val:
            roi_states_text[i] = 'MOVING'
            time_offset_sec = (persistence_frames_val - 1) / fps if fps > 0 else 0
            start_time_sec = frame_timestamp_sec - time_offset_sec
            roi_start_times_sec[i] = start_time_sec
            event = (i, 'START', start_time_sec, None, filtered_score)
            new_events.append(event)
            roi_change_counters[i] = 0 # Reset counter after triggering
        elif current_roi_state_text == 'MOVING' and roi_static_counters[i] >= persistence_frames_val:
            roi_states_text[i] = 'STOPPED'
            time_offset_sec = (persistence_frames_val - 1) / fps if fps > 0 else 0
            stop_time_sec = frame_timestamp_sec - time_offset_sec
            duration_sec = None
            if roi_start_times_sec[i] is not None:
                duration_sec = stop_time_sec - roi_start_times_sec[i]
            roi_start_times_sec[i] = None
            event = (i, 'STOP', stop_time_sec, duration_sec, filtered_score)
            new_events.append(event)
            roi_static_counters[i] = 0 # Reset counter

        previous_gray_rois[i] = current_roi_gray.copy()

    state['previous_gray_rois'] = previous_gray_rois
    state['roi_states_text'] = roi_states_text
    state['roi_start_times_sec'] = roi_start_times_sec
    state['roi_change_counters'] = roi_change_counters
    state['roi_static_counters'] = roi_static_counters
    state['roi_change_history'] = roi_change_history

    return state, new_events, current_roi_change_scores, roi_states_text

def get_initial_state1(num_rois, frame1_init_for_roi_shape=None, roi_coords_list=None):
    state = {
        'prev_grays': [None] * num_rois,
        'prev_pts': [np.array([], dtype=np.float32).reshape(0, 1, 2)] * num_rois,
        'previous_motion_statuses': [False] * num_rois,
        'roi_start_times_sec': [None] * num_rois
    }
    if frame1_init_for_roi_shape is not None and roi_coords_list is not None:
        fh_init, fw_init = frame1_init_for_roi_shape.shape[:2]
        for i_init in range(num_rois):
            x_init, y_init, w_init, h_init = roi_coords_list[i_init]
            if not (0 <= y_init < fh_init and 0 <= x_init < fw_init and \
                    y_init+h_init <= fh_init and x_init+w_init <= fw_init and \
                    w_init > 0 and h_init > 0):
                continue
            state['prev_grays'][i_init] = cv2.cvtColor(
                frame1_init_for_roi_shape[y_init:y_init+h_init, x_init:x_init+w_init],
                cv2.COLOR_BGR2GRAY
            )
    return state

def get_initial_state2(num_rois):
    return {
        'previous_gray_rois': [None] * num_rois,
        'roi_states_text': ['STOPPED'] * num_rois,
        'roi_start_times_sec': [None] * num_rois,
        'roi_change_counters': [0] * num_rois,
        'roi_static_counters': [0] * num_rois,
        'roi_change_history': [collections.deque(maxlen=30) for _ in range(num_rois)]
    }

# --- Main Execution ---
if __name__ == "__main__":
    cap1 = cv2.VideoCapture(VIDEO_SOURCE_1)
    if not cap1.isOpened(): print(f"ERROR: Could not open video {VIDEO_SOURCE_1}"); exit()
    original_fps1 = cap1.get(cv2.CAP_PROP_FPS)
    total_frames1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    if original_fps1 <= 0: original_fps1 = 30
    config1 = {
        'ROI_COORDS': ROI_COORDS_1, 'NUM_ROIS': NUM_ROIS_1,
        'MOTION_THRESHOLD': MOTION_THRESHOLD_1,
        'FEATURE_PARAMS': FEATURE_PARAMS_1, 'LK_PARAMS': LK_PARAMS_1,
        'REDETECT_INTERVAL': REDETECT_INTERVAL_1, 'FPS': original_fps1
    }
    ret1_init, frame1_init = cap1.read()
    if not ret1_init: print(f"ERROR: Could not read initial frame from {VIDEO_SOURCE_1}"); cap1.release(); exit()
    state1 = get_initial_state1(NUM_ROIS_1, frame1_init, ROI_COORDS_1)
    cap1.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reset to start

    cap2 = cv2.VideoCapture(VIDEO_SOURCE_2)
    if not cap2.isOpened(): print(f"ERROR: Could not open video {VIDEO_SOURCE_2}"); cap1.release(); exit()
    original_fps2 = cap2.get(cv2.CAP_PROP_FPS)
    total_frames2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    if original_fps2 <= 0: original_fps2 = 30
    config2 = {
        'ROI_COORDS': ROI_COORDS_2, 'NUM_ROIS': NUM_ROIS_2,
        'CHANGE_THRESHOLD': CHANGE_THRESHOLD_2,
        'PERSISTENCE_FRAMES': PERSISTENCE_FRAMES_2, 'FPS': original_fps2
    }
    state2 = get_initial_state2(NUM_ROIS_2)
    # No need to read initial frame for state2 as it initializes ROIs on first valid process call

    max_total_frames = min(total_frames1, total_frames2)
    if max_total_frames <= 0: print("Error: Video has no frames or failed to get frame count."); exit()

    motion_histories1 = {i: collections.deque(np.zeros(GRAPH_HISTORY_LENGTH), maxlen=GRAPH_HISTORY_LENGTH) for i in range(NUM_ROIS_1)}
    motion_histories2 = {i: collections.deque(np.zeros(GRAPH_HISTORY_LENGTH), maxlen=GRAPH_HISTORY_LENGTH) for i in range(NUM_ROIS_2)}

    all_timestamps_global = []
    all_scores1_per_roi = {i: [] for i in range(NUM_ROIS_1)}
    all_scores2_per_roi = {i: [] for i in range(NUM_ROIS_2)}

    top_row_h = VIDEO_PANE_H
    graph_section_h_total = NUM_GRAPH_ROWS_PER_COLUMN * (INDIVIDUAL_GRAPH_H + LIVE_GRAPH_PADDING) + LIVE_GRAPH_PADDING
    combined_canvas_w = 2 * VIDEO_PANE_W
    combined_canvas_h = top_row_h + graph_section_h_total

    current_frame_idx = 0
    paused = False
    system_start_time_sec = time.time() # For fallback timestamp, less relevant now
    cached_combined_canvas = np.zeros((combined_canvas_h, combined_canvas_w, 3), dtype=np.uint8)
    
    # Attempt to determine actual arrow key codes
    print("INFO: Press Left/Right arrow keys in the OpenCV window once to confirm detection if they don't work initially.")
    print(f"INFO: Fallback keys: 'j' (left), 'l' (right). Space to pause/play. 'q' to quit.")

    if SHOW_VIDEO_OUTPUT:
        cv2.namedWindow('Dual Stream Motion Analysis', cv2.WINDOW_NORMAL)
        # Try to get one key press to see if it's an arrow for initial setup
        temp_key_prompt = np.zeros((100,400,3), dtype=np.uint8)
        cv2.putText(temp_key_prompt, "Press Left or Right Arrow", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)
        cv2.putText(temp_key_prompt, " (or j/l) then any other key", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)
        cv2.imshow('Dual Stream Motion Analysis', temp_key_prompt)
        
        # First key try
        key_detect_1 = cv2.waitKeyEx(0)
        if key_detect_1 in KEY_LEFT_ARROW_OPTIONS: KEY_LEFT_ARROW = key_detect_1
        elif key_detect_1 in KEY_RIGHT_ARROW_OPTIONS: KEY_RIGHT_ARROW = key_detect_1
        
        # Second key try (if first wasn't definitive for both)
        if KEY_LEFT_ARROW == -1 or KEY_RIGHT_ARROW == -1:
            key_detect_2 = cv2.waitKeyEx(0)
            if KEY_LEFT_ARROW == -1 and key_detect_2 in KEY_LEFT_ARROW_OPTIONS: KEY_LEFT_ARROW = key_detect_2
            if KEY_RIGHT_ARROW == -1 and key_detect_2 in KEY_RIGHT_ARROW_OPTIONS: KEY_RIGHT_ARROW = key_detect_2

        # Fallbacks if specific arrow keys weren't pressed/detected
        if KEY_LEFT_ARROW == -1: KEY_LEFT_ARROW = ord('j'); print("WARN: Left Arrow not detected, using 'j'.")
        else: print(f"INFO: Detected Left Arrow Key Code: {KEY_LEFT_ARROW}")
        if KEY_RIGHT_ARROW == -1: KEY_RIGHT_ARROW = ord('l'); print("WARN: Right Arrow not detected, using 'l'.")
        else: print(f"INFO: Detected Right Arrow Key Code: {KEY_RIGHT_ARROW}")


    try:
        with open(LOG_FILENAME, 'w') as log_file:
            log_file.write("Timestamp(s),VideoName,ROI_Index,Event,Duration(s),ValueAtEvent\n")
            print(f"Logging events to: {LOG_FILENAME}")

            last_frame_processing_time = 0.0

            while True:
                loop_iter_start_time = time.time()
                
                # --- Key Input ---
                target_fps_for_wait = max(original_fps1, original_fps2)
                if not paused and (KEY_LEFT_ARROW != -1 or KEY_RIGHT_ARROW != -1) : # If arrows are active, assume fast mode intent
                    # Check if an arrow key was pressed in the *last* cycle to enable fast mode this cycle
                    # This requires knowing the key from *last* iteration. For now, simplify:
                    # If playing and an arrow key is pressed *now*, use fast FPS.
                    # This will be handled by `key` variable obtained below.
                    pass # Will adjust target_fps_for_wait based on key pressed

                effective_wait_ms = 1
                if paused:
                    effective_wait_ms = 0 # Wait indefinitely
                else: # Playing or fast mode
                    # Calculate delay to approximate target_fps
                    desired_interval_ms = 1000.0 / target_fps_for_wait
                    # last_frame_processing_time is from the previous full frame processing block
                    wait_duration_ms = max(1, int(desired_interval_ms - last_frame_processing_time * 1000))
                    effective_wait_ms = wait_duration_ms
                
                key = cv2.waitKeyEx(effective_wait_ms)

                # --- Determine Action from Key ---
                needs_processing = False
                is_seek_operation = False
                next_target_frame_idx = current_frame_idx

                if key == KEY_Q: break
                if key == KEY_SPACEBAR:
                    paused = not paused
                    if not paused: # Just unpaused
                        needs_processing = True # Process current or next frame
                        next_target_frame_idx = current_frame_idx # Start from current
                    # If paused, just display cached, no processing unless arrow key
                
                # Arrow key logic
                if key == KEY_LEFT_ARROW:
                    target_fps_for_wait = MAX_FPS_FAST_MODE # For next iter if key held
                    if current_frame_idx > 0:
                        next_target_frame_idx = current_frame_idx - 1
                        if not paused: # Fast backward
                            is_seek_operation = True # Treat as seek
                        needs_processing = True
                    if paused: is_seek_operation = True # Step back is a seek
                elif key == KEY_RIGHT_ARROW:
                    target_fps_for_wait = MAX_FPS_FAST_MODE # For next iter if key held
                    if current_frame_idx < max_total_frames - 1:
                        next_target_frame_idx = current_frame_idx + 1
                        if not paused: # Fast forward
                             is_seek_operation = (next_target_frame_idx != current_frame_idx +1) # True if jumped
                        needs_processing = True
                    if paused: is_seek_operation = True # Step forward is a seek
                
                if not paused and not needs_processing: # Normal playback, no specific key action other than time passing
                    if current_frame_idx < max_total_frames:
                        next_target_frame_idx = current_frame_idx # Will process this, then auto-advance
                        needs_processing = True
                    else: # Reached end
                        paused = True 

                if not needs_processing: # Paused and no navigation
                    if cached_combined_canvas is not None and SHOW_VIDEO_OUTPUT:
                         cv2.imshow('Dual Stream Motion Analysis', cached_combined_canvas)
                    continue # Go back to waitKey

                # --- Frame Index and Data History Management ---
                if is_seek_operation or \
                   (paused and (key == KEY_LEFT_ARROW or key == KEY_RIGHT_ARROW)) or \
                   (next_target_frame_idx != current_frame_idx and next_target_frame_idx != current_frame_idx + 1) : # Any non-sequential move
                    
                    # Reset processing states for a clean start on the new frame
                    state1 = get_initial_state1(NUM_ROIS_1, frame1_init, ROI_COORDS_1) # Re-init with first frame for ROIs
                    state2 = get_initial_state2(NUM_ROIS_2)
                    # Clear future data from history lists if jumping back
                    if next_target_frame_idx < len(all_timestamps_global):
                        all_timestamps_global = all_timestamps_global[:next_target_frame_idx]
                        for i_roi in range(NUM_ROIS_1):
                            all_scores1_per_roi[i_roi] = all_scores1_per_roi[i_roi][:next_target_frame_idx]
                        for i_roi in range(NUM_ROIS_2):
                            all_scores2_per_roi[i_roi] = all_scores2_per_roi[i_roi][:next_target_frame_idx]
                    # Live graph deques will naturally adjust as new data comes in.
                    # For a large jump back, could clear and re-fill, but this is simpler.
                    for i_roi in range(NUM_ROIS_1): motion_histories1[i_roi].clear(); motion_histories1[i_roi].extend(np.zeros(GRAPH_HISTORY_LENGTH))
                    for i_roi in range(NUM_ROIS_2): motion_histories2[i_roi].clear(); motion_histories2[i_roi].extend(np.zeros(GRAPH_HISTORY_LENGTH))


                current_frame_idx = next_target_frame_idx
                if current_frame_idx >= max_total_frames:
                    current_frame_idx = max_total_frames - 1
                    paused = True # Auto-pause at the very end
                    if not SHOW_VIDEO_OUTPUT: break # End if not showing video

                frame_process_start_time = time.time()

                # --- Seek and Read ---
                cap1.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
                cap2.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)

                # Get timestamp for current_frame_idx (after seek, before read)
                # Using current_frame_idx directly for frame_num argument
                ts1 = get_timestamp_from_video_seconds(cap1, system_start_time_sec, current_frame_idx, original_fps1)
                ts2 = get_timestamp_from_video_seconds(cap2, system_start_time_sec, current_frame_idx, original_fps2)
                current_ts_global = max(ts1, ts2)

                ret1, frame1_orig = cap1.read()
                ret2, frame2_orig = cap2.read()

                if not ret1 or not ret2:
                    print(f"Error reading frame {current_frame_idx} or end of stream.")
                    paused = True
                    if cached_combined_canvas is not None and SHOW_VIDEO_OUTPUT:
                         cv2.imshow('Dual Stream Motion Analysis', cached_combined_canvas)
                    continue

                # Ensure timestamp monotonicity if appending new
                if current_frame_idx >= len(all_timestamps_global) and \
                   all_timestamps_global and current_ts_global <= all_timestamps_global[-1]:
                    fallback_fps = max(original_fps1, original_fps2, 1) # Avoid div by zero
                    current_ts_global = all_timestamps_global[-1] + (1.0 / fallback_fps)
                
                # --- Process Frames ---
                # Pass current_frame_idx + 1 for 1-based count if function expects it
                state1, events1, scores1, statuses_text1 = process_frame_video1(frame1_orig, current_ts_global, config1, state1, current_frame_idx + 1)
                state2, events2, scores2, statuses_text2 = process_frame_video2(frame2_orig, current_ts_global, config2, state2)

                # --- Store/Update Data ---
                if current_frame_idx < len(all_timestamps_global): # Overwriting data due to seek/reprocess
                    all_timestamps_global[current_frame_idx] = current_ts_global
                    for i in range(NUM_ROIS_1): all_scores1_per_roi[i][current_frame_idx] = scores1[i]
                    for i in range(NUM_ROIS_2): all_scores2_per_roi[i][current_frame_idx] = scores2[i]
                else: # Appending new frame data
                    all_timestamps_global.append(current_ts_global)
                    for i in range(NUM_ROIS_1): all_scores1_per_roi[i].append(scores1[i])
                    for i in range(NUM_ROIS_2): all_scores2_per_roi[i].append(scores2[i])

                for i in range(NUM_ROIS_1): motion_histories1[i].append(scores1[i])
                for i in range(NUM_ROIS_2): motion_histories2[i].append(scores2[i])

                for event_data in events1:
                    roi_idx, ev_type, ts, dur, val = event_data
                    dur_str = f"{dur:.3f}" if dur is not None else "N/A"
                    log_file.write(f"{ts:.3f},{VIDEO_1_NAME},{roi_idx},{ev_type},{dur_str},{val:.4f}\n")
                for event_data in events2:
                    roi_idx, ev_type, ts, dur, val = event_data
                    dur_str = f"{dur:.3f}" if dur is not None else "N/A"
                    log_file.write(f"{ts:.3f},{VIDEO_2_NAME},{roi_idx},{ev_type},{dur_str},{val:.4f}\n")
                log_file.flush()

                last_frame_processing_time = time.time() - frame_process_start_time

                # --- Display Update ---
                if SHOW_VIDEO_OUTPUT:
                    combined_canvas = np.zeros((combined_canvas_h, combined_canvas_w, 3), dtype=np.uint8)
                    
                    frame1_display = cv2.resize(frame1_orig, (VIDEO_PANE_W, VIDEO_PANE_H))
                    for i, (x,y,w,h) in enumerate(config1['ROI_COORDS']):
                        sx, sy = VIDEO_PANE_W/frame1_orig.shape[1], VIDEO_PANE_H/frame1_orig.shape[0]
                        color = (0,255,0) if statuses_text1[i]=='MOVING' else (0,0,255)
                        cv2.rectangle(frame1_display, (int(x*sx), int(y*sy)), (int((x+w)*sx), int((y+h)*sy)), color, 2)
                        cv2.putText(frame1_display, f"V1_R{i}", (int(x*sx), int(y*sy)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,1)
                    combined_canvas[0:VIDEO_PANE_H, 0:VIDEO_PANE_W] = frame1_display

                    frame2_display = cv2.resize(frame2_orig, (VIDEO_PANE_W, VIDEO_PANE_H))
                    for i, (x,y,w,h) in enumerate(config2['ROI_COORDS']):
                        sx, sy = VIDEO_PANE_W/frame2_orig.shape[1], VIDEO_PANE_H/frame2_orig.shape[0]
                        color = (0,255,0) if statuses_text2[i]=='MOVING' else (0,0,255)
                        cv2.rectangle(frame2_display, (int(x*sx), int(y*sy)), (int((x+w)*sx), int((y+h)*sy)), color, 2)
                        cv2.putText(frame2_display, f"V2_R{i}", (int(x*sx), int(y*sy)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,1)
                    combined_canvas[0:VIDEO_PANE_H, VIDEO_PANE_W:2*VIDEO_PANE_W] = frame2_display
                    
                    for col_idx in range(NUM_GRAPH_COLUMNS):
                        v1_roi_idx = col_idx
                        gx = LIVE_GRAPH_PADDING + col_idx * (INDIVIDUAL_GRAPH_W + LIVE_GRAPH_PADDING)
                        gy_top = VIDEO_PANE_H + LIVE_GRAPH_PADDING
                        draw_live_graph(combined_canvas, motion_histories1[v1_roi_idx],
                                        f"V1_R{v1_roi_idx}", statuses_text1[v1_roi_idx],
                                        gx, gy_top, INDIVIDUAL_GRAPH_W, INDIVIDUAL_GRAPH_H,
                                        LIVE_GRAPH_MAX_MAGNITUDE_1, MOTION_THRESHOLD_1)

                        v2_actual_roi_idx_to_display = V2_ROI_MAPPING_FOR_DISPLAY[col_idx]
                        gy_bottom = VIDEO_PANE_H + LIVE_GRAPH_PADDING + (INDIVIDUAL_GRAPH_H + LIVE_GRAPH_PADDING)
                        draw_live_graph(combined_canvas, motion_histories2[v2_actual_roi_idx_to_display],
                                        f"V2_R{v2_actual_roi_idx_to_display}", statuses_text2[v2_actual_roi_idx_to_display],
                                        gx, gy_bottom, INDIVIDUAL_GRAPH_W, INDIVIDUAL_GRAPH_H,
                                        LIVE_GRAPH_MAX_MAGNITUDE_2, CHANGE_THRESHOLD_2)

                    # --- Display Info Text ---
                    total_loop_time = time.time() - loop_iter_start_time
                    proc_fps_text = f"UI FPS: {1.0 / total_loop_time if total_loop_time > 0 else 0:.1f}"
                    time_text = f"Time: {current_ts_global:.2f}s"
                    frame_idx_text = f"Frame: {current_frame_idx}/{max_total_frames-1}"
                    pause_status_text = "PAUSED" if paused else ("PLAYING" if key != KEY_LEFT_ARROW and key != KEY_RIGHT_ARROW else "FAST_MODE")

                    cv2.putText(combined_canvas, proc_fps_text, (10, combined_canvas_h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
                    cv2.putText(combined_canvas, time_text, (10, combined_canvas_h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
                    cv2.putText(combined_canvas, frame_idx_text, (10, combined_canvas_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
                    cv2.putText(combined_canvas, pause_status_text, (combined_canvas_w - 180, combined_canvas_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

                    disp_h, disp_w = combined_canvas.shape[:2]
                    scale_factor = min(MAX_DISPLAY_W / disp_w if disp_w > 0 else 1, MAX_DISPLAY_H / disp_h if disp_h > 0 else 1, 1.0)
                    
                    if scale_factor < 1.0:
                        new_w, new_h = int(disp_w * scale_factor), int(disp_h * scale_factor)
                        if new_w > 0 and new_h > 0:
                            cached_combined_canvas = cv2.resize(combined_canvas, (new_w, new_h), interpolation=cv2.INTER_AREA)
                        else: cached_combined_canvas = combined_canvas.copy() # Fallback
                    else:
                        cached_combined_canvas = combined_canvas.copy()
                    
                    cv2.imshow('Dual Stream Motion Analysis', cached_combined_canvas)
                else: # Not SHOW_VIDEO_OUTPUT
                    if current_frame_idx % 100 == 0: print(f"Processed frame {current_frame_idx}, Time: {current_ts_global:.2f}s")


                # --- Advance Frame Counter for next iteration (if not paused) ---
                if not paused:
                    current_frame_idx += 1 # Auto-advance if playing
                    if current_frame_idx >= max_total_frames:
                        paused = True # Pause at end
                        if not SHOW_VIDEO_OUTPUT: break # End if not showing video

        
    except IOError as e: print(f"Error with log file {LOG_FILENAME}: {e}")
    except Exception as e: print(f"An unexpected error in main loop: {e}"); import traceback; traceback.print_exc()
    finally:
        print("Releasing video captures...")
        cap1.release(); cap2.release()
        if SHOW_VIDEO_OUTPUT: print("Destroying OpenCV windows..."); cv2.destroyAllWindows()
        
        processed_frames_count = len(all_timestamps_global)
        print(f"Analysis complete. Processed up to frame index {current_frame_idx-1 if current_frame_idx > 0 else 0} (Total unique frames in history: {processed_frames_count}).")

        print("Generating final summary graph...")
        try:
            if all_timestamps_global and (any(all_scores1_per_roi[i] for i in range(NUM_ROIS_1)) or \
                                         any(all_scores2_per_roi[i] for i in range(NUM_ROIS_2))):
                
                fig_rows = NUM_GRAPH_COLUMNS
                fig, axes = plt.subplots(fig_rows, 2, figsize=(18, 3 * fig_rows), sharex=True, squeeze=False)

                for plot_row_idx in range(fig_rows):
                    v1_roi_to_plot = plot_row_idx
                    ax1 = axes[plot_row_idx, 0]
                    if v1_roi_to_plot < NUM_ROIS_1 and len(all_scores1_per_roi[v1_roi_to_plot]) == len(all_timestamps_global):
                        ax1.plot(all_timestamps_global, all_scores1_per_roi[v1_roi_to_plot], label=f'{VIDEO_1_NAME} ROI {v1_roi_to_plot} Mag', lw=1)
                    ax1.axhline(MOTION_THRESHOLD_1, color='r', linestyle='--', label=f'Thresh={MOTION_THRESHOLD_1:.1f}')
                    ax1.set_ylabel("Avg Motion Magnitude (px)")
                    ax1.set_title(f"{VIDEO_1_NAME} ROI {v1_roi_to_plot}")
                    ax1.set_ylim(bottom=-0.5, top=PLOT_YLIM_TOP_1)
                    ax1.legend(fontsize='small'); ax1.grid(True)

                    v2_actual_roi_to_plot = V2_ROI_MAPPING_FOR_DISPLAY[plot_row_idx]
                    ax2 = axes[plot_row_idx, 1]
                    if v2_actual_roi_to_plot < NUM_ROIS_2 and len(all_scores2_per_roi[v2_actual_roi_to_plot]) == len(all_timestamps_global):
                        ax2.plot(all_timestamps_global, all_scores2_per_roi[v2_actual_roi_to_plot], label=f'{VIDEO_2_NAME} ROI {v2_actual_roi_to_plot} Score', lw=1, color='g')
                    ax2.axhline(CHANGE_THRESHOLD_2, color='m', linestyle='--', label=f'Thresh={CHANGE_THRESHOLD_2:.1f}')
                    ax2.set_ylabel("Avg Change Score")
                    ax2.set_title(f"{VIDEO_2_NAME} ROI {v2_actual_roi_to_plot}")
                    ax2.set_ylim(bottom=-max(1.0, PLOT_YLIM_TOP_2*0.05), top=PLOT_YLIM_TOP_2)
                    ax2.legend(fontsize='small'); ax2.grid(True)

                if fig_rows > 0:
                     axes[-1, 0].set_xlabel("Time (s)")
                     axes[-1, 1].set_xlabel("Time (s)")
                
                plt.suptitle("Combined Drum Motion Analysis Summary (Mapped ROIs)", fontsize=16)
                plt.tight_layout(rect=[0, 0.03, 1, 0.97])
                plt.savefig(FINAL_GRAPH_FILENAME, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"Final summary graph saved to {FINAL_GRAPH_FILENAME}")
            else:
                print("Warning: No motion data collected or data length mismatch, cannot generate final graph.")
        except Exception as e:
            print(f"Error generating final graph: {e}"); import traceback; traceback.print_exc()
        
        print(f"Log file saved to {LOG_FILENAME}")
        print("Script finished.")