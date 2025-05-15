import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import time
import collections
from scipy.signal import savgol_filter
import csv # Added for lead/lag CSV output

# --- General Configuration ---
OUTPUT_FOLDER = "DualStreamAnalysis"
LOG_FILENAME_BASE = "combined_motion_log.csv"
FINAL_GRAPH_FILENAME_BASE = "combined_motion_summary.png"
LEAD_LAG_FILENAME_BASE = "lead_lag_analysis.csv" # New: For lead/lag analysis
SHOW_VIDEO_OUTPUT = True
GRAPH_HISTORY_LENGTH = 150  # Number of data points for live graph
PLAYBACK_FPS = 25           # FPS for playback of the video streams
MAX_TIME_DIFFERENCE_FOR_MATCH_SEC = 2.0 # New: Max time diff for lead/lag event pairing

# Display layout: Two video panes on top, then two rows of 4 graphs each below.
VIDEO_PANE_W = 960  # Width for each displayed video pane
VIDEO_PANE_H = 540  # Height for each displayed video pane

# For live graphs (8 total, arranged in columns with V1 on top of corresponding V2)
NUM_GRAPH_COLUMNS = 4 # Corresponds to number of ROIs per video stream
NUM_GRAPH_ROWS_PER_COLUMN = 2 # V1 graph, then V2 graph
LIVE_GRAPH_PADDING = 10
LIVE_GRAPH_MARGIN = 5

# Calculated live graph dimensions (per graph)
# Total width for graphs is the width of two video panes.
TOTAL_GRAPH_AREA_W = 2 * VIDEO_PANE_W
INDIVIDUAL_GRAPH_W = (TOTAL_GRAPH_AREA_W // NUM_GRAPH_COLUMNS) - LIVE_GRAPH_PADDING

# Height for each graph row (2 rows of graphs in total height)
# Each graph row takes half the height of a video pane approximately
INDIVIDUAL_GRAPH_H = (VIDEO_PANE_H // NUM_GRAPH_ROWS_PER_COLUMN) - LIVE_GRAPH_PADDING


MAX_DISPLAY_W = 1920 # Max width of the OpenCV window
MAX_DISPLAY_H = 1080 # Max height of the OpenCV window

# --- Create Output Folder ---
try:
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"Output will be saved in folder: '{OUTPUT_FOLDER}'")
except OSError as e:
    print(f"Error creating directory {OUTPUT_FOLDER}: {e}"); exit()

LOG_FILENAME = os.path.join(OUTPUT_FOLDER, LOG_FILENAME_BASE)
FINAL_GRAPH_FILENAME = os.path.join(OUTPUT_FOLDER, FINAL_GRAPH_FILENAME_BASE)
LEAD_LAG_FILENAME = os.path.join(OUTPUT_FOLDER, LEAD_LAG_FILENAME_BASE) # New


# --- Configuration for Video 1 (Adapted from front.py) ---
VIDEO_SOURCE_1 = "output_video1_synced.mp4"
ROI_COORDS_1 = [ # front.py ROIs
    [470, 432, 125, 120],  # V1_ROI0: Cylinder 1
    [600, 432, 125, 120],  # V1_ROI1: Cylinder 2
    [765, 432, 127, 120],  # V1_ROI2: Cylinder 3
    [900, 432, 127, 120]   # V1_ROI3: Cylinder 4
]
NUM_ROIS_1 = len(ROI_COORDS_1)
if NUM_ROIS_1 != NUM_GRAPH_COLUMNS:
    print(f"Warning: NUM_ROIS_1 ({NUM_ROIS_1}) does not match NUM_GRAPH_COLUMNS ({NUM_GRAPH_COLUMNS}). Layout might be affected.")
MOTION_THRESHOLD_1 = 1.0
FEATURE_PARAMS_1 = dict(maxCorners=50, qualityLevel=0.3, minDistance=7, blockSize=7)
LK_PARAMS_1 = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
REDETECT_INTERVAL_1 = 30
LIVE_GRAPH_MAX_MAGNITUDE_1 = 5.0
PLOT_YLIM_TOP_1 = 5.0
VIDEO_1_NAME = "FrontView"


# --- Configuration for Video 2 (Adapted from back.py) ---
VIDEO_SOURCE_2 = "output_video2_synced.mp4"
ROI_COORDS_2 = [ # back.py ROIs
    (954, 534, 156, 160),    # V2_ROI0: Cylinder 1
    (1143, 510, 174, 187),   # V2_ROI1: Cylinder 2
    (1363, 543, 182, 176),   # V2_ROI2: Cylinder 3
    (1605, 532, 129, 168),   # V2_ROI3: Cylinder 4
]
NUM_ROIS_2 = len(ROI_COORDS_2)
if NUM_ROIS_2 != NUM_GRAPH_COLUMNS:
    print(f"Warning: NUM_ROIS_2 ({NUM_ROIS_2}) does not match NUM_GRAPH_COLUMNS ({NUM_GRAPH_COLUMNS}). Layout might be affected.")

CHANGE_THRESHOLD_2 = 3.0
PERSISTENCE_FRAMES_2 = 1
LIVE_GRAPH_MAX_MAGNITUDE_2 = 12.0
PLOT_YLIM_TOP_2 = 12.0
VIDEO_2_NAME = "BackView"

# ROI mapping for Video 2 (V2_R3 corresponds to V1_R0, etc.)
# This list provides the V2 ROI index that should appear in a given column,
# when columns are indexed 0 (for V1_R0) to 3 (for V1_R3).
V2_ROI_MAPPING_FOR_DISPLAY = [3, 2, 1, 0]


# --- Helper Function: Get Timestamp (from back.py) ---
def get_timestamp_from_video_seconds(cap, start_time_sec=0.0, frame_num=0, fps_fallback= PLAYBACK_FPS):
    msec = cap.get(cv2.CAP_PROP_POS_MSEC)
    if msec > 0:
        return msec / 1000.0
    else:
        current_fps = cap.get(cv2.CAP_PROP_FPS)
        if not current_fps or current_fps <= 0: current_fps = fps_fallback
        if current_fps > 0: return float(frame_num) / current_fps
        else: return time.time() - start_time_sec

# --- Helper Function to Draw Live Graph (from back.py) ---
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

# --- Savitzky-Golay filter (used by Video 2 processing) ---
def apply_savgol_filter(signal, window_length=11, polyorder=2):
    if len(signal) < window_length: return signal
    wl = min(window_length, len(signal))
    if wl % 2 == 0: wl -= 1
    if wl < 3 : return signal 
    return savgol_filter(signal, window_length=wl, polyorder=polyorder)


# --- Motion Processing Function for Video 1 (Optical Flow) ---
def process_frame_video1(frame, frame_timestamp_sec, config, state, frame_count_v1):
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

        if p0 is None or len(p0) < 5 or frame_count_v1 % config['REDETECT_INTERVAL'] == 0 or prev_gray_roi is None:
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


# --- Motion Processing Function for Video 2 (Frame Differencing from back.py) ---
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
            previous_gray_rois[i] = current_roi_gray
            current_roi_change_scores[i] = 0.0
            continue

        diff_roi = cv2.absdiff(previous_gray_rois[i], current_roi_gray)
        change_score = np.mean(diff_roi)
        
        roi_change_history[i].append(change_score)
        filtered_score = change_score
        if len(roi_change_history[i]) >= 5: 
            wl = min(11, len(roi_change_history[i])) 
            if wl % 2 == 0: wl -=1
            if wl >=3: 
                 filtered_values = apply_savgol_filter(list(roi_change_history[i]), wl, 2)
                 filtered_score = filtered_values[-1]
        
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
            roi_change_counters[i] = 0
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
            roi_static_counters[i] = 0
        
        previous_gray_rois[i] = current_roi_gray.copy()

    state['previous_gray_rois'] = previous_gray_rois
    state['roi_states_text'] = roi_states_text
    state['roi_start_times_sec'] = roi_start_times_sec
    state['roi_change_counters'] = roi_change_counters
    state['roi_static_counters'] = roi_static_counters
    state['roi_change_history'] = roi_change_history
    
    return state, new_events, current_roi_change_scores, roi_states_text

# --- New Function: Calculate and Save Lead/Lag Analysis ---
def calculate_and_save_lead_lag(v1_start_events, v2_start_events, roi_mapping,
                                max_diff_sec, output_filename):
    print(f"Calculating lead/lag analysis, saving to {output_filename}...")
    lead_lag_results = []

    # Sort events by timestamp, just in case they are not already
    v1_start_events.sort(key=lambda x: x['timestamp'])
    v2_start_events.sort(key=lambda x: x['timestamp'])

    for v1_event in v1_start_events:
        v1_roi_idx = v1_event['roi_idx']
        v1_ts = v1_event['timestamp']
        v1_val = v1_event['value']

        if v1_roi_idx >= len(roi_mapping):
            # This V1 ROI does not have a mapping defined (e.g., if NUM_ROIS_1 > NUM_GRAPH_COLUMNS)
            continue
        
        mapped_v2_roi_idx = roi_mapping[v1_roi_idx]

        candidate_v2_events = [
            v2_event for v2_event in v2_start_events 
            if v2_event['roi_idx'] == mapped_v2_roi_idx
        ]

        if not candidate_v2_events:
            continue # No V2 start events for the corresponding ROI

        # Find the V2 event closest in time to the V1 event
        best_v2_match = None
        min_time_abs_diff = float('inf')

        for v2_event in candidate_v2_events:
            time_abs_diff = abs(v2_event['timestamp'] - v1_ts)
            if time_abs_diff < min_time_abs_diff:
                min_time_abs_diff = time_abs_diff
                best_v2_match = v2_event
        
        if best_v2_match and min_time_abs_diff <= max_diff_sec:
            v2_ts = best_v2_match['timestamp']
            v2_val = best_v2_match['value']
            time_difference_v2_minus_v1 = v2_ts - v1_ts

            if time_difference_v2_minus_v1 > 0:
                description = f"{VIDEO_1_NAME} leads {VIDEO_2_NAME} by {abs(time_difference_v2_minus_v1):.3f}s"
            elif time_difference_v2_minus_v1 < 0:
                description = f"{VIDEO_2_NAME} leads {VIDEO_1_NAME} by {abs(time_difference_v2_minus_v1):.3f}s"
            else:
                description = "Simultaneous Start"
            
            lead_lag_results.append({
                'V1_ROI_Index': v1_roi_idx,
                'V2_ROI_Index_Corresp': mapped_v2_roi_idx,
                'V1_Start_Time_s': f"{v1_ts:.3f}",
                'V2_Start_Time_s': f"{v2_ts:.3f}",
                'Time_Diff_V2_minus_V1_s': f"{time_difference_v2_minus_v1:.3f}",
                'Lead_Lag_Description': description,
                'V1_Event_Value': f"{v1_val:.4f}",
                'V2_Event_Value': f"{v2_val:.4f}"
            })

    if not lead_lag_results:
        print("No valid lead/lag pairs found within the specified time difference.")
        # Create an empty file with headers if desired, or just skip
        with open(output_filename, 'w', newline='') as csvfile:
            fieldnames = ['V1_ROI_Index', 'V2_ROI_Index_Corresp', 'V1_Start_Time_s', 
                          'V2_Start_Time_s', 'Time_Diff_V2_minus_V1_s', 'Lead_Lag_Description',
                          'V1_Event_Value', 'V2_Event_Value']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        return

    try:
        with open(output_filename, 'w', newline='') as csvfile:
            fieldnames = ['V1_ROI_Index', 'V2_ROI_Index_Corresp', 'V1_Start_Time_s', 
                          'V2_Start_Time_s', 'Time_Diff_V2_minus_V1_s', 'Lead_Lag_Description',
                          'V1_Event_Value', 'V2_Event_Value']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in lead_lag_results:
                writer.writerow(row)
        print(f"Lead/lag analysis successfully saved to {output_filename}")
    except IOError as e:
        print(f"Error writing lead/lag CSV file {output_filename}: {e}")


# --- Main Execution ---
if __name__ == "__main__":
    cap1 = cv2.VideoCapture(VIDEO_SOURCE_1)
    if not cap1.isOpened(): print(f"ERROR: Could not open video {VIDEO_SOURCE_1}"); exit()
    original_fps1 = cap1.get(cv2.CAP_PROP_FPS)
    if original_fps1 <= 0: original_fps1 = 30 
    config1 = {
        'ROI_COORDS': ROI_COORDS_1, 'NUM_ROIS': NUM_ROIS_1,
        'MOTION_THRESHOLD': MOTION_THRESHOLD_1,
        'FEATURE_PARAMS': FEATURE_PARAMS_1, 'LK_PARAMS': LK_PARAMS_1,
        'REDETECT_INTERVAL': REDETECT_INTERVAL_1, 'FPS': original_fps1
    }
    state1 = {
        'prev_grays': [None] * NUM_ROIS_1,
        'prev_pts': [np.array([], dtype=np.float32).reshape(0, 1, 2)] * NUM_ROIS_1,
        'previous_motion_statuses': [False] * NUM_ROIS_1, 
        'roi_start_times_sec': [None] * NUM_ROIS_1
    }
    ret1_init, frame1_init = cap1.read()
    if ret1_init:
        for i_init in range(NUM_ROIS_1):
            x_init, y_init, w_init, h_init = ROI_COORDS_1[i_init]
            fh_init, fw_init = frame1_init.shape[:2]
            if not (0 <= y_init < fh_init and 0 <= x_init < fw_init and y_init+h_init <= fh_init and x_init+w_init <= fw_init and w_init > 0 and h_init > 0) :
                continue
            state1['prev_grays'][i_init] = cv2.cvtColor(frame1_init[y_init:y_init+h_init, x_init:x_init+w_init], cv2.COLOR_BGR2GRAY)
        cap1.set(cv2.CAP_PROP_POS_FRAMES, 0) 
    else:
        print(f"ERROR: Could not read initial frame from {VIDEO_SOURCE_1}"); cap1.release(); exit()

    cap2 = cv2.VideoCapture(VIDEO_SOURCE_2)
    if not cap2.isOpened(): print(f"ERROR: Could not open video {VIDEO_SOURCE_2}"); cap1.release(); exit()
    original_fps2 = cap2.get(cv2.CAP_PROP_FPS)
    if original_fps2 <= 0: original_fps2 = 30 
    config2 = {
        'ROI_COORDS': ROI_COORDS_2, 'NUM_ROIS': NUM_ROIS_2,
        'CHANGE_THRESHOLD': CHANGE_THRESHOLD_2,
        'PERSISTENCE_FRAMES': PERSISTENCE_FRAMES_2, 'FPS': original_fps2
    }
    state2 = {
        'previous_gray_rois': [None] * NUM_ROIS_2,
        'roi_states_text': ['STOPPED'] * NUM_ROIS_2, 
        'roi_start_times_sec': [None] * NUM_ROIS_2,
        'roi_change_counters': [0] * NUM_ROIS_2,
        'roi_static_counters': [0] * NUM_ROIS_2,
        'roi_change_history': [collections.deque(maxlen=30) for _ in range(NUM_ROIS_2)]
    }

    motion_histories1 = {i: collections.deque(np.zeros(GRAPH_HISTORY_LENGTH), maxlen=GRAPH_HISTORY_LENGTH) for i in range(NUM_ROIS_1)}
    motion_histories2 = {i: collections.deque(np.zeros(GRAPH_HISTORY_LENGTH), maxlen=GRAPH_HISTORY_LENGTH) for i in range(NUM_ROIS_2)}
    
    all_timestamps_global = []
    all_scores1_per_roi = {i: [] for i in range(NUM_ROIS_1)}
    all_scores2_per_roi = {i: [] for i in range(NUM_ROIS_2)}

    # New: Lists to store all 'START' events for lead/lag analysis
    all_v1_start_events = []
    all_v2_start_events = []

    top_row_h = VIDEO_PANE_H
    graph_section_h_total = NUM_GRAPH_ROWS_PER_COLUMN * (INDIVIDUAL_GRAPH_H + LIVE_GRAPH_PADDING) + LIVE_GRAPH_PADDING
    combined_canvas_w = 2 * VIDEO_PANE_W
    combined_canvas_h = top_row_h + graph_section_h_total
    
    frame_count_global = 0
    system_start_time_sec = time.time()

    if SHOW_VIDEO_OUTPUT:
        cv2.namedWindow('Dual Stream Motion Analysis', cv2.WINDOW_NORMAL) # Make window resizable

    try:
        with open(LOG_FILENAME, 'w') as log_file:
            log_file.write("Timestamp(s),VideoName,ROI_Index,Event,Duration(s),ValueAtEvent\n")
            print(f"Logging events to: {LOG_FILENAME}")

            while True:
                loop_start_time = time.time()
                ret1, frame1_orig = cap1.read()
                ret2, frame2_orig = cap2.read()

                if not ret1 or not ret2:
                    print("End of one or both video streams.")
                    break
                
                frame_count_global += 1
                timestamp1 = get_timestamp_from_video_seconds(cap1, system_start_time_sec, frame_count_global, original_fps1)
                timestamp2 = get_timestamp_from_video_seconds(cap2, system_start_time_sec, frame_count_global, original_fps2)
                current_timestamp_global = max(timestamp1, timestamp2) 
                
                if all_timestamps_global and current_timestamp_global <= all_timestamps_global[-1]:
                    fallback_fps = max(original_fps1, original_fps2)
                    current_timestamp_global = all_timestamps_global[-1] + (1.0 / fallback_fps if fallback_fps > 0 else 0.033)
                all_timestamps_global.append(current_timestamp_global)

                state1, events1, scores1, statuses_text1 = process_frame_video1(frame1_orig, timestamp1, config1, state1, frame_count_global)
                for i in range(NUM_ROIS_1):
                    motion_histories1[i].append(scores1[i])
                    all_scores1_per_roi[i].append(scores1[i])
                for event_data in events1:
                    roi_idx, ev_type, ts, dur, val = event_data
                    dur_str = f"{dur:.3f}" if dur is not None else "N/A"
                    log_msg = f"{ts:.3f},{VIDEO_1_NAME},{roi_idx},{ev_type},{dur_str},{val:.4f}\n"
                    log_file.write(log_msg);
                    if ev_type == 'START': # New: Collect V1 START events
                        all_v1_start_events.append({'timestamp': ts, 'roi_idx': roi_idx, 'value': val})

                state2, events2, scores2, statuses_text2 = process_frame_video2(frame2_orig, timestamp2, config2, state2)
                for i in range(NUM_ROIS_2):
                    motion_histories2[i].append(scores2[i])
                    all_scores2_per_roi[i].append(scores2[i])
                for event_data in events2:
                    roi_idx, ev_type, ts, dur, val = event_data
                    dur_str = f"{dur:.3f}" if dur is not None else "N/A"
                    log_msg = f"{ts:.3f},{VIDEO_2_NAME},{roi_idx},{ev_type},{dur_str},{val:.4f}\n"
                    log_file.write(log_msg);
                    if ev_type == 'START': # New: Collect V2 START events
                        all_v2_start_events.append({'timestamp': ts, 'roi_idx': roi_idx, 'value': val})
                
                log_file.flush()

                if SHOW_VIDEO_OUTPUT:
                    combined_canvas = np.zeros((combined_canvas_h, combined_canvas_w, 3), dtype=np.uint8)
                    
                    frame1_display = cv2.resize(frame1_orig, (VIDEO_PANE_W, VIDEO_PANE_H))
                    for i, (x, y, w, h) in enumerate(config1['ROI_COORDS']):
                        scale_x, scale_y = VIDEO_PANE_W / frame1_orig.shape[1], VIDEO_PANE_H / frame1_orig.shape[0]
                        dx, dy, dw, dh = int(x*scale_x), int(y*scale_y), int(w*scale_x), int(h*scale_y)
                        color = (0, 255, 0) if statuses_text1[i] == 'MOVING' else (0, 0, 255)
                        cv2.rectangle(frame1_display, (dx, dy), (dx + dw, dy + dh), color, 2)
                        cv2.putText(frame1_display, f"V1_R{i}", (dx, dy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    combined_canvas[0:VIDEO_PANE_H, 0:VIDEO_PANE_W] = frame1_display

                    frame2_display = cv2.resize(frame2_orig, (VIDEO_PANE_W, VIDEO_PANE_H))
                    for i, (x, y, w, h) in enumerate(config2['ROI_COORDS']):
                        scale_x, scale_y = VIDEO_PANE_W / frame2_orig.shape[1], VIDEO_PANE_H / frame2_orig.shape[0]
                        dx, dy, dw, dh = int(x*scale_x), int(y*scale_y), int(w*scale_x), int(h*scale_y)
                        color = (0, 255, 0) if statuses_text2[i] == 'MOVING' else (0, 0, 255)
                        cv2.rectangle(frame2_display, (dx, dy), (dx + dw, dy + dh), color, 2)
                        cv2.putText(frame2_display, f"V2_R{i}", (dx, dy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    combined_canvas[0:VIDEO_PANE_H, VIDEO_PANE_W:2*VIDEO_PANE_W] = frame2_display
                    
                    # Draw Live Graphs in mapped columns
                    for col_idx in range(NUM_GRAPH_COLUMNS):
                        # Top graph in column (from Video 1)
                        v1_roi_idx = col_idx
                        gx = LIVE_GRAPH_PADDING + col_idx * (INDIVIDUAL_GRAPH_W + LIVE_GRAPH_PADDING)
                        gy_top = VIDEO_PANE_H + LIVE_GRAPH_PADDING
                        draw_live_graph(combined_canvas, motion_histories1[v1_roi_idx], 
                                        f"V1_R{v1_roi_idx}", statuses_text1[v1_roi_idx],
                                        gx, gy_top, INDIVIDUAL_GRAPH_W, INDIVIDUAL_GRAPH_H,
                                        LIVE_GRAPH_MAX_MAGNITUDE_1, MOTION_THRESHOLD_1)

                        # Bottom graph in column (from Video 2, mapped)
                        v2_actual_roi_idx_to_display = V2_ROI_MAPPING_FOR_DISPLAY[col_idx]
                        gy_bottom = VIDEO_PANE_H + LIVE_GRAPH_PADDING + (INDIVIDUAL_GRAPH_H + LIVE_GRAPH_PADDING)
                        draw_live_graph(combined_canvas, motion_histories2[v2_actual_roi_idx_to_display], 
                                        f"V2_R{v2_actual_roi_idx_to_display}", statuses_text2[v2_actual_roi_idx_to_display],
                                        gx, gy_bottom, INDIVIDUAL_GRAPH_W, INDIVIDUAL_GRAPH_H,
                                        LIVE_GRAPH_MAX_MAGNITUDE_2, CHANGE_THRESHOLD_2)

                    proc_fps_text = f"Proc. FPS: {1.0 / (time.time() - loop_start_time) if (time.time() - loop_start_time) > 0 else 0:.1f}"
                    time_text = f"Time: {current_timestamp_global:.2f}s"
                    cv2.putText(combined_canvas, proc_fps_text, (10, combined_canvas_h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
                    cv2.putText(combined_canvas, time_text, (10, combined_canvas_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
                    
                    disp_h, disp_w = combined_canvas.shape[:2]
                    scale_factor = 1.0
                    scale_w_factor = 1.0
                    if disp_w > MAX_DISPLAY_W: scale_w_factor = MAX_DISPLAY_W / disp_w
                    scale_h_factor = 1.0
                    if disp_h > MAX_DISPLAY_H: scale_h_factor = MAX_DISPLAY_H / disp_h
                    scale_factor = min(scale_w_factor, scale_h_factor, 1.0) 

                    display_scaled_window = combined_canvas
                    if scale_factor < 1.0:
                        new_w, new_h = int(disp_w * scale_factor), int(disp_h * scale_factor)
                        if new_w > 0 and new_h > 0:
                            display_scaled_window = cv2.resize(combined_canvas, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    
                    cv2.imshow('Dual Stream Motion Analysis', display_scaled_window)
                    
                    wait_ms = max(1, int(1000 / PLAYBACK_FPS - (time.time() - loop_start_time) * 1000)) # Use PLAYBACK_FPS for controlled speed
                    key = cv2.waitKey(wait_ms) 
                    if key == 27: print("ESC key pressed, exiting..."); break
                    elif key == ord('q'): print("Q key pressed, exiting..."); break                        
                    elif key == 32:  # Space key
                        print("Space key pressed, pausing...")
                        while True:
                            key_pause = cv2.waitKey(0)
                            if key_pause == 32: print("Space key pressed, resuming..."); break   
                            elif key_pause == 27: print("ESC key pressed, exiting..."); cv2.destroyAllWindows(); cap1.release(); cap2.release(); exit()
                            elif key_pause == ord('q'): print("Q key pressed, exiting..."); cv2.destroyAllWindows(); cap1.release(); cap2.release(); exit()
                else: 
                    if frame_count_global % 100 == 0: 
                        print(f"Processed frame {frame_count_global}, Time: {current_timestamp_global:.2f}s (video output disabled)")
        
    except IOError as e: print(f"Error with log file {LOG_FILENAME}: {e}")
    except Exception as e: print(f"An unexpected error in main loop: {e}"); import traceback; traceback.print_exc()
    finally:
        print("Releasing video captures...")
        cap1.release(); cap2.release()
        if SHOW_VIDEO_OUTPUT: print("Destroying OpenCV windows..."); cv2.destroyAllWindows()
        print(f"Analysis complete. Processed {frame_count_global} frames.")

        # --- New: Generate Lead/Lag Analysis File ---
        try:
            calculate_and_save_lead_lag(
                all_v1_start_events,
                all_v2_start_events,
                V2_ROI_MAPPING_FOR_DISPLAY,
                MAX_TIME_DIFFERENCE_FOR_MATCH_SEC,
                LEAD_LAG_FILENAME
            )
        except Exception as e:
            print(f"Error generating lead/lag analysis file: {e}"); import traceback; traceback.print_exc()

        print("Generating final summary graph...")
        try:
            if all_timestamps_global and (any(all_scores1_per_roi[i] for i in range(NUM_ROIS_1)) or \
                                         any(all_scores2_per_roi[i] for i in range(NUM_ROIS_2))):
                
                fig_rows = NUM_GRAPH_COLUMNS 
                fig, axes = plt.subplots(fig_rows, 2, figsize=(18, 3 * fig_rows), sharex=True, squeeze=False)

                for plot_row_idx in range(fig_rows):
                    v1_roi_to_plot = plot_row_idx
                    ax1 = axes[plot_row_idx, 0]
                    if v1_roi_to_plot < NUM_ROIS_1 and all_scores1_per_roi[v1_roi_to_plot]:
                        ax1.plot(all_timestamps_global, all_scores1_per_roi[v1_roi_to_plot], label=f'{VIDEO_1_NAME} ROI {v1_roi_to_plot} Magnitude', linewidth=1)
                    ax1.axhline(MOTION_THRESHOLD_1, color='r', linestyle='--', label=f'Thresh={MOTION_THRESHOLD_1:.1f}')
                    ax1.set_ylabel("Avg Motion Magnitude (px)")
                    ax1.set_title(f"{VIDEO_1_NAME} ROI {v1_roi_to_plot}")
                    ax1.set_ylim(bottom=-0.5, top=PLOT_YLIM_TOP_1)
                    ax1.legend(fontsize='small'); ax1.grid(True)

                    v2_actual_roi_to_plot = V2_ROI_MAPPING_FOR_DISPLAY[plot_row_idx]
                    ax2 = axes[plot_row_idx, 1]
                    if v2_actual_roi_to_plot < NUM_ROIS_2 and all_scores2_per_roi[v2_actual_roi_to_plot]:
                        ax2.plot(all_timestamps_global, all_scores2_per_roi[v2_actual_roi_to_plot], label=f'{VIDEO_2_NAME} ROI {v2_actual_roi_to_plot} Change Score', linewidth=1, color='green')
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
                print("Warning: No motion data collected, cannot generate final graph.")
        except Exception as e:
            print(f"Error generating final graph: {e}"); import traceback; traceback.print_exc()
        
        print(f"Log file saved to {LOG_FILENAME}")
        print(f"Lead/lag analysis file saved to {LEAD_LAG_FILENAME}") # Reminder
        print("Script finished.")