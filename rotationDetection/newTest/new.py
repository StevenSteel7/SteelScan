import cv2
import numpy as np
import math

# --- Configuration Parameters ---
THRESHOLD_VALUE = 115  # Initial, use trackbar to tune
MIN_CONTOUR_AREA_MARK = 3   # For a single mark in its search window
MAX_CONTOUR_AREA_MARK = 80  # Max area for a single mark
GAUSSIAN_BLUR_KERNEL_SIZE = (3, 3)
MORPH_KERNEL_SIZE = (3,3)
NUM_MARKS_EXPECTED = 4

# --- Main ROI (for debug display and fallback if needed) ---
# This can be broader than the immediate chalk mark area if you want a general view for threshold tuning
ROI_X_DBG, ROI_Y_DBG, ROI_W_DBG, ROI_H_DBG = 600, 430, 150, 150 # Example for debug view
ROI_RECT_DEBUG = (ROI_X_DBG, ROI_Y_DBG, ROI_W_DBG, ROI_H_DBG)

# --- TRACKING PARAMETERS ---
MARK_SEARCH_WINDOW_RADIUS = 15 # Pixels around last known position

# --- Global state variables ---
tracked_marks_abs = [None] * NUM_MARKS_EXPECTED # Stores np.array([x,y]) for each mark
previous_center_of_tracked_marks = None
total_rotation_accumulator = 0.0
initial_marks_defined = False
clicked_points = [] # To store points clicked by the user

# Helper to find ONE best mark in a small ROI (search window)
def find_best_mark_in_sub_roi(frame_segment, min_area, max_area):
    if frame_segment.size == 0: return None
    gray = cv2.cvtColor(frame_segment, cv2.COLOR_BGR2GRAY)
    blurred = gray
    if GAUSSIAN_BLUR_KERNEL_SIZE[0] > 0: blurred = cv2.GaussianBlur(gray, GAUSSIAN_BLUR_KERNEL_SIZE, 0)
    _, thresh = cv2.threshold(blurred, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
    opened_thresh = thresh
    if MORPH_KERNEL_SIZE[0] > 0:
        kernel = np.ones(MORPH_KERNEL_SIZE, np.uint8)
        opened_thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(opened_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_mark_coord_in_segment = None
    max_found_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            if area > max_found_area:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"]); cy = int(M["m01"] / M["m00"])
                    best_mark_coord_in_segment = np.array([cx, cy])
                    max_found_area = area
    return best_mark_coord_in_segment

# (calculate_rotation_angle - same as before)
def calculate_rotation_angle(prev_points, curr_points, prev_center, curr_center):
    delta_angles = []
    valid_pairs = 0
    for prev_pt, curr_pt in zip(prev_points, curr_points):
        if prev_pt is None or curr_pt is None or prev_center is None or curr_center is None:
            continue # Skip if any part of this pair is missing
        valid_pairs +=1
        vec_prev = prev_pt - prev_center; vec_curr = curr_pt - curr_center
        angle_prev = np.arctan2(vec_prev[1], vec_prev[0]); angle_curr = np.arctan2(vec_curr[1], vec_curr[0])
        delta_angle = (np.degrees(angle_curr - angle_prev) + 180) % 360 - 180
        delta_angles.append(delta_angle)
    if not delta_angles or valid_pairs < 2 : return 0.0 # Need at least 2 points for a somewhat reliable angle
    avg_delta_angle = np.mean(delta_angles) # Consider median for robustness if needed
    # Normalize final average
    if avg_delta_angle > 180: avg_delta_angle -= 360
    elif avg_delta_angle < -180: avg_delta_angle += 360
    return avg_delta_angle


def mouse_callback(event, x, y, flags, param):
    global clicked_points, initial_marks_defined, tracked_marks_abs
    if initial_marks_defined: # Don't process clicks after initial definition
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) < NUM_MARKS_EXPECTED:
            clicked_points.append(np.array([x, y]))
            print(f"Clicked point {len(clicked_points)}: ({x}, {y})")
            # Draw on the frame being displayed for clicking
            cv2.circle(param['frame_for_clicking'], (x,y), 5, (0,255,0), -1)
            cv2.putText(param['frame_for_clicking'], str(len(clicked_points)), (x+5, y+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.imshow("Click Marks (Frame 1)", param['frame_for_clicking'])

        if len(clicked_points) == NUM_MARKS_EXPECTED:
            print("All 4 marks defined by click. Press any key to start tracking.")
            tracked_marks_abs = list(clicked_points) # Initialize tracked_marks_abs
            initial_marks_defined = True
            # cv2.destroyWindow("Click Marks (Frame 1)") # Keep window open to see clicks then close on key


def process_tracked_marks(frame, frame_idx):
    global tracked_marks_abs, previous_center_of_tracked_marks, total_rotation_accumulator, initial_marks_defined

    if not initial_marks_defined:
        print("Initial marks not yet defined. Click on the first frame.")
        return frame.copy(), 0.0 # Return original frame if not initialized

    display_frame = frame.copy()
    current_found_marks_abs = [None] * NUM_MARKS_EXPECTED
    all_marks_successfully_tracked = True

    for i in range(NUM_MARKS_EXPECTED):
        last_pos = tracked_marks_abs[i]
        if last_pos is None: # Should not happen after click initialization
            all_marks_successfully_tracked = False
            print(f"Error: Mark {i} became None after initialization.")
            continue

        sw_x = int(last_pos[0] - MARK_SEARCH_WINDOW_RADIUS)
        sw_y = int(last_pos[1] - MARK_SEARCH_WINDOW_RADIUS)
        sw_w = int(2 * MARK_SEARCH_WINDOW_RADIUS); sw_h = int(2 * MARK_SEARCH_WINDOW_RADIUS)

        frame_h, frame_w = frame.shape[:2]
        sw_x = max(0, sw_x); sw_y = max(0, sw_y)
        sw_w = min(sw_w, frame_w - sw_x); sw_h = min(sw_h, frame_h - sw_y)
        
        cv2.rectangle(display_frame, (sw_x, sw_y), (sw_x+sw_w, sw_y+sw_h), (0,255,255), 1)

        if sw_w <=0 or sw_h <=0:
            current_found_marks_abs[i] = None
            all_marks_successfully_tracked = False; continue

        mark_segment = frame[sw_y:sw_y+sw_h, sw_x:sw_x+sw_w]
        coord_in_segment = find_best_mark_in_sub_roi(mark_segment, MIN_CONTOUR_AREA_MARK, MAX_CONTOUR_AREA_MARK)

        if coord_in_segment is not None:
            current_found_marks_abs[i] = coord_in_segment + np.array([sw_x, sw_y])
        else:
            current_found_marks_abs[i] = None
            all_marks_successfully_tracked = False
            # print(f"Frame {frame_idx}: Mark {i} lost in its search window.")

    if not all_marks_successfully_tracked or not all(m is not None for m in current_found_marks_abs):
        # If any mark lost, don't update rotation, keep previous valid positions.
        # Draw what was found (if anything)
        num_lost = sum(1 for m in current_found_marks_abs if m is None)
        if frame_idx % 30 == 0: print(f"Frame {frame_idx}: Lost {num_lost} marks. Using last known good positions.")
        
        # For display, show last known good positions if current is None
        display_current_marks = []
        for curr, prev in zip(current_found_marks_abs, tracked_marks_abs):
            display_current_marks.append(curr if curr is not None else prev)
        
        if all(m is not None for m in display_current_marks):
            temp_center = np.mean(np.array(display_current_marks), axis=0)
            cv2.circle(display_frame, tuple(temp_center.astype(int)), 3, (0,0,255), -1) # Red center if uncertain
            for i, mc in enumerate(display_current_marks):
                color = (0,165,255) if current_found_marks_abs[i] is None else (0,255,0) # Orange if estimated, Green if tracked
                cv2.circle(display_frame, tuple(mc.astype(int)), 5, color, 2)
                cv2.putText(display_frame, str(i), tuple(mc.astype(int)+np.array([5,5])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        return display_frame, 0.0 # No rotation calculated

    # All marks successfully tracked
    current_center = np.mean(np.array(current_found_marks_abs), axis=0)
    cv2.circle(display_frame, tuple(current_center.astype(int)), 5, (255,100,100), -1)
    for i, mc in enumerate(current_found_marks_abs):
        cv2.circle(display_frame, tuple(mc.astype(int)), 7, (0,255,0), 2)
        cv2.putText(display_frame, str(i), tuple(mc.astype(int)+np.array([5,5])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,180,0),1)

    rotation_this_frame = 0.0
    if previous_center_of_tracked_marks is not None and all(m is not None for m in tracked_marks_abs):
        rotation_this_frame = calculate_rotation_angle(
            tracked_marks_abs, current_found_marks_abs,
            previous_center_of_tracked_marks, current_center
        )
        total_rotation_accumulator += rotation_this_frame
        # print(f"Frame {frame_idx}: Rot = {rotation_this_frame:.2f} deg, Total = {total_rotation_accumulator:.2f} deg")
    else:
        print(f"Frame {frame_idx}: Initializing rotation baseline.")

    tracked_marks_abs = current_found_marks_abs
    previous_center_of_tracked_marks = current_center
    
    cv2.putText(display_frame, f"Rot: {rotation_this_frame:.2f}",(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)
    cv2.putText(display_frame, f"Total: {total_rotation_accumulator:.2f}",(10,40),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)
    return display_frame, rotation_this_frame

# --- Main Execution ---
if __name__ == "__main__":
    video_path = "fronti.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): print(f"Error opening video {video_path}"); exit()

    # --- Initial Mark Selection by Clicking ---
    ret_first, first_frame = cap.read()
    if not ret_first: print("Cannot read first frame."); cap.release(); exit()
    
    first_frame_copy = first_frame.copy() # Keep original first_frame clean
    cv2.namedWindow("Click Marks (Frame 1)")
    cv2.setMouseCallback("Click Marks (Frame 1)", mouse_callback, {'frame_for_clicking': first_frame_copy})
    cv2.imshow("Click Marks (Frame 1)", first_frame_copy)
    
    print(f"Please click on the {NUM_MARKS_EXPECTED} chalk marks in a consistent order (e.g., clockwise).")
    print("After clicking all marks, press any key on the 'Click Marks' window to continue.")
    
    while not initial_marks_defined:
        cv2.imshow("Click Marks (Frame 1)", first_frame_copy) # Keep showing updates if user clicks
        if cv2.waitKey(50) != -1 and len(clicked_points) == NUM_MARKS_EXPECTED: # Any key press after all points clicked
            initial_marks_defined = True # Ensure flag is set if loop exits due to key press
            break
        if cv2.getWindowProperty("Click Marks (Frame 1)", cv2.WND_PROP_VISIBLE) < 1: # Window closed
            print("Mark selection window closed. Exiting.")
            cap.release()
            cv2.destroyAllWindows()
            exit()


    cv2.destroyWindow("Click Marks (Frame 1)")
    print("Initial mark positions defined:", tracked_marks_abs)
    # --- End Initial Mark Selection ---

    cv2.namedWindow("Thresholded ROI (for tuning)") # For debug threshold view
    def on_trackbar(val): global THRESHOLD_VALUE; THRESHOLD_VALUE = val
    cv2.createTrackbar("Threshold", "Thresholded ROI (for tuning)", THRESHOLD_VALUE, 255, on_trackbar)

    frame_idx_main = 0 # Start from 0 as first frame was used for clicking
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Rewind to process from the first frame again

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx_main += 1

        processed_frame, _ = process_tracked_marks(frame, frame_idx_main)
        cv2.imshow("Wheel Smart Track with Click Init", processed_frame)

        # Debug display for ROI thresholding (using ROI_RECT_DEBUG)
        if ROI_RECT_DEBUG:
            x,y,w,h = ROI_RECT_DEBUG
            fh,fw = frame.shape[:2]
            x,y,w,h = max(0,x),max(0,y),min(w,fw-x),min(h,fh-y)
            if w>0 and h>0:
                roi_dbg_content = frame[y:y+h, x:x+w]
                if roi_dbg_content.size > 0:
                    gray_dbg = cv2.cvtColor(roi_dbg_content, cv2.COLOR_BGR2GRAY)
                    if GAUSSIAN_BLUR_KERNEL_SIZE[0]>0: blurred_dbg = cv2.GaussianBlur(gray_dbg,GAUSSIAN_BLUR_KERNEL_SIZE,0)
                    else: blurred_dbg = gray_dbg
                    _,thresh_dbg = cv2.threshold(blurred_dbg, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
                    if MORPH_KERNEL_SIZE[0]>0:
                        kernel_dbg = np.ones(MORPH_KERNEL_SIZE, np.uint8)
                        opened_dbg = cv2.morphologyEx(thresh_dbg, cv2.MORPH_OPEN, kernel_dbg, iterations=1)
                    else: opened_dbg = thresh_dbg
                    cv2.imshow("Thresholded ROI (for tuning)", opened_dbg)
        
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'): break
        elif key == ord('p'): cv2.waitKey(0)
        elif key == ord('r'): # Force re-initialization by re-clicking
            print("Re-initialization requested. Please click marks on the next first frame.")
            initial_marks_defined = False
            clicked_points = []
            tracked_marks_abs = [None] * NUM_MARKS_EXPECTED
            previous_center_of_tracked_marks = None
            total_rotation_accumulator = 0.0
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Rewind
            frame_idx_main = 0
            
            ret_first_r, first_frame_r = cap.read()
            if not ret_first_r: print("Cannot read first frame for re-click."); break
            first_frame_copy_r = first_frame_r.copy()
            cv2.namedWindow("Click Marks (Frame 1)")
            cv2.setMouseCallback("Click Marks (Frame 1)", mouse_callback, {'frame_for_clicking': first_frame_copy_r})
            cv2.imshow("Click Marks (Frame 1)", first_frame_copy_r)
            while not initial_marks_defined:
                cv2.imshow("Click Marks (Frame 1)", first_frame_copy_r)
                if cv2.waitKey(50) != -1 and len(clicked_points) == NUM_MARKS_EXPECTED:
                    initial_marks_defined = True; break
                if cv2.getWindowProperty("Click Marks (Frame 1)",cv2.WND_PROP_VISIBLE)<1: break
            cv2.destroyWindow("Click Marks (Frame 1)")
            if not initial_marks_defined: print("Re-click cancelled or failed."); break # Exit if re-click fails


    cap.release()
    cv2.destroyAllWindows()
    print(f"--- Final Total Accumulated Rotation: {total_rotation_accumulator:.2f} degrees ---")