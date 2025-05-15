import cv2
import numpy as np
import math

# --- Configuration Parameters ---
# START TUNING THESE VALUES FIRST:
THRESHOLD_VALUE = 90  # STARTING POINT: Lower this if marks don't appear; raise if too much noise.
MIN_CONTOUR_AREA = 5   # STARTING POINT: For small marks, area will be small.
MAX_CONTOUR_AREA = 100 # STARTING POINT: Max area for a mark within the small ROI.
GAUSSIAN_BLUR_KERNEL_SIZE = (3, 3) # Reduced blur for smaller features. Try (1,1) for almost no blur.

# These usually stay fixed:
NUM_MARKS_EXPECTED = 4
MORPH_KERNEL_SIZE = (3,3) # Kernel for morphological opening

# --- ROI Definition ---
ROI_X, ROI_Y, ROI_W, ROI_H = 630, 460, 73, 65
ROI_RECT = (ROI_X, ROI_Y, ROI_W, ROI_H)

# --- Global state variables ---
previous_marks_coords = None
previous_center_of_marks = None
total_rotation_accumulator = 0.0
DEBUG_CONTOUR_AREAS = False # Set to True to print all contour areas found in ROI

def find_chalk_marks(frame, roi_rect=None):
    global DEBUG_CONTOUR_AREAS
    if roi_rect:
        x, y, w, h = roi_rect
        frame_h, frame_w = frame.shape[:2]
        x = max(0, x); y = max(0, y)
        w = min(w, frame_w - x); h = min(h, frame_h - y)
        if w <= 0 or h <= 0: return []
        frame_to_process = frame[y:y+h, x:x+w]
        offset_x, offset_y = x, y
    else:
        frame_to_process = frame
        offset_x, offset_y = 0, 0

    if frame_to_process.size == 0: return []

    gray = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2GRAY)
    
    if GAUSSIAN_BLUR_KERNEL_SIZE[0] > 0 and GAUSSIAN_BLUR_KERNEL_SIZE[1] > 0: # Apply blur only if kernel size is positive
        blurred = cv2.GaussianBlur(gray, GAUSSIAN_BLUR_KERNEL_SIZE, 0)
    else:
        blurred = gray # No blur
        
    _, thresh = cv2.threshold(blurred, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
    
    if MORPH_KERNEL_SIZE[0] > 0 and MORPH_KERNEL_SIZE[1] > 0:
        kernel = np.ones(MORPH_KERNEL_SIZE, np.uint8)
        thresh_opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    else:
        thresh_opened = thresh # No morphological operation

    contours, _ = cv2.findContours(thresh_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_marks = []
    if DEBUG_CONTOUR_AREAS and contours:
        all_areas = [cv2.contourArea(cnt) for cnt in contours]
        print(f"    ROI raw contour areas: {sorted(all_areas)}")

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if MIN_CONTOUR_AREA < area < MAX_CONTOUR_AREA:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx_roi = int(M["m10"] / M["m00"])
                cy_roi = int(M["m01"] / M["m00"])
                abs_cX = cx_roi + offset_x
                abs_cY = cy_roi + offset_y
                detected_marks.append(np.array([abs_cX, abs_cY]))
    return detected_marks

def sort_marks_by_angle(marks, center):
    if not marks or center is None: return []
    angles = [np.arctan2(mark[1] - center[1], mark[0] - center[0]) for mark in marks]
    sorted_indices = np.argsort(angles)
    return [marks[i] for i in sorted_indices]

def calculate_rotation_angle(prev_points, curr_points, prev_center, curr_center):
    delta_angles = []
    for prev_pt, curr_pt in zip(prev_points, curr_points):
        vec_prev = prev_pt - prev_center
        vec_curr = curr_pt - curr_center
        angle_prev = np.arctan2(vec_prev[1], vec_prev[0])
        angle_curr = np.arctan2(vec_curr[1], vec_curr[0])
        delta_angle = angle_curr - angle_prev
        delta_angle = (delta_angle + np.pi) % (2 * np.pi) - np.pi
        delta_angles.append(np.degrees(delta_angle))
    if not delta_angles: return 0.0
    avg_delta_angle = np.mean(delta_angles)
    if avg_delta_angle > 180: avg_delta_angle -= 360
    elif avg_delta_angle < -180: avg_delta_angle += 360
    return avg_delta_angle

def process_frame(frame, frame_number, current_roi_rect):
    global previous_marks_coords, previous_center_of_marks, total_rotation_accumulator, DEBUG_CONTOUR_AREAS

    display_frame = frame.copy()

    if current_roi_rect:
        x, y, w, h = current_roi_rect
        # (draw ROI rectangle - code omitted for brevity, it's the same as before)
        cv2.rectangle(display_frame, (x,y), (x+w, y+h), (255,255,0), 2)


    current_raw_marks = find_chalk_marks(frame, current_roi_rect)

    if len(current_raw_marks) != NUM_MARKS_EXPECTED:
        # (draw raw marks code - omitted for brevity)
        for i, mark_coord in enumerate(current_raw_marks):
             cv2.circle(display_frame, tuple(mark_coord.astype(int)), 3, (0, 165, 255), -1) # Smaller circle for raw
             # cv2.putText(display_frame, f"r{i}", tuple(mark_coord.astype(int) + np.array([3,3])),
             #            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 165, 255), 1)
        if frame_number % 30 == 0 or len(current_raw_marks) > 0 : # Print less frequently unless some marks are found
             print(f"Frame {frame_number}: Expected {NUM_MARKS_EXPECTED} marks, found {len(current_raw_marks)}. Skipping.")
        return display_frame, 0.0

    current_center_of_marks = np.mean(current_raw_marks, axis=0)
    current_sorted_marks = sort_marks_by_angle(current_raw_marks, current_center_of_marks)

    # (drawing sorted marks and centers code - omitted for brevity)
    center_curr_tuple = tuple(current_center_of_marks.astype(int))
    cv2.circle(display_frame, center_curr_tuple, 5, (255, 100, 100), -1)
    for i, mark_coord in enumerate(current_sorted_marks):
        mark_tuple = tuple(mark_coord.astype(int))
        cv2.circle(display_frame, mark_tuple, 7, (0, 255, 0), 1) # Thinner circles

    rotation_this_frame = 0.0
    if previous_marks_coords is not None and previous_center_of_marks is not None and \
       len(previous_marks_coords) == NUM_MARKS_EXPECTED:
        rotation_this_frame = calculate_rotation_angle(
            previous_marks_coords, current_sorted_marks,
            previous_center_of_marks, current_center_of_marks
        )
        total_rotation_accumulator += rotation_this_frame
        print(f"Frame {frame_number}: Rotation = {rotation_this_frame:.2f} deg, Total = {total_rotation_accumulator:.2f} deg")
    else:
        if not previous_marks_coords : print(f"Frame {frame_number}: Initializing marks.")

    previous_marks_coords = current_sorted_marks
    previous_center_of_marks = current_center_of_marks
    
    cv2.putText(display_frame, f"Rot: {rotation_this_frame:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),1)
    cv2.putText(display_frame, f"Total: {total_rotation_accumulator:.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)

    return display_frame, rotation_this_frame

# --- Main Execution ---
if __name__ == "__main__":
    video_path = "fronti.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video source {video_path}")
        exit()

    frame_count = 0
    SELECTED_ROI_RECT = ROI_RECT

    # Create a trackbar window for THRESHOLD_VALUE for easier tuning
    cv2.namedWindow("Thresholded ROI (for tuning)")
    def on_trackbar(val):
        global THRESHOLD_VALUE
        THRESHOLD_VALUE = val
    cv2.createTrackbar("Threshold", "Thresholded ROI (for tuning)", THRESHOLD_VALUE, 255, on_trackbar)


    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break
        frame_count += 1

        processed_display_frame, rotation_val = process_frame(frame, frame_count, SELECTED_ROI_RECT)
        cv2.imshow("Wheel Rotation Tracking", processed_display_frame)
        
        # Display the thresholded ROI for tuning
        if SELECTED_ROI_RECT:
            x_r, y_r, w_r, h_r = SELECTED_ROI_RECT
            # (safe ROI cropping - same as before)
            frame_h, frame_w = frame.shape[:2]
            x_r = max(0, x_r); y_r = max(0, y_r)
            w_r = min(w_r, frame_w - x_r); h_r = min(h_r, frame_h - y_r)

            if w_r > 0 and h_r > 0:
                roi_content = frame[y_r:y_r+h_r, x_r:x_r+w_r]
                if roi_content.size > 0:
                    gray_roi = cv2.cvtColor(roi_content, cv2.COLOR_BGR2GRAY)
                    
                    if GAUSSIAN_BLUR_KERNEL_SIZE[0]>0 and GAUSSIAN_BLUR_KERNEL_SIZE[1]>0:
                        blurred_roi = cv2.GaussianBlur(gray_roi, GAUSSIAN_BLUR_KERNEL_SIZE, 0)
                    else:
                        blurred_roi = gray_roi
                        
                    _, thresh_roi_display = cv2.threshold(blurred_roi, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
                    
                    if MORPH_KERNEL_SIZE[0]>0 and MORPH_KERNEL_SIZE[1]>0:
                        kernel_display = np.ones(MORPH_KERNEL_SIZE, np.uint8)
                        thresh_opened_roi_display = cv2.morphologyEx(thresh_roi_display, cv2.MORPH_OPEN, kernel_display, iterations=1)
                    else:
                        thresh_opened_roi_display = thresh_roi_display
                        
                    cv2.imshow("Thresholded ROI (for tuning)", thresh_opened_roi_display)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'): break
        elif key == ord('p'): cv2.waitKey(0)
        elif key == ord('d'): # Toggle debug print for contour areas
            DEBUG_CONTOUR_AREAS = not DEBUG_CONTOUR_AREAS
            print(f"DEBUG Contour Areas: {'ON' if DEBUG_CONTOUR_AREAS else 'OFF'}")


    cap.release()
    cv2.destroyAllWindows()
    print(f"--- Final Total Accumulated Rotation: {total_rotation_accumulator:.2f} degrees ---")