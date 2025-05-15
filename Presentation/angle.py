import cv2
import numpy as np
import math
import sys

# --- Parameters ---
VIDEO_SOURCE = 'fronti.mp4'

# ROI parameters
ROI_X, ROI_Y, ROI_W, ROI_H = 630, 460, 73, 65
# Center of rotation within the ROI (used for visualization, actual calculation is transformation-based)
CENTER_X_ROI, CENTER_Y_ROI = 35, 40

# HSV range for chalk color
CHALK_HSV_LOWER = np.array([0, 0, 148])
CHALK_HSV_UPPER = np.array([180, 253, 255])

# Area filters for detected contours
MIN_CHALK_AREA = 6
MAX_CHALK_AREA = 500

# Parameters for Optical Flow (Good Features to Track + PyrLK)
lk_params = dict( winSize  = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Parameters for Transformation Estimation (RANSAC)
# ransacReprojThreshold: Max distance for a point to be considered an inlier
RANSAC_REPROJ_THRESHOLD = 5.0
# Minimum number of points required for affine estimation (usually 3)
MIN_POINTS_FOR_TRANSFORMATION = 3

# --- Visualization Parameters ---
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
LINE_THICKNESS = 2
MARK_COLOR = (255, 0, 0)      # Blue for detected marks
CENTER_COLOR = (0, 0, 255)    # Red for the center point (fixed ROI center)
ROI_COLOR = (0, 255, 0)       # Green for the ROI rectangle
INFO_COLOR = (255, 255, 255)  # White for status text
TRACKED_LINE_COLOR = (0, 255, 255) # Yellow for lines showing tracked movement

# New parameters for virtual circle visualization (Still useful context)
VIRTUAL_CIRCLE_RADIUS = 50  # Radius of the drawn virtual protractor circle
VIRTUAL_CIRCLE_COLOR = (255, 255, 0) # Cyan for the virtual circle and markings
VIRTUAL_MARK_LINE_COLOR = (0, 255, 255) # Yellow for the line from center to mark
VIRTUAL_LINE_THICKNESS = 1    # Thickness for virtual lines
ANGLE_TEXT_COLOR = (255, 255, 255) # White for angle labels


# --- Helper Functions ---
def safe_divide(numerator, denominator):
    """Avoid division by zero."""
    return 0 if denominator == 0 else numerator / denominator

def get_mark_centroids(mask_roi, min_area, max_area):
    """Finds contours in the mask and returns their centroids within ROI coordinates."""
    contours, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_marks_roi = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                mark_x_roi = int(safe_divide(M["m10"], M["m00"]))
                mark_y_roi = int(safe_divide(M["m01"], M["m00"]))
                detected_marks_roi.append((mark_x_roi, mark_y_roi))
    return detected_marks_roi


# --- Main Processing ---
cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    print(f"Error: Could not open video file: {VIDEO_SOURCE}")
    sys.exit()

# --- Read FPS for Real-Time Playback ---
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30  # fallback if FPS is not available
wait_time = int(1000 / fps)

# Variables for tracking
previous_frame_gray = None
# previous_points_for_tracking will store points in (N, 1, 2) format for Optical Flow input
previous_points_for_tracking = np.array([], dtype=np.float32).reshape(-1, 1, 2)

# Variables for rotation measurement
accumulated_rotation_deg = 0.0
frame_rotation_deg = 0.0
num_tracked_points = 0
num_inlier_points = 0

# State variables
frame_count = 0
paused = False
processed_display_frame = None # Variable to store the frame when paused
last_processed_roi_mask = None # Store the last processed ROI and mask for display when paused

# Read the first frame to get dimensions and initialize tracking
ret, frame = cap.read()
if not ret:
    print("Error: Could not read the first frame.")
    cap.release()
    sys.exit()

frame_h, frame_w = frame.shape[:2]
previous_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

print(f"Video Frame Size: {frame_w}x{frame_h}, FPS: {fps}")


while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            # End of video, loop or break
            print("End of video reached. Restarting...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_count = 0 # Reset frame count if looping
            accumulated_rotation_deg = 0.0 # Reset rotation
            # Reset tracked points to empty (N, 1, 2) array
            previous_points_for_tracking = np.array([], dtype=np.float32).reshape(-1, 1, 2)
            previous_frame_gray = None # Force re-initialization on next frame read
            continue # Skip processing for this iteration

        frame_count += 1
        current_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Create a copy of the frame for drawing visualizations
        display_frame = frame.copy()

        # --- Processing ---

        # Extract the Region of Interest (ROI)
        roi = frame[ROI_Y:ROI_Y+ROI_H, ROI_X:ROI_X+ROI_W]

        current_detected_marks_roi = []
        mask = None # Initialize mask to None

        if roi.size != 0:
            # Convert ROI to HSV color space
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            # Create a binary mask using the chalk color range
            mask = cv2.inRange(hsv_roi, CHALK_HSV_LOWER, CHALK_HSV_UPPER)

            # --- Morphological Cleaning ---
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

            # Get centroids of detected marks in ROI coordinates
            current_detected_marks_roi = get_mark_centroids(mask, MIN_CHALK_AREA, MAX_CHALK_AREA)

            # Store ROI and Mask for potential display when paused
            last_processed_roi_mask = (roi.copy(), mask.copy())

        # Convert current ROI points to full frame coordinates, format for Optical Flow output/next input
        # It seems reshape(-1, 1, 2) is the standard for OpenCV point arrays
        current_points_full = np.array([[p[0] + ROI_X, p[1] + ROI_Y] for p in current_detected_marks_roi], dtype=np.float32).reshape(-1, 1, 2)


        # --- Optical Flow Tracking and Transformation Estimation ---
        frame_rotation_deg = 0.0 # Reset per-frame rotation
        num_tracked_points = 0
        num_inlier_points = 0
        M = None # Transformation matrix
        good_prev_inliers_2d = np.array([], dtype=np.float32).reshape(-1, 2) # Store inlier points for drawing (2D)
        good_curr_inliers_2d = np.array([], dtype=np.float32).reshape(-1, 2)

        # Need at least MIN_POINTS_FOR_TRANSFORMATION points to estimate affine transformation
        if previous_points_for_tracking.shape[0] >= MIN_POINTS_FOR_TRANSFORMATION and current_points_full.shape[0] >= MIN_POINTS_FOR_TRANSFORMATION:
            # Use Optical Flow to track previous points in the current frame
            next_points, status, error = cv2.calcOpticalFlowPyrLK(
                previous_frame_gray,
                current_frame_gray,
                previous_points_for_tracking,
                None,
                **lk_params
            )

            # Select only the points that were successfully tracked
            # good_prev and good_curr will inherit the (N, 1, 2) shape
            good_prev = previous_points_for_tracking[status == 1]
            good_curr = next_points[status == 1]
            num_tracked_points = good_prev.shape[0]

            if good_prev.shape[0] >= MIN_POINTS_FOR_TRANSFORMATION:
                 # Estimate the affine transformation between the successfully tracked points
                 # estimateAffinePartial2D expects points in (N, 2) format
                 M, inliers = cv2.estimateAffinePartial2D(
                     good_prev.reshape(-1, 2), # Reshape (N', 1, 2) to (N', 2) for estimation
                     good_curr.reshape(-1, 2), # Reshape (N', 1, 2) to (N', 2) for estimation
                     method=cv2.RANSAC,
                     ransacReprojThreshold=RANSAC_REPROJ_THRESHOLD
                 )

                 if M is not None:
                     # Extract rotation from the transformation matrix M
                     frame_rotation_rad = math.atan2(M[0, 1], M[0, 0])
                     frame_rotation_deg = math.degrees(frame_rotation_rad)

                     # Accumulate the rotation
                     accumulated_rotation_deg += frame_rotation_deg

                     # Get the number of inliers and store the inlier points (reshaped to 2D)
                     if inliers is not None:
                          num_inlier_points = np.sum(inliers)
                          # Select the inlier points from the original (N', 1, 2) good_prev/good_curr arrays
                          # and reshape them to (N'', 2) for drawing
                          good_prev_inliers_3d = good_prev[inliers.ravel() == 1]
                          good_curr_inliers_3d = good_curr[inliers.ravel() == 1]
                          good_prev_inliers_2d = good_prev_inliers_3d.reshape(-1, 2)
                          good_curr_inliers_2d = good_curr_inliers_3d.reshape(-1, 2)


        # Update points for tracking in the next frame
        # Use the newly detected points as the basis for tracking in the next frame
        previous_points_for_tracking = current_points_full.copy()
        previous_frame_gray = current_frame_gray.copy() # Store current frame for next iteration


        # Store the processed frame for when paused
        processed_display_frame = display_frame.copy()

    else: # If paused, use the last processed frame and stored ROI/Mask
        display_frame = processed_display_frame.copy()
        # Retrieve last ROI/Mask if available
        if last_processed_roi_mask is not None:
             roi, mask = last_processed_roi_mask
        else:
             # If not processed yet, initialize blank or skip ROI/Mask display
             roi = np.zeros((ROI_H, ROI_W, 3), dtype=np.uint8)
             mask = np.zeros((ROI_H, ROI_W), dtype=np.uint8)
             last_processed_roi_mask = (roi, mask) # Store the blanks to avoid repeated check

        # If paused, we need the last known points to draw visualizations
        # The variables holding the *current* frame's data won't be updated in the paused block.
        # We would need to store current_detected_marks_roi, frame_rotation_deg, etc.
        # For simplicity in the paused state, we will show the visualizations
        # based on the *last processed frame's* data that was stored.
        # The state variables like frame_rotation_deg, accumulated_rotation_deg,
        # num_tracked_points, num_inlier_points retain their values from
        # the last unpaused frame.
        # The detected marks visualization needs points:
        # Let's assume we have 'current_detected_marks_roi' from the last processed frame.
        # This requires storing it. Add storage:
        # last_detected_marks_roi = [] # Initialize outside loop
        # ... Inside not paused block after getting centroids ...
        # last_detected_marks_roi = current_detected_marks_roi.copy()
        # ... Inside paused block ...
        # if 'last_detected_marks_roi' in locals():
        #      current_detected_marks_roi_for_display = last_detected_marks_roi
        #      # Need corresponding inlier points for drawing tracks too.
        #      # Store good_prev_inliers_2d and good_curr_inliers_2d as well.
        #      # last_good_prev_inliers_2d = good_prev_inliers_2d.copy()
        #      # last_good_curr_inliers_2d = good_curr_inliers_2d.copy()
        # else:
        #      current_detected_marks_roi_for_display = []
        #      # Initialize empty 2D arrays for inliers for drawing
        #      good_prev_inliers_2d = np.array([], dtype=np.float32).reshape(-1, 2)
        #      good_curr_inliers_2d = np.array([], dtype=np.float32).reshape(-1, 2)

        # To avoid adding more complex state variables just for paused drawing,
        # we'll simplify: visualizations derived directly from the *current* frame's
        # processing results (like `current_detected_marks_roi`, `good_prev_inliers_2d`)
        # will only be fully accurate in the *not paused* state. When paused,
        # these variables hold the values from the last processed frame, which is acceptable.

    # --- Visualization on the display_frame ---

    # Calculate the center of the ROI in full frame coordinates (for visualization)
    center_full_x = ROI_X + CENTER_X_ROI
    center_full_y = ROI_Y + CENTER_Y_ROI

    # Draw the ROI rectangle
    cv2.rectangle(display_frame, (ROI_X, ROI_Y), (ROI_X+ROI_W, ROI_Y+ROI_H), ROI_COLOR, LINE_THICKNESS)

    # Draw the fixed ROI center point
    cv2.circle(display_frame, (center_full_x, center_full_y), 5, CENTER_COLOR, -1)

    # Draw the currently detected mark circles
    # 'current_detected_marks_roi' should retain its value from the last processed frame if paused
    if 'current_detected_marks_roi' in locals() and current_detected_marks_roi:
         for mark_x_roi, mark_y_roi in current_detected_marks_roi:
             mark_full_x = ROI_X + mark_x_roi
             mark_full_y = ROI_Y + mark_y_roi
             cv2.circle(display_frame, (mark_full_x, mark_full_y), 4, MARK_COLOR, -1) # Draw the mark itself

             # Calculate and display angle for the detected mark relative to fixed center
             # This angle is just for visualization, the core rotation comes from affine matrix
             mark_angle_rad = math.atan2(mark_y_roi - CENTER_Y_ROI, mark_x_roi - CENTER_X_ROI)
             mark_angle_deg = math.degrees(mark_angle_rad)
             angle_text = f"{mark_angle_deg:.1f} deg"
             text_offset_distance = 15
             # Ensure text position is within bounds if needed
             text_x_offset = int(text_offset_distance * math.cos(mark_angle_rad))
             text_y_offset = int(text_offset_distance * math.sin(mark_angle_rad))
             text_pos = (mark_full_x + text_x_offset, mark_full_y + text_y_offset)
             cv2.putText(display_frame, angle_text, text_pos, FONT, FONT_SCALE * 0.5, ANGLE_TEXT_COLOR, 1)


    # Draw lines showing the tracked movement (if tracking happened)
    # These arrays should also retain their values from the last processed frame if paused
    if good_prev_inliers_2d.shape[0] > 0 and good_curr_inliers_2d.shape[0] > 0:
        # Draw a line for each inlier point showing its movement
        # Use the 2D arrays directly for indexing
        for i in range(good_prev_inliers_2d.shape[0]):
             # Access points using [i, 0] and [i, 1] for the (x, y) coordinates
             p1 = (int(good_prev_inliers_2d[i, 0]), int(good_prev_inliers_2d[i, 1]))
             p2 = (int(good_curr_inliers_2d[i, 0]), int(good_curr_inliers_2d[i, 1]))
             cv2.line(display_frame, p1, p2, TRACKED_LINE_COLOR, VIRTUAL_LINE_THICKNESS)
             # Also draw a small circle at the current (tracked) position
             cv2.circle(display_frame, p2, 3, TRACKED_LINE_COLOR, -1)


    # Draw the virtual protractor circle (relative to fixed center)
    cv2.circle(display_frame, (center_full_x, center_full_y), VIRTUAL_CIRCLE_RADIUS, VIRTUAL_CIRCLE_COLOR, VIRTUAL_LINE_THICKNESS)

    # Draw some protractor reference lines (e.g., every 45 degrees)
    for angle_deg in range(0, 360, 45):
        angle_rad = math.radians(angle_deg)
        x_end = int(center_full_x + VIRTUAL_CIRCLE_RADIUS * math.cos(angle_rad))
        y_end = int(center_full_y + VIRTUAL_CIRCLE_RADIUS * math.sin(angle_rad))
        start_x = int(center_full_x + (VIRTUAL_CIRCLE_RADIUS * 0.8) * math.cos(angle_rad))
        start_y = int(center_full_y + (VIRTUAL_CIRCLE_RADIUS * 0.8) * math.sin(angle_rad))
        cv2.line(display_frame, (start_x, start_y), (x_end, y_end), VIRTUAL_CIRCLE_COLOR, VIRTUAL_LINE_THICKNESS)


    # --- Add Status Text ---
    # These variables retain their value from the last processed frame if paused
    num_detected = len(current_detected_marks_roi) if 'current_detected_marks_roi' in locals() else 0
    status_line1 = f"Frame: {frame_count} {'(Paused)' if paused else ''}"
    status_line2 = f"Marks Detected: {num_detected}"
    status_line3 = f"Frame Rot: {frame_rotation_deg:.2f} deg"
    status_line4 = f"Accumulated Rot: {accumulated_rotation_deg:.2f} deg"
    status_line5 = f"Tracked Pts: {num_tracked_points}, Inliers: {num_inlier_points}"


    cv2.putText(display_frame, status_line1, (10, frame_h - 80), FONT, FONT_SCALE, INFO_COLOR, LINE_THICKNESS)
    cv2.putText(display_frame, status_line2, (10, frame_h - 60), FONT, FONT_SCALE, INFO_COLOR, LINE_THICKNESS)
    cv2.putText(display_frame, status_line3, (10, frame_h - 40), FONT, FONT_SCALE, INFO_COLOR, LINE_THICKNESS)
    cv2.putText(display_frame, status_line4, (10, frame_h - 20), FONT, FONT_SCALE, INFO_COLOR, LINE_THICKNESS)
    cv2.putText(display_frame, status_line5, (frame_w - 250, frame_h - 20), FONT, FONT_SCALE, INFO_COLOR, LINE_THICKNESS)


    # --- Show Frames ---
    cv2.imshow('Frame', display_frame)

    # Show ROI and Mask if available (either live or from last processed frame)
    # The 'roi' and 'mask' variables retain their value from the last processed frame if paused
    if last_processed_roi_mask is not None: # Check if anything has been processed yet
        roi_display, mask_display = last_processed_roi_mask
        if roi_display.size > 0: # Check if the stored ROI is valid (not the initial blank)
            mask_display_bgr = cv2.cvtColor(mask_display, cv2.COLOR_GRAY2BGR)
            cv2.imshow('ROI', roi_display)
            cv2.imshow('Mask', mask_display_bgr)
        # If roi_display.size is 0, it means no ROI was successfully processed yet (maybe video error or first frame had no ROI),
        # in which case we don't show the ROI/Mask windows until a valid one exists.


    # --- Handle User Input ---
    key = cv2.waitKey(wait_time) & 0xFF
    if key == ord('q'):
        print("Exit requested.")
        break
    elif key == ord('p'):
        paused = not paused
        print("Paused." if paused else "Resumed.")


# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
print("Processing finished.")