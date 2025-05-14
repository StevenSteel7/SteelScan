import cv2
import numpy as np
import math
import sys

# --- Parameters ---
VIDEO_SOURCE = 'fronti.mp4'

# ROI parameters
ROI_X, ROI_Y, ROI_W, ROI_H = 630, 460, 73, 65
# Center of rotation within the ROI
CENTER_X_ROI, CENTER_Y_ROI = 35, 40

# HSV range for chalk color
CHALK_HSV_LOWER = np.array([0, 0, 148])
CHALK_HSV_UPPER = np.array([180, 253, 255])

# Area filters for detected contours
MIN_CHALK_AREA = 6
MAX_CHALK_AREA = 500

# Distance threshold for matching marks between frames
DISTANCE_THRESHOLD = 15

# --- Visualization Parameters ---
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
LINE_THICKNESS = 2
MARK_COLOR = (255, 0, 0)      # Blue for detected marks
CENTER_COLOR = (0, 0, 255)    # Red for the center point
ROI_COLOR = (0, 255, 0)       # Green for the ROI rectangle
INFO_COLOR = (255, 255, 255)  # White for status text

# New parameters for virtual circle visualization
VIRTUAL_CIRCLE_RADIUS = 50  # Radius of the drawn virtual protractor circle
VIRTUAL_CIRCLE_COLOR = (255, 255, 0) # Cyan for the virtual circle and markings
VIRTUAL_MARK_LINE_COLOR = (0, 255, 255) # Yellow for the line from center to mark
VIRTUAL_LINE_THICKNESS = 1    # Thickness for virtual lines
ANGLE_TEXT_COLOR = (255, 255, 255) # White for angle labels


# --- Helper Functions ---
def safe_divide(numerator, denominator):
    """Avoid division by zero."""
    return 0 if denominator == 0 else numerator / denominator

def calculate_angular_difference(angle1_deg, angle2_deg):
    """Calculates the shortest signed angular difference between two angles in degrees."""
    diff = angle1_deg - angle2_deg
    # Normalize difference to be within (-180, 180]
    while diff <= -180:
        diff += 360
    while diff > 180:
        diff -= 360
    return diff

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

previous_detected_marks = []
frame_count = 0
paused = False
processed_display_frame = None # Variable to store the frame when paused

# Read the first frame to get dimensions
ret, first_frame = cap.read()
if not ret:
    print("Error: Could not read the first frame.")
    cap.release()
    sys.exit()

frame_h, frame_w = first_frame.shape[:2]
# Reset video to the beginning
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

print(f"Video Frame Size: {frame_w}x{frame_h}, FPS: {fps}")

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            # End of video, loop or break
            print("End of video reached. Restarting...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_count = 0 # Reset frame count if looping
            previous_detected_marks = [] # Reset previous marks
            continue

        frame_count += 1
        # Create a copy of the frame for drawing visualizations
        display_frame = frame.copy()
    else:
        # If paused, use the last processed frame for display
        display_frame = processed_display_frame.copy()

    # --- Processing happens only when not paused ---
    if not paused:
        # Extract the Region of Interest (ROI)
        roi = frame[ROI_Y:ROI_Y+ROI_H, ROI_X:ROI_X+ROI_W]
        if roi.size == 0:
            print(f"Warning: ROI empty at frame {frame_count}. Skipping processing for this frame.")
            # Still display the previous frame if available, or a blank frame
            if processed_display_frame is None:
                 display_frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
            else:
                 display_frame = processed_display_frame.copy()
            # Need to skip the rest of the processing block
            # continue # Skipping means the status text won't update, better to just display prev frame
            # Fall through to visualization/display of the paused frame
            pass # Do nothing and let the paused block handle display

        else: # Process the ROI only if it's valid
            # Convert ROI to HSV color space
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            # Create a binary mask using the chalk color range
            mask = cv2.inRange(hsv_roi, CHALK_HSV_LOWER, CHALK_HSV_UPPER)

            # --- Morphological Cleaning ---
            # Use closing then opening to remove noise and connect components
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

            # Contour Detection on the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Sort contours by area (largest first) and select top 4
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:4]

            detected_marks_roi = [] # List to store (x, y) coordinates of detected mark centers within ROI
            for cnt in contours:
                area = cv2.contourArea(cnt)
                # Filter contours based on area
                if MIN_CHALK_AREA < area < MAX_CHALK_AREA:
                    M = cv2.moments(cnt)
                    # Calculate the center of the contour
                    if M["m00"] != 0:
                        mark_x_roi = int(safe_divide(M["m10"], M["m00"]))
                        mark_y_roi = int(safe_divide(M["m01"], M["m00"]))
                        detected_marks_roi.append((mark_x_roi, mark_y_roi))

            # --- Matching detected marks with previous frame's marks ---
            # This is done to track the movement of individual marks if needed,
            # and more importantly, to calculate the average rotation delta.
            average_delta = 0.0
            num_matches = 0 # How many marks were successfully matched to calculate delta

            if previous_detected_marks and detected_marks_roi:
                matched = [] # List of tuples: (current_mark_roi, previous_mark_roi)
                # Try to find a close previous mark for each current mark
                for curr_mark_roi in detected_marks_roi:
                    best_distance = float('inf')
                    best_prev_roi = None
                    for prev_mark_roi in previous_detected_marks:
                        dist = math.hypot(curr_mark_roi[0] - prev_mark_roi[0], curr_mark_roi[1] - prev_mark_roi[1])
                        if dist < best_distance:
                            best_distance = dist
                            best_prev_roi = prev_mark_roi

                    # If the closest previous mark is within the distance threshold, consider it a match
                    if best_prev_roi is not None and best_distance < DISTANCE_THRESHOLD:
                         # Add the match and remove the previous mark from consideration
                         # (simple approach, doesn't handle multiple current marks matching one previous well)
                         # A more robust approach might use the Hungarian algorithm or similar.
                         # For now, this simple closest-match-with-threshold is sufficient for delta calculation.
                         matched.append((curr_mark_roi, best_prev_roi))
                         # Optional: remove best_prev_roi from previous_detected_marks to prevent double matching
                         # This simple implementation doesn't remove, allowing multiple curr marks to potentially match one prev.
                         # For delta average, this is okay as long as the match is valid.

                if matched:
                    angle_deltas = [] # List of angular differences for matched pairs
                    # Calculate angle delta for each matched pair
                    for (curr, prev) in matched:
                        # Calculate angle for current and previous marks relative to ROI center
                        # atan2(y, x) gives angle in radians, converts to degrees.
                        # Note: In image coordinates, positive y is downwards. atan2 handles this correctly.
                        curr_angle = math.degrees(math.atan2(curr[1] - CENTER_Y_ROI, curr[0] - CENTER_X_ROI))
                        prev_angle = math.degrees(math.atan2(prev[1] - CENTER_Y_ROI, prev[0] - CENTER_X_ROI))
                        # Calculate the difference, handling the -180/180 wrap-around
                        delta = calculate_angular_difference(curr_angle, prev_angle)
                        angle_deltas.append(delta)

                    if angle_deltas:
                        # Calculate the average rotation delta across all matched marks
                        average_delta = sum(angle_deltas) / len(angle_deltas)
                        num_matches = len(angle_deltas) # Number of pairs used for average calculation

            # Update the list of marks from the previous frame for the next iteration
            previous_detected_marks = detected_marks_roi.copy()

            # --- Visualization on the display_frame ---

            # Calculate the center of the ROI in full frame coordinates
            center_full_x = ROI_X + CENTER_X_ROI
            center_full_y = ROI_Y + CENTER_Y_ROI

            # 1. Draw the virtual protractor circle
            cv2.circle(display_frame, (center_full_x, center_full_y), VIRTUAL_CIRCLE_RADIUS, VIRTUAL_CIRCLE_COLOR, VIRTUAL_LINE_THICKNESS)

            # 2. Draw some protractor reference lines (e.g., every 45 degrees)
            for angle_deg in range(0, 360, 45):
                angle_rad = math.radians(angle_deg)
                # Calculate endpoint on the circle for the given angle and radius
                # Note: math.cos and math.sin assume standard mathematical coordinates (y up is positive).
                # Since our angle is calculated using atan2 which considers positive y as DOWN (image coords),
                # we use the angle directly, and the endpoint calculation maps correctly to image pixels.
                x_end = int(center_full_x + VIRTUAL_CIRCLE_RADIUS * math.cos(angle_rad))
                y_end = int(center_full_y + VIRTUAL_CIRCLE_RADIUS * math.sin(angle_rad))
                # Draw a small line segment from near the center out to the circle edge
                # To make it look more like markings rather than full lines to center
                start_x = int(center_full_x + (VIRTUAL_CIRCLE_RADIUS * 0.8) * math.cos(angle_rad)) # Start closer to edge
                start_y = int(center_full_y + (VIRTUAL_CIRCLE_RADIUS * 0.8) * math.sin(angle_rad)) # Start closer to edge

                cv2.line(display_frame, (start_x, start_y), (x_end, y_end), VIRTUAL_CIRCLE_COLOR, VIRTUAL_LINE_THICKNESS)

            # 3. Draw lines from the center to each detected mark and label the angle
            for mark_x_roi, mark_y_roi in detected_marks_roi:
                # Convert mark coordinates from ROI to full frame
                mark_full_x = ROI_X + mark_x_roi
                mark_full_y = ROI_Y + mark_y_roi

                # Draw the line from the center to the mark
                cv2.line(display_frame, (center_full_x, center_full_y), (mark_full_x, mark_full_y), VIRTUAL_MARK_LINE_COLOR, VIRTUAL_LINE_THICKNESS)

                # Calculate the angle of the mark relative to the center
                current_mark_angle_rad = math.atan2(mark_y_roi - CENTER_Y_ROI, mark_x_roi - CENTER_X_ROI)
                current_mark_angle_deg = math.degrees(current_mark_angle_rad)

                # Optional: Convert angle to 0-360 range if preferred for display
                # current_mark_angle_deg_360 = (current_mark_angle_deg + 360) % 360

                # Prepare angle text
                angle_text = f"{current_mark_angle_deg:.1f} deg"

                # Position the text slightly away from the mark
                # Calculate offset vector based on the angle to place text radially
                text_offset_distance = 15 # How far from the mark the text should be
                text_x_offset = int(text_offset_distance * math.cos(current_mark_angle_rad))
                text_y_offset = int(text_offset_distance * math.sin(current_mark_angle_rad))
                text_pos = (mark_full_x + text_x_offset, mark_full_y + text_y_offset)

                # Draw the angle text
                cv2.putText(display_frame, angle_text, text_pos, FONT, FONT_SCALE * 0.5, ANGLE_TEXT_COLOR, 1)


            # Draw the ROI rectangle on the display frame
            cv2.rectangle(display_frame, (ROI_X, ROI_Y), (ROI_X+ROI_W, ROI_Y+ROI_H), ROI_COLOR, LINE_THICKNESS)

            # Draw the center point circle on the display frame
            cv2.circle(display_frame, (center_full_x, center_full_y), 5, CENTER_COLOR, -1)

            # Draw the detected mark circles on the display frame (these were already there)
            for mark_x_roi, mark_y_roi in detected_marks_roi:
                mark_full_x = ROI_X + mark_x_roi
                mark_full_y = ROI_Y + mark_y_roi
                cv2.circle(display_frame, (mark_full_x, mark_full_y), 4, MARK_COLOR, -1) # Draw the mark itself

            # --- Add Status Text ---
            num_detected = len(detected_marks_roi)
            status_line1 = f"Frame: {frame_count}"
            status_line2 = f"Marks Detected: {num_detected}"
            status_line3 = f"Avg Rot: {average_delta:.1f} deg/fr ({num_matches} matched)"

            cv2.putText(display_frame, status_line1, (10, frame_h - 60), FONT, FONT_SCALE, INFO_COLOR, LINE_THICKNESS)
            cv2.putText(display_frame, status_line2, (10, frame_h - 40), FONT, FONT_SCALE, INFO_COLOR, LINE_THICKNESS)
            cv2.putText(display_frame, status_line3, (10, frame_h - 20), FONT, FONT_SCALE, INFO_COLOR, LINE_THICKNESS)

            # Store the processed frame for when paused
            processed_display_frame = display_frame.copy()

    # --- Show Frames ---
    # The display_frame is either the newly processed one or the stored one if paused
    cv2.imshow('Frame', display_frame)

    # Show ROI and Mask only when not paused, or maybe show the last ones?
    # Let's show the last processed ROI and Mask when paused for context.
    if processed_display_frame is not None: # Only show if at least one frame was processed
        if roi.size > 0: # Only show if the last ROI was valid
             # Need to re-create the ROI and mask display if paused
             # This requires keeping the last valid roi and mask. Let's modify the code to store them.
             # Alternative: Only show these windows when not paused. This is simpler.
             if not paused:
                 roi_display = roi.copy()
                 mask_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                 # Ensure dimensions match for stacking or resize
                 # Resize for consistent window size
                 target_w, target_h = 400, 300
                 roi_resized = cv2.resize(roi_display, (target_w, target_h))
                 mask_resized = cv2.resize(mask_display, (target_w, target_h))
                 stacked = np.hstack((roi_resized, mask_resized))
                 cv2.imshow('ROI + Mask', stacked)
             # else: when paused, the windows are not updated, showing the last non-paused state.
             # This is probably acceptable.

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