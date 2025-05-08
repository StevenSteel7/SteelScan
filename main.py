import cv2
import numpy as np
import time

# --- Parameters to Tune ---

VIDEO_SOURCE = "front.mp4"  # <<<--- PUT YOUR VIDEO FILE PATH HERE

# Define Regions of Interest (ROIs) for the wheels (x, y, width, height)
# YOU MUST ADJUST THESE COORDINATES FOR YOUR VIDEO RESOLUTION AND WHEEL POSITIONS
# ROI[0]: 1st wheel from the right
# ROI[1]: 3rd wheel from the right
# Example coordinates (assuming a hypothetical video resolution, ADJUST THESE!):
ROIS = [
    (467, 428, 131, 121),  # ROI for Wheel 1 (Rightmost)
    (760, 424, 140, 133)   # ROI for Wheel 3 (from Right)
]
ROI_NAMES = ["Wheel 1 (Right)", "Wheel 3 (Right)"]

# --- Optical Flow Parameters ---
# Parameters for ShiTomasi corner detection (finding features)
feature_params = dict( maxCorners = 100,      # Max number of corners to detect per ROI
                       qualityLevel = 0.3,   # Minimal accepted quality of image corners (0-1)
                       minDistance = 7,      # Minimum possible Euclidean distance between corners
                       blockSize = 7 )       # Size of an average block for computing derivatives

# Parameters for Lucas-Kanade optical flow
lk_params = dict( winSize  = (15, 15),    # Size of the search window at each pyramid level
                  maxLevel = 2,          # 0: use original image only, >0: use pyramids
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)) # Termination criteria

MOTION_THRESHOLD = 1.5  # <<<--- TUNE THIS THRESHOLD (pixels) - Minimum average movement to detect motion
MIN_FEATURES_FOR_MOTION = 5 # Minimum number of tracked features required to reliably detect motion

# --- Initialization ---
cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    print(f"Error: Could not open video source: {VIDEO_SOURCE}")
    exit()

# Read the first frame
ret, old_frame = cap.read()
if not ret:
    print("Error: Could not read the first frame.")
    cap.release()
    exit()

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# --- Data structures to store points for each ROI ---
# List of NumPy arrays, one for each ROI's feature points
p0_list = []
# Detect initial features in each ROI
for i, (x, y, w, h) in enumerate(ROIS):
    roi_gray = old_gray[y:y+h, x:x+w]
    p0 = cv2.goodFeaturesToTrack(roi_gray, mask = None, **feature_params)
    if p0 is not None:
        # Offset points to be relative to the full frame
        p0[:, 0, 0] += x
        p0[:, 0, 1] += y
        print(f"Detected {len(p0)} initial features in {ROI_NAMES[i]}.")
    else:
        print(f"Warning: No initial features found in {ROI_NAMES[i]}.")
    p0_list.append(p0) # Add even if None, to keep indices aligned

# Create a mask image for drawing purposes (optional)
mask = np.zeros_like(old_frame)
colors = [(0, 255, 0), (0, 0, 255)] # Colors for drawing points/boxes

frame_count = 0
start_time = time.time()

# --- Main Loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video reached or error reading frame.")
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    vis_frame = frame.copy() # Create a copy for drawing visuals

    motion_detected_flags = [False] * len(ROIS)

    # Process each ROI
    for i, (x, y, w, h) in enumerate(ROIS):
        p0 = p0_list[i] # Get points tracked from the previous frame for this ROI

        # --- Check if we have points to track ---
        if p0 is not None and p0.shape[0] > 0:
            # Calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

            # Select good points (where flow was found)
            if p1 is not None and st is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]

                # --- Calculate Movement ---
                if good_new.shape[0] >= MIN_FEATURES_FOR_MOTION:
                    # Calculate Euclidean distance for each point pair
                    distances = np.linalg.norm(good_new - good_old, axis=1)
                    # Use median distance - more robust to outliers than mean
                    median_displacement = np.median(distances)

                    # --- Detect Motion ---
                    if median_displacement > MOTION_THRESHOLD:
                        motion_detected_flags[i] = True
                        print(f"Frame {frame_count}: MOTION DETECTED in {ROI_NAMES[i]} (Median Disp: {median_displacement:.2f})")

                    # --- Draw Tracks (Optional) ---
                    for j, (new, old) in enumerate(zip(good_new, good_old)):
                        a, b = new.ravel().astype(int)
                        c, d = old.ravel().astype(int)
                        # Draw the track line
                        # mask = cv2.line(mask, (a, b), (c, d), colors[i], 1)
                        # Draw the current point
                        vis_frame = cv2.circle(vis_frame, (a, b), 4, colors[i], -1)

                    # Update the points for the next frame (only keep good ones)
                    p0_list[i] = good_new.reshape(-1, 1, 2)

                else: # Not enough features tracked reliably
                    print(f"Frame {frame_count}: Not enough features ({good_new.shape[0]}) tracked for {ROI_NAMES[i]}. Resetting.")
                    p0_list[i] = None # Reset points for this ROI

            else: # Optical flow failed completely for this ROI
                 print(f"Frame {frame_count}: Optical flow failed for {ROI_NAMES[i]}. Resetting.")
                 p0_list[i] = None # Reset points

        # --- Re-detect features if needed ---
        if p0_list[i] is None or p0_list[i].shape[0] < feature_params['maxCorners'] // 2:
            # If points were lost or never found, try detecting again in the current frame's ROI
            roi_gray = frame_gray[y:y+h, x:x+w]
            p0_new = cv2.goodFeaturesToTrack(roi_gray, mask=None, **feature_params)
            if p0_new is not None:
                # Offset points to be relative to the full frame
                p0_new[:, 0, 0] += x
                p0_new[:, 0, 1] += y
                p0_list[i] = p0_new
                # print(f"Frame {frame_count}: Re-detected {len(p0_new)} features in {ROI_NAMES[i]}.")
            else:
                p0_list[i] = None # Still no features found


    # --- Display Output ---
    # Draw ROIs
    for i, (x, y, w, h) in enumerate(ROIS):
         color = colors[i]
         cv2.rectangle(vis_frame, (x, y), (x + w, y + h), color, 2)
         status_text = f"{ROI_NAMES[i]}: {'MOVING' if motion_detected_flags[i] else 'Stopped'}"
         cv2.putText(vis_frame, status_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


    # Combine frame with mask (if drawing lines)
    # img = cv2.add(vis_frame, mask) # Uncomment if drawing track lines on mask
    img = vis_frame # Use this if only drawing circles

    cv2.imshow('Frame - Wheel Motion Detection', img)

    # --- Update state for next iteration ---
    old_gray = frame_gray.copy()
    frame_count += 1

    # Exit condition
    key = cv2.waitKey(30) & 0xFF # Wait ~30ms (adjust for video speed)
    if key == ord('q'):
        print("Exit requested by user.")
        break
    elif key == ord('p'):
        print("Paused. Press any key to continue...")
        cv2.waitKey(-1) # Wait indefinitely until a key is pressed


# --- Cleanup ---
end_time = time.time()
print(f"\nProcessed {frame_count} frames in {end_time - start_time:.2f} seconds.")
cap.release()
cv2.destroyAllWindows()