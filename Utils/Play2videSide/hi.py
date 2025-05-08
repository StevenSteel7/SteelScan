import cv2
import numpy as np
import sys
import os

# --- Configuration ---
video1_filename = "video1.mp4"  # <<< CHANGE THIS to your first video file
video2_filename = "video2.mp4"  # <<< CHANGE THIS to your second video file

# --- Display Configuration ---
# Set maximum dimensions for the combined video window
# Adjust these based on your screen resolution if needed
MAX_DISPLAY_WIDTH = 1800
MAX_DISPLAY_HEIGHT = 900
window_title = "Side-by-Side Video Playback (Scaled to Fit)"
# -------------------------

# --- Helper Function for Resizing with Aspect Ratio ---
def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image # No resize needed
    if width is None: # Calculate width based on desired height
        ratio = height / float(h)
        dim = (int(w * ratio), height)
    else: # Calculate height based on desired width
        ratio = width / float(w)
        dim = (width, int(h * ratio))
    return cv2.resize(image, dim, interpolation=inter)
# ----------------------------------------------------

# Get the absolute paths to the video files
script_dir = os.path.dirname(os.path.abspath(__file__))
video1_path = os.path.join(script_dir, video1_filename)
video2_path = os.path.join(script_dir, video2_filename)

# Check if files exist
if not os.path.exists(video1_path):
    print(f"Error: Video file not found at {video1_path}")
    sys.exit(1)
if not os.path.exists(video2_path):
    print(f"Error: Video file not found at {video2_path}")
    sys.exit(1)

# Create VideoCapture objects
cap1 = cv2.VideoCapture(video1_path)
cap2 = cv2.VideoCapture(video2_path)

# Check if videos opened successfully
if not cap1.isOpened():
    print(f"Error: Could not open video 1: {video1_path}")
    sys.exit(1)
if not cap2.isOpened():
    print(f"Error: Could not open video 2: {video2_path}")
    sys.exit(1)

print(f"Playing '{video1_filename}' and '{video2_filename}' side-by-side, scaled to fit.")
print(f"Max window size: {MAX_DISPLAY_WIDTH}x{MAX_DISPLAY_HEIGHT}")
print(f"Press 'q' in the '{window_title}' window to quit.")

# Read first frames to potentially determine dimensions
ret1, frame1 = cap1.read()
ret2, frame2 = cap2.read()

if not ret1:
    print(f"Error: Could not read first frame from {video1_filename}")
    cap1.release()
    cap2.release()
    sys.exit(1)
if not ret2:
    print(f"Error: Could not read first frame from {video2_filename}")
    cap1.release()
    cap2.release()
    sys.exit(1)


# Main loop: Read, combine, scale, and display frames
while True: # Loop continues as long as we have frames or user hasn't quit

    # Check if either video stream ended in the *previous* iteration
    if not ret1 or not ret2:
        print("One or both videos finished.")
        break

    # --- Step 1: Ensure frames have the same height ---
    h1, w1 = frame1.shape[:2]
    h2, w2 = frame2.shape[:2]

    if h1 != h2:
        # Resize frame2 to match frame1's height, keeping aspect ratio
        frame2_resized = resize_with_aspect_ratio(frame2, height=h1)
    else:
        frame2_resized = frame2 # Heights match, no resize needed yet

    # --- Step 2: Combine frames horizontally ---
    try:
        combined_frame = np.hstack((frame1, frame2_resized))
    except ValueError as e:
        print(f"Error stacking frames: {e}")
        print(f"Frame 1 shape: {frame1.shape}, Frame 2 resized shape: {frame2_resized.shape}")
        break # Exit if stacking fails (e.g., unexpected dimension mismatch)

    # --- Step 3: Scale the combined frame to fit max display size ---
    h_comb, w_comb = combined_frame.shape[:2]
    scale_factor = 1.0 # Default: no scaling

    # Calculate scale factor needed for width
    if w_comb > MAX_DISPLAY_WIDTH:
        scale_factor = min(scale_factor, MAX_DISPLAY_WIDTH / w_comb)

    # Calculate scale factor needed for height
    if h_comb > MAX_DISPLAY_HEIGHT:
        scale_factor = min(scale_factor, MAX_DISPLAY_HEIGHT / h_comb)

    # Apply scaling only if needed (scale_factor < 1.0)
    if scale_factor < 1.0:
        new_w = int(w_comb * scale_factor)
        new_h = int(h_comb * scale_factor)
        final_frame = cv2.resize(combined_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        final_frame = combined_frame # No scaling needed

    # --- Step 4: Display the final frame ---
    cv2.imshow(window_title, final_frame)

    # --- Step 5: Wait for key press and check for 'q' ---
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quitting...")
        break

    # --- Step 6: Read the next frames for the next iteration ---
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

# --- Cleanup ---
print("Playback finished or quit.")
cap1.release()
cap2.release()
cv2.destroyAllWindows()