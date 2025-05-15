import cv2
import numpy as np
import math
import time

# --- Initial Configuration Values (will be controlled by sliders) ---
# These are starting points for the sliders.
# It's still good to have reasonable initial estimates.
INITIAL_CENTER_X = 320
INITIAL_CENTER_Y = 240
INITIAL_LOCUS_RADIUS = 100
INITIAL_ROI_INNER_RADIUS = 80
INITIAL_ROI_OUTER_RADIUS = 120
INITIAL_BINARY_THRESHOLD = 180
INITIAL_MIN_CONTOUR_AREA = 15
INITIAL_MAX_CONTOUR_AREA = 500
INITIAL_LOCUS_TOLERANCE = 20

# Global to store previous angle for calculation continuity
previous_angle_deg_global = None
# For FPS calculation
prev_frame_time_global = 0
new_frame_time_global = 0

def calculate_angle(p1_x, p1_y, p2_x, p2_y):
    """Calculates angle in degrees between two points (p2 relative to p1)."""
    angle_rad = math.atan2(p2_y - p1_y, p2_x - p1_x)
    angle_deg = math.degrees(angle_rad)
    return angle_deg

def process_frame(frame, frame_count, params):
    global previous_angle_deg_global, prev_frame_time_global, new_frame_time_global

    if frame is None:
        print("Error: Received an empty frame.")
        return None, None

    output_frame = frame.copy()

    # Unpack parameters
    center_x = params['center_x']
    center_y = params['center_y']
    locus_radius = params['locus_radius']
    roi_inner_radius = params['roi_inner_radius']
    roi_outer_radius = params['roi_outer_radius']
    binary_threshold = params['binary_threshold']
    min_contour_area = params['min_contour_area']
    max_contour_area = params['max_contour_area']
    locus_tolerance = params['locus_tolerance']

    # --- FPS Calculation ---
    new_frame_time_global = time.time()
    fps = 1/(new_frame_time_global-prev_frame_time_global) if (new_frame_time_global-prev_frame_time_global) > 0 else 0
    prev_frame_time_global = new_frame_time_global
    cv2.putText(output_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 0), 2, cv2.LINE_AA)

    # 1. Preprocessing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2. Thresholding
    _, thresh = cv2.threshold(blurred, binary_threshold, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((3,3),np.uint8)
    thresh_cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    thresh_cleaned = cv2.morphologyEx(thresh_cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    cv2.imshow("Thresholded", thresh_cleaned)

    # 3. Find Contours
    contours, _ = cv2.findContours(thresh_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_marks = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_contour_area < area < max_contour_area:
            M = cv2.moments(cnt)
            if M["m00"] == 0: continue
            
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            dist_from_center = math.sqrt((cx - center_x)**2 + (cy - center_y)**2)

            if roi_inner_radius < dist_from_center < roi_outer_radius and \
               abs(dist_from_center - locus_radius) < locus_tolerance:
                valid_marks.append({
                    'cx': cx, 'cy': cy, 'contour': cnt, 'area': area,
                    'dist_from_center': dist_from_center
                })
                cv2.drawContours(output_frame, [cnt], -1, (0, 255, 0), 2)
                cv2.circle(output_frame, (cx, cy), 3, (0, 0, 255), -1)

    current_angle_deg = None
    delta_angle = None

    if valid_marks:
        best_mark = min(valid_marks, key=lambda m: abs(m['dist_from_center'] - locus_radius))
        mark_cx, mark_cy = best_mark['cx'], best_mark['cy']
        cv2.circle(output_frame, (mark_cx, mark_cy), 7, (255, 0, 255), 2)

        current_angle_deg = calculate_angle(center_x, center_y, mark_cx, mark_cy)
        
        cv2.line(output_frame, (center_x, center_y), (mark_cx, mark_cy), (255, 255, 0), 1)
        cv2.putText(output_frame, f"{current_angle_deg:.1f}deg", (mark_cx + 10, mark_cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        if previous_angle_deg_global is not None:
            delta_angle = current_angle_deg - previous_angle_deg_global
            if delta_angle > 180: delta_angle -= 360
            elif delta_angle < -180: delta_angle += 360
            
            # print(f"Frame: {frame_count}, Angle: {current_angle_deg:.2f}, Prev: {previous_angle_deg_global:.2f}, Delta: {delta_angle:.2f}")
            cv2.putText(output_frame, f"Delta: {delta_angle:.1f}deg", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(output_frame, f"Angle: {current_angle_deg:.1f}deg", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

        else:
            # print(f"Frame: {frame_count}, Angle: {current_angle_deg:.2f} (First detection)")
            cv2.putText(output_frame, f"Angle: {current_angle_deg:.1f}deg", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

        previous_angle_deg_global = current_angle_deg
    else:
        # print(f"Frame: {frame_count}, No valid chalk mark found.")
        # previous_angle_deg_global = None # Optional: Reset if mark is lost
        pass


    # --- Draw ROI and Locus for visualization ---
    cv2.circle(output_frame, (center_x, center_y), int(roi_inner_radius), (0, 0, 255), 1)
    cv2.circle(output_frame, (center_x, center_y), int(roi_outer_radius), (0, 0, 255), 1)
    cv2.circle(output_frame, (center_x, center_y), int(locus_radius), (0, 255, 255), 1)
    cv2.circle(output_frame, (center_x, center_y), 3, (0,255,0), -1)

    return output_frame, delta_angle

def nothing(x):
    """Dummy callback function for createTrackbar."""
    pass

# --- Main Execution ---
if __name__ == "__main__":
    video_source = "fronti.mp4" # Or 0 for webcam

    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"Error: Could not open video source: {video_source}")
        exit()

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video Resolution: {width}x{height}.")
    
    # Update initial values if they are placeholders or depend on resolution
    # For example, if CENTER_X/Y were initially set to width/2, height/2
    # INITIAL_CENTER_X = width // 2
    # INITIAL_CENTER_Y = height // 2


    # --- Create Trackbars ---
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 400, 500) # Adjust size as needed

    cv2.createTrackbar("Center X", "Trackbars", INITIAL_CENTER_X, width, nothing)
    cv2.createTrackbar("Center Y", "Trackbars", INITIAL_CENTER_Y, height, nothing)
    cv2.createTrackbar("Locus R", "Trackbars", INITIAL_LOCUS_RADIUS, width // 2, nothing) # Max radius can't exceed half width/height
    cv2.createTrackbar("ROI In R", "Trackbars", INITIAL_ROI_INNER_RADIUS, width // 2, nothing)
    cv2.createTrackbar("ROI Out R", "Trackbars", INITIAL_ROI_OUTER_RADIUS, width // 2, nothing)
    cv2.createTrackbar("Threshold", "Trackbars", INITIAL_BINARY_THRESHOLD, 255, nothing)
    cv2.createTrackbar("Min Area", "Trackbars", INITIAL_MIN_CONTOUR_AREA, 1000, nothing) # Adjust max area as needed
    cv2.createTrackbar("Max Area", "Trackbars", INITIAL_MAX_CONTOUR_AREA, 5000, nothing)
    cv2.createTrackbar("Locus Tol", "Trackbars", INITIAL_LOCUS_TOLERANCE, 100, nothing) # Max tolerance

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break
        
        frame_count += 1

        # --- Read values from Trackbars ---
        current_params = {
            'center_x': cv2.getTrackbarPos("Center X", "Trackbars"),
            'center_y': cv2.getTrackbarPos("Center Y", "Trackbars"),
            'locus_radius': cv2.getTrackbarPos("Locus R", "Trackbars"),
            'roi_inner_radius': cv2.getTrackbarPos("ROI In R", "Trackbars"),
            'roi_outer_radius': cv2.getTrackbarPos("ROI Out R", "Trackbars"),
            'binary_threshold': cv2.getTrackbarPos("Threshold", "Trackbars"),
            'min_contour_area': cv2.getTrackbarPos("Min Area", "Trackbars"),
            'max_contour_area': cv2.getTrackbarPos("Max Area", "Trackbars"),
            'locus_tolerance': cv2.getTrackbarPos("Locus Tol", "Trackbars")
        }
        
        # Ensure ROI inner radius is not greater than outer radius from sliders
        if current_params['roi_inner_radius'] > current_params['roi_outer_radius']:
            current_params['roi_inner_radius'] = current_params['roi_outer_radius'] -1 # or set to 0
            # Optionally, you could update the trackbar position itself here,
            # but reading it fresh next iteration usually handles it.
            # cv2.setTrackbarPos("ROI In R", "Trackbars", current_params['roi_inner_radius'])


        processed_image, delta = process_frame(frame, frame_count, current_params)

        if processed_image is not None:
            cv2.imshow("Live Wheel Tracking", processed_image)
        else:
            cv2.imshow("Live Wheel Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()