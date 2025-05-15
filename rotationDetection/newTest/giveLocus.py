import cv2
import numpy as np
import math
import time
import json

CONFIG_FILE_PATH = "wheel_config.json"
TARGET_FPS = 25.0  # Your desired playback FPS
TARGET_FRAME_DURATION = 1.0 / TARGET_FPS

# --- Initial Configuration Values (used if config file doesn't exist or for first save) ---
current_config_params = {
    'center_x': 320,
    'center_y': 240,
    'semi_major_a': 100,
    'semi_minor_b': 70,
    'ellipse_angle_deg': 0,
    'ellipse_eq_tolerance_percent': 20,
    'roi_inner_radius': 50,
    'roi_outer_radius': 150,
    'binary_threshold': 180,
    'min_contour_area': 15,
    'max_contour_area': 500,
}
TRACKBAR_PARAM_MAP = {
    "Center X": "center_x", "Center Y": "center_y", "SemiMajorA": "semi_major_a",
    "SemiMinorB": "semi_minor_b", "EllipseAngle": "ellipse_angle_deg",
    "EllipseEqTol%": "ellipse_eq_tolerance_percent", "ROI In R": "roi_inner_radius",
    "ROI Out R": "roi_outer_radius", "Threshold": "binary_threshold",
    "Min Area": "min_contour_area", "Max Area": "max_contour_area",
}
previous_angle_deg_global = None
# prev_frame_time_global and new_frame_time_global are for actual processing FPS calculation
prev_frame_time_global = 0
new_frame_time_global = 0


def calculate_angle(p1_x, p1_y, p2_x, p2_y):
    angle_rad = math.atan2(p2_y - p1_y, p2_x - p1_x)
    angle_deg = math.degrees(angle_rad)
    return angle_deg

def process_frame(frame, frame_count, params_dict):
    global previous_angle_deg_global, prev_frame_time_global, new_frame_time_global
    if frame is None: return None, None
    output_frame = frame.copy()
    center_x = params_dict['center_x']
    center_y = params_dict['center_y']
    semi_major_a = max(1, params_dict['semi_major_a'])
    semi_minor_b = max(1, params_dict['semi_minor_b'])
    ellipse_angle_deg = params_dict['ellipse_angle_deg']
    ellipse_eq_tolerance = params_dict['ellipse_eq_tolerance_percent'] / 100.0
    roi_inner_radius = params_dict['roi_inner_radius']
    roi_outer_radius = params_dict['roi_outer_radius']
    binary_threshold = params_dict['binary_threshold']
    min_contour_area = params_dict['min_contour_area']
    max_contour_area = params_dict['max_contour_area']

    # Actual processing FPS calculation
    new_frame_time_global = time.time()
    processing_fps = 1/(new_frame_time_global-prev_frame_time_global) if (new_frame_time_global-prev_frame_time_global) > 0 else 0
    prev_frame_time_global = new_frame_time_global
    cv2.putText(output_frame, f"Proc. FPS: {int(processing_fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 0), 2, cv2.LINE_AA)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, binary_threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3,3),np.uint8)
    thresh_cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    thresh_cleaned = cv2.morphologyEx(thresh_cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
    cv2.imshow("Thresholded", thresh_cleaned)

    contours, _ = cv2.findContours(thresh_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_marks = []
    phi_rad = math.radians(ellipse_angle_deg)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_contour_area < area < max_contour_area:
            M = cv2.moments(cnt)
            if M["m00"] == 0: continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            dist_from_center = math.sqrt((cx - center_x)**2 + (cy - center_y)**2)
            if not (roi_inner_radius < dist_from_center < roi_outer_radius):
                continue
            x_p = float(cx - center_x)
            y_p = float(cy - center_y)
            x_rot = x_p * math.cos(phi_rad) + y_p * math.sin(phi_rad)
            y_rot = -x_p * math.sin(phi_rad) + y_p * math.cos(phi_rad)
            try:
                ellipse_eq_val = (x_rot / semi_major_a)**2 + (y_rot / semi_minor_b)**2
            except ZeroDivisionError:
                continue
            if abs(ellipse_eq_val - 1.0) < ellipse_eq_tolerance:
                valid_marks.append({'cx': cx, 'cy': cy, 'contour': cnt, 'ellipse_eq_value': ellipse_eq_val})
                cv2.drawContours(output_frame, [cnt], -1, (0, 255, 0), 2)
                cv2.circle(output_frame, (cx, cy), 3, (0, 0, 255), -1)

    current_angle_deg = None
    delta_angle = None
    if valid_marks:
        best_mark = min(valid_marks, key=lambda m: abs(m['ellipse_eq_value'] - 1.0))
        mark_cx, mark_cy = best_mark['cx'], best_mark['cy']
        cv2.circle(output_frame, (mark_cx, mark_cy), 7, (255, 0, 255), 2)
        current_angle_deg = calculate_angle(center_x, center_y, mark_cx, mark_cy)
        cv2.line(output_frame, (center_x, center_y), (mark_cx, mark_cy), (255, 255, 0), 1)
        cv2.putText(output_frame, f"{current_angle_deg:.1f}d", (mark_cx + 5, mark_cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        if previous_angle_deg_global is not None:
            delta_angle = current_angle_deg - previous_angle_deg_global
            if delta_angle > 180: delta_angle -= 360
            elif delta_angle < -180: delta_angle += 360
            cv2.putText(output_frame, f"Delta: {delta_angle:.1f}d", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(output_frame, f"Angle: {current_angle_deg:.1f}d", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
        previous_angle_deg_global = current_angle_deg

    cv2.circle(output_frame, (center_x, center_y), int(roi_inner_radius), (0, 0, 150), 1)
    cv2.circle(output_frame, (center_x, center_y), int(roi_outer_radius), (0, 0, 150), 1)
    cv2.ellipse(output_frame, (center_x, center_y), (int(semi_major_a), int(semi_minor_b)), ellipse_angle_deg, 0, 360, (0, 255, 255), 1)
    cv2.circle(output_frame, (center_x, center_y), 3, (0,255,0), -1)
    return output_frame, delta_angle

def nothing(x): pass

def load_config(config_path, default_params):
    try:
        with open(config_path, 'r') as f:
            loaded_params = json.load(f)
            merged_params = default_params.copy()
            merged_params.update(loaded_params)
            print(f"Config loaded from {config_path}")
            return merged_params
    except FileNotFoundError:
        print(f"Config file {config_path} not found. Using defaults.")
        return default_params
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {config_path}. Using defaults.")
        return default_params
    except Exception as e:
        print(f"Error loading config: {e}. Using defaults.")
        return default_params

def save_config(config_path, params_dict):
    try:
        with open(config_path, 'w') as f: json.dump(params_dict, f, indent=4)
        print(f"Config saved to {config_path}")
    except Exception as e: print(f"Error saving config: {e}")

def update_trackbars_from_params(params_dict, window_name="Trackbars"):
    for trackbar_name, param_key in TRACKBAR_PARAM_MAP.items():
        value = params_dict.get(param_key)
        if value is not None:
            try: cv2.setTrackbarPos(trackbar_name, window_name, int(value))
            except cv2.error as e: print(f"Warn: Trackbar '{trackbar_name}' to {value} failed. {e}")

if __name__ == "__main__":
    current_config_params = load_config(CONFIG_FILE_PATH, current_config_params)
    video_source = "fronti.mp4" # Your video file
    is_video_file = not isinstance(video_source, int)
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source: {video_source}")
        exit()

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps_from_video = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video Res: {width}x{height}. Original Video FPS: {original_fps_from_video:.2f}. Target Playback FPS: {TARGET_FPS:.2f}")

    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 400, 600)
    for trackbar_name, param_key in TRACKBAR_PARAM_MAP.items():
        max_val = 255
        if param_key == "center_x": max_val = width
        elif param_key == "center_y": max_val = height
        elif param_key in ["semi_major_a", "semi_minor_b", "roi_inner_radius", "roi_outer_radius"]: max_val = max(width, height) // 2
        elif param_key == "ellipse_angle_deg": max_val = 180
        elif param_key == "ellipse_eq_tolerance_percent": max_val = 100
        elif param_key == "min_contour_area": max_val = 1000
        elif param_key == "max_contour_area": max_val = 5000
        initial_val = current_config_params.get(param_key, 0)
        initial_val = max(0, min(initial_val, max_val)) if param_key != "ellipse_angle_deg" else min(max(initial_val,0),180)
        cv2.createTrackbar(trackbar_name, "Trackbars", initial_val, max_val, nothing)

    frame_count = 0
    # loop_start_time = time.time() # Not strictly needed per loop for this FPS method

    while True:
        iteration_start_time = time.time() # Time at the start of this iteration

        ret, frame = cap.read()
        if not ret:
            if is_video_file:
                frame_count = 0
                previous_angle_deg_global = None
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret: break
            else: break
        frame_count += 1

        for trackbar_name, param_key in TRACKBAR_PARAM_MAP.items():
            current_config_params[param_key] = cv2.getTrackbarPos(trackbar_name, "Trackbars")

        if current_config_params['roi_inner_radius'] >= current_config_params['roi_outer_radius'] and current_config_params['roi_outer_radius'] > 0:
            new_roi_in_r = current_config_params['roi_outer_radius'] - 1
            current_config_params['roi_inner_radius'] = max(0, new_roi_in_r)
            cv2.setTrackbarPos("ROI In R", "Trackbars", current_config_params['roi_inner_radius'])
        elif current_config_params['roi_inner_radius'] < 0:
            current_config_params['roi_inner_radius'] = 0
            cv2.setTrackbarPos("ROI In R", "Trackbars", 0)

        processed_image, delta = process_frame(frame, frame_count, current_config_params)

        display_text = "'s':Save | 'l':Load | 'q':Quit"
        if processed_image is not None:
            cv2.putText(processed_image, display_text, (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 1, cv2.LINE_AA)
            cv2.imshow("Live Wheel Tracking", processed_image)
        else:
            cv2.putText(frame, display_text, (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 1, cv2.LINE_AA)
            cv2.imshow("Live Wheel Tracking", frame)

        # --- Enforce Target FPS ---
        iteration_processing_time = time.time() - iteration_start_time
        wait_time_seconds = TARGET_FRAME_DURATION - iteration_processing_time
        
        wait_key_delay_ms = 1 # Default minimal delay for GUI responsiveness
        if wait_time_seconds > 0:
            wait_key_delay_ms = int(wait_time_seconds * 1000)
        
        key = cv2.waitKey(wait_key_delay_ms) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            for tb_name, p_key in TRACKBAR_PARAM_MAP.items():
                current_config_params[p_key] = cv2.getTrackbarPos(tb_name, "Trackbars")
            save_config(CONFIG_FILE_PATH, current_config_params)
        elif key == ord('l'):
            current_config_params = load_config(CONFIG_FILE_PATH, current_config_params)
            update_trackbars_from_params(current_config_params)

    cap.release()
    cv2.destroyAllWindows()