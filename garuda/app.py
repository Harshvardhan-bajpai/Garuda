import cv2
import numpy as np
import time
import base64
import uuid
import threading
from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
import eventlet
import socket
from datetime import datetime
from ultralytics import YOLO

# Use eventlet for async operations
eventlet.monkey_patch()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-very-secret-key!'
socketio = SocketIO(app, async_mode='eventlet')

# --- Global State Variables ---
mission_active = False
video_thread = None
drone_sim_thread = None
detection_thread = None
camera_reader_thread = None

# --- Camera Globals ---
global_frame = None
global_detections = []
camera_lock = threading.Lock()
model = None

# --- Camera Configuration ---
RPI_IP = "192.168.137.44"  # Your Pi's IP address
CAMERA_URL_DAY = f"http://{RPI_IP}:8080/?action=stream"
CAMERA_URL_NIGHT = f"http://{RPI_IP}:8081/?action=stream"

# State variables for switching
current_camera_url = CAMERA_URL_DAY
camera_switch_requested = False

# --- Night Vision Globals ---
vignette_mask = None 
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) # Adaptive Contrast enhancer

# --- Drone Simulation State ---
drone_lat = 28.457180
drone_lng = 77.49364


# --- Hardcoded Detection Coordinates ---
SIMULATED_DETECTION_LAT = 28.457180
SIMULATED_DETECTION_LNG = 77.49364

mission_waypoints = []

# --- Utility Functions ---

def is_point_in_polygon(point, polygon):
    lat, lng = point
    n = len(polygon)
    inside = False
    p1_lat, p1_lng = polygon[0]
    for i in range(n + 1):
        p2_lat, p2_lng = polygon[i % n]
        if lat > min(p1_lat, p2_lat):
            if lat <= max(p1_lat, p2_lat):
                if lng <= max(p1_lng, p2_lng):
                    if p1_lat != p2_lat:
                        lng_intersection = (lat - p1_lat) * (p2_lng - p1_lng) / (p2_lat - p1_lat) + p1_lng
                        if p1_lng == p2_lng or lng <= lng_intersection:
                            inside = not inside
        p1_lat, p1_lng = p2_lat, p2_lng
    return inside

def generate_snake_path(polygon_coords, step_meters=25):
    if not polygon_coords: return []
    lats = [c[0] for c in polygon_coords]
    lngs = [c[1] for c in polygon_coords]
    min_lat, max_lat = min(lats), max(lats)
    min_lng, max_lng = min(lngs), max(lngs)
    lat_step = step_meters / 111111.0
    avg_lat_rad = np.deg2rad((min_lat + max_lat) / 2.0)
    lng_step = step_meters / (111111.0 * np.cos(avg_lat_rad))
    SCAN_STEP_LAT = lat_step
    path = []
    current_lat = min_lat
    direction = 1
    while current_lat <= max_lat:
        if direction == 1:
            current_lng = min_lng
            while current_lng <= max_lng:
                path.append([current_lat, current_lng])
                current_lng += lng_step
        else:
            current_lng = max_lng
            while current_lng >= min_lng:
                path.append([current_lat, current_lng])
                current_lng -= lng_step
        current_lat += SCAN_STEP_LAT
        direction *= -1
    return path

# --- Clean Infrared Logic ---

def init_vignette_mask(shape):
    rows, cols = shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, cols/2.0) 
    kernel_y = cv2.getGaussianKernel(rows, rows/2.0)
    kernel = kernel_y * kernel_x.T
    mask = 255 * kernel / np.linalg.norm(kernel)
    mask = mask / mask.max()
    return mask

def apply_cctv_effect(frame):
    global vignette_mask, clahe
    try:
        if vignette_mask is None or vignette_mask.shape[:2] != frame.shape[:2]:
            vignette_mask = init_vignette_mask(frame.shape)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        contrast_enhanced = clahe.apply(gray)
        vignetted = (contrast_enhanced * vignette_mask).astype('uint8')
        final_frame = cv2.cvtColor(vignetted, cv2.COLOR_GRAY2BGR)
        
        return final_frame
    except Exception as e:
        print(f"Error applying CCTV effect: {e}")
        return frame

# --- Background Tasks ---

def camera_reader_task():
    global global_frame, camera_lock
    global current_camera_url, camera_switch_requested

    print(f"Starting camera reader task with: {current_camera_url}")
    cap = cv2.VideoCapture(current_camera_url)
    
    while True:
        if camera_switch_requested:
            print(f"🔄 Switching camera stream to: {current_camera_url}")
            if cap.isOpened(): cap.release()
            socketio.sleep(0.5)
            cap = cv2.VideoCapture(current_camera_url)
            camera_switch_requested = False
            print("🔄 Camera switch complete.")

        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                with camera_lock:
                    global_frame = frame.copy()
            else:
                socketio.sleep(0.5)
        else:
            cap = cv2.VideoCapture(current_camera_url)
            socketio.sleep(1)
        
        socketio.sleep(0.02)

    if cap.isOpened(): cap.release()

def video_streaming_task():
    global global_frame, global_detections, camera_lock
    global current_camera_url, CAMERA_URL_NIGHT
    
    print("Starting video streaming task...")
    
    while True:
        frame = None
        detections_to_draw = []
        is_night_mode = False
        
        with camera_lock:
            if global_frame is not None:
                frame = global_frame.copy()
            if global_detections:
                detections_to_draw = list(global_detections)
            if current_camera_url == CAMERA_URL_NIGHT:
                is_night_mode = True

        if frame is None:
            socketio.sleep(0.05)
            continue
            
        if is_night_mode:
            frame = apply_cctv_effect(frame)
            
        try:
            h, w, _ = frame.shape
            scale_x = w / 400.0
            scale_y = h / 300.0
        except:
            scale_x, scale_y = 1.0, 1.0
            
        for det in detections_to_draw:
            box = det['box']
            conf = det['conf']
            x1 = int(box[0] * scale_x)
            y1 = int(box[1] * scale_y)
            x2 = int(box[2] * scale_x)
            y2 = int(box[3] * scale_y)
            
            # Draw bright green boxes
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Human {conf*100:.0f}%"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = base64.b64encode(buffer).decode('utf-8')
        socketio.emit('video_frame', {'frame': frame_bytes})
        socketio.sleep(0.03) 

def human_detection_task():
    global global_frame, global_detections, camera_lock, model
    global SIMULATED_DETECTION_LAT, SIMULATED_DETECTION_LNG
    print("Starting human detection task...")
        
    while mission_active:
        frame = None
        with camera_lock:
            if global_frame is not None: frame = global_frame.copy()
        
        if frame is None or model is None:
            socketio.sleep(0.1)
            continue
            
        frame_resized = cv2.resize(frame, (400, 300))
        results = model(frame_resized, classes=[0], verbose=False)
        
        new_detections = []
        detection_triggered = False
        max_conf = 0.0 # Store highest confidence found
        
        if results and len(results[0].boxes) > 0:
            detection_triggered = True
            for box in results[0].boxes:
                conf = float(box.conf[0].cpu().numpy())
                if conf > max_conf: max_conf = conf
                
                new_detections.append({
                    'box': box.xyxy[0].cpu().numpy(),
                    'conf': conf
                })
        
        with camera_lock:
            global_detections = new_detections
        
        if detection_triggered:
            socketio.emit('human_detected', {
                'lat': SIMULATED_DETECTION_LAT,
                'lng': SIMULATED_DETECTION_LNG,
                'id': f'det-{uuid.uuid4().hex[:6]}',
                'conf': max_conf # NEW: Send confidence score
            })
            socketio.sleep(3) 
        else:
            socketio.sleep(0.5)

    with camera_lock:
        global_detections = []
    print("Human detection task stopped.")

def drone_position_simulator():
    global drone_lat, drone_lng
    print("Starting drone position simulator...")
    while True:
        socketio.emit('drone_update', {'lat': drone_lat, 'lng': drone_lng})
        socketio.sleep(1)

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

# --- SocketIO Events ---
@socketio.on('connect')
def handle_connect():
    global drone_sim_thread, camera_reader_thread, video_thread
    if drone_sim_thread is None:
        drone_sim_thread = socketio.start_background_task(drone_position_simulator)
    if camera_reader_thread is None:
        camera_reader_thread = socketio.start_background_task(camera_reader_task)
    if video_thread is None:
        video_thread = socketio.start_background_task(video_streaming_task)
    emit('mission_status', {'active': mission_active})

@socketio.on('switch_camera')
def handle_camera_switch():
    global current_camera_url, camera_switch_requested
    print("Received camera switch request.")
    if current_camera_url == CAMERA_URL_DAY:
        current_camera_url = CAMERA_URL_NIGHT
        print(">> Target set to NIGHT camera (8081)")
    else:
        current_camera_url = CAMERA_URL_DAY
        print(">> Target set to DAY camera (8080)")
    camera_switch_requested = True

@socketio.on('start_mission')
def handle_start_mission(data):
    global mission_active, detection_thread, model, mission_waypoints
    if not mission_active:
        try:
            model = YOLO('yolov8s.pt')
        except Exception as e:
            emit('mission_status', {'active': False, 'error': 'YOLO model failed.'})
            return
        mission_active = True
        polygon_coords = data.get('area', [])
        if not polygon_coords:
            emit('mission_status', {'active': False, 'error': 'No mission area.'})
            mission_active = False
            return
        bounding_box_path = generate_snake_path(polygon_coords, step_meters=25)
        inside_path = [p for p in bounding_box_path if is_point_in_polygon(p, polygon_coords)]
        mission_waypoints = [[drone_lat, drone_lng]] + inside_path
        emit('mission_plan', {'path': mission_waypoints, 'corners': polygon_coords, 'return_to_home': [drone_lat, drone_lng]})
        if detection_thread is None or not detection_thread.is_alive():
            detection_thread = socketio.start_background_task(human_detection_task)
        emit('mission_status', {'active': True}, broadcast=True)

@socketio.on('stop_mission')
def handle_stop_mission():
    global mission_active, detection_thread, model, mission_waypoints
    if mission_active:
        mission_active = False
        mission_waypoints = []
        socketio.sleep(0.1)
        detection_thread = None
        model = None
        emit('mission_status', {'active': False}, broadcast=True)

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    def get_local_ip():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(('8.8.8.8', 80))
            ip = s.getsockname()[0]
        except Exception: ip = '127.0.0.1'
        finally: s.close()
        return ip
    local_ip = get_local_ip()
    port = 5000
    print(f"\n--- Server running! Access from other devices: http://{local_ip}:{port}\n")
    socketio.run(app, host='0.0.0.0', port=port, debug=True)