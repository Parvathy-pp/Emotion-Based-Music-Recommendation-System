"""
app.py
Flask backend:
  GET /          → serves the main UI
  GET /video     → MJPEG webcam stream with annotations
  GET /emotion   → JSON {emotion, confidence, playlist_url}
"""

import cv2
import time
import threading
from flask import Flask, Response, jsonify, render_template
from emotion_engine import EmotionEngine

app = Flask(__name__)

# ─────────────────────────────────────────────
# Spotify embed playlists (no OAuth required)
# ─────────────────────────────────────────────
PLAYLISTS = {
    "Happy":   "https://open.spotify.com/embed/playlist/37i9dQZF1DX3rxVfibe1L0?utm_source=generator",
    "Sad":     "https://open.spotify.com/embed/playlist/37i9dQZF1DX7qK8ma5wgG1?utm_source=generator",
    "Angry":   "https://open.spotify.com/embed/playlist/37i9dQZF1DX4sWSpwq3LiO?utm_source=generator",
    "Neutral": "https://open.spotify.com/embed/playlist/37i9dQZF1DWWQRwui0ExPn?utm_source=generator",
}

# ─────────────────────────────────────────────
# Shared state (thread-safe with a lock)
# ─────────────────────────────────────────────
state_lock = threading.Lock()
shared_state = {
    "stable_emotion": "Neutral",
    "confidence":     0.0,
}

engine = EmotionEngine()
camera = None
camera_lock = threading.Lock()


# ─────────────────────────────────────────────
# Camera helpers
# ─────────────────────────────────────────────
def get_camera():
    global camera
    with camera_lock:
        if camera is None or not camera.isOpened():
            camera = cv2.VideoCapture(0)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_FPS, 30)
    return camera


def generate_frames():
    """Generator that yields MJPEG frames with emotion annotations."""
    cam = get_camera()
    frame_count = 0
    DETECT_EVERY = 5       # run DeepFace every N frames to keep FPS smooth

    stable_emotion = "Neutral"
    confidence     = 0.0

    while True:
        success, frame = cam.read()
        if not success:
            time.sleep(0.05)
            continue

        frame_count += 1

        # Run heavy detection only on every Nth frame
        if frame_count % DETECT_EVERY == 0:
            stable_emotion, confidence = engine.process_frame(frame)
            with state_lock:
                shared_state["stable_emotion"] = stable_emotion
                shared_state["confidence"]     = confidence

        # Always annotate every frame for smooth display
        annotated = engine.annotate_frame(frame.copy(), stable_emotion, confidence)

        ret, buffer = cv2.imencode(
            ".jpg", annotated,
            [cv2.IMWRITE_JPEG_QUALITY, 80],
        )
        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", playlists=PLAYLISTS)


@app.route("/video")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/emotion")
def emotion_api():
    with state_lock:
        emotion    = shared_state["stable_emotion"]
        confidence = shared_state["confidence"]
    return jsonify({
        "emotion":      emotion,
        "confidence":   confidence,
        "playlist_url": PLAYLISTS.get(emotion, PLAYLISTS["Neutral"]),
    })


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Emotion Music Recommender  —  http://127.0.0.1:5000")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
