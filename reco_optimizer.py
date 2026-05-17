import os
import time

import cv2
import torch
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from facenet_pytorch import MTCNN, InceptionResnetV1
from ultralytics import YOLO

try:
    from twilio.rest import Client
except ImportError:
    Client = None


# ==========================================================
# 1. Configuration
# ==========================================================

load_dotenv()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# FaceNet recognition config — kept the same as your original logic
KNOWN_EMBEDDING_PATH = "embeddings/person1.npy"
KNOWN_PERSON_NAME = "Person1"
THRESHOLD = 0.9

# YOLOv8n human detector config
YOLO_MODEL_PATH = "yolov8n.pt"
PERSON_CLASS_ID = 0
PERSON_CONFIDENCE = 0.45

# Twilio alert config
UNKNOWN_DURATION_SECONDS = 10
ALERT_COOLDOWN_SECONDS = 40

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
ALERT_PHONE_NUMBER = os.getenv("ALERT_PHONE_NUMBER")
TWILIO_VOICE_URL = os.getenv(
    "TWILIO_VOICE_URL",
    "http://demo.twilio.com/docs/voice.xml"
)


# ==========================================================
# 2. Load models
# ==========================================================

print(f"Using device: {DEVICE}")

# Tiny model: YOLOv8n only checks whether a human is present.
human_detector = YOLO(YOLO_MODEL_PATH)

# Face detector and FaceNet model — same functionality as your original code.
mtcnn = MTCNN(image_size=160, margin=20, device=DEVICE)
facenet_model = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)

# Load known embedding
known_embedding = np.load(KNOWN_EMBEDDING_PATH)


# ==========================================================
# 3. Twilio helper functions
# ==========================================================

def twilio_is_configured() -> bool:
    """Return True only if Twilio is installed and all required env variables exist."""
    return all([
        Client is not None,
        TWILIO_ACCOUNT_SID,
        TWILIO_AUTH_TOKEN,
        TWILIO_PHONE_NUMBER,
        ALERT_PHONE_NUMBER,
    ])


def send_twilio_alert() -> None:
    """Send SMS and make a call when an unknown person stays visible too long."""
    if not twilio_is_configured():
        print("[TWILIO] Not configured. Skipping SMS/call.")
        return

    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

    message_body = (
        "Security alert: an unknown person has been detected "
        f"for more than {UNKNOWN_DURATION_SECONDS} seconds."
    )

    try:
        sms = client.messages.create(
            body=message_body,
            from_=TWILIO_PHONE_NUMBER,
            to=ALERT_PHONE_NUMBER,
        )
        print(f"[TWILIO] SMS sent: {sms.sid}")
    except Exception as exc:
        print(f"[TWILIO] SMS failed: {exc}")

    try:
        call = client.calls.create(
            url=TWILIO_VOICE_URL,
            from_=TWILIO_PHONE_NUMBER,
            to=ALERT_PHONE_NUMBER,
        )
        print(f"[TWILIO] Call started: {call.sid}")
    except Exception as exc:
        print(f"[TWILIO] Call failed: {exc}")


# ==========================================================
# 4. Detection and recognition functions
# ==========================================================

def human_detected(frame) -> bool:
    """
    Use YOLOv8n to detect if at least one person exists in the frame.
    If no person is detected, we skip FaceNet to save computation.
    """
    results = human_detector(frame, verbose=False)[0]

    if results.boxes is None:
        return False

    for box in results.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        if class_id == PERSON_CLASS_ID and confidence >= PERSON_CONFIDENCE:
            return True

    return False


def recognize_person_with_facenet(frame):
    """
    Original FaceNet functionality preserved:
    - Convert BGR frame to RGB
    - Detect face with MTCNN
    - Extract embedding using InceptionResnetV1
    - Compare with known embedding using Euclidean distance
    - Return Person1 if distance < THRESHOLD, otherwise Unknown
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)

    face = mtcnn(img)
    label = "Unknown"
    distance = None

    if face is not None:
        face = face.unsqueeze(0).to(DEVICE)

        embedding = facenet_model(face)
        embedding = embedding.detach().cpu().numpy()[0]

        distance = np.linalg.norm(embedding - known_embedding)
        print("Distance:", distance)

        if distance < THRESHOLD:
            label = KNOWN_PERSON_NAME

    return label, distance


# ==========================================================
# 5. Main webcam loop
# ==========================================================

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la caméra.")

    unknown_start_time = None
    last_alert_time = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        label = "No human"
        distance = None

        # Step 1: cheap human detection using YOLOv8n
        if human_detected(frame):
            # Step 2: expensive FaceNet recognition only if a human exists
            label, distance = recognize_person_with_facenet(frame)

            # Step 3: unknown duration tracking
            current_time = time.time()

            if label == "Unknown":
                if unknown_start_time is None:
                    unknown_start_time = current_time

                unknown_duration = current_time - unknown_start_time

                if (
                    unknown_duration >= UNKNOWN_DURATION_SECONDS
                    and current_time - last_alert_time >= ALERT_COOLDOWN_SECONDS
                ):
                    print("[ALERT] Unknown person detected for more than 10 seconds.")
                    send_twilio_alert()
                    last_alert_time = current_time
            else:
                unknown_start_time = None
        else:
            unknown_start_time = None

        # Display label
        if label == KNOWN_PERSON_NAME:
            color = (0, 255, 0)
        elif label == "Unknown":
            color = (0, 0, 255)
        else:
            color = (255, 255, 0)

        cv2.putText(
            frame,
            label,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
        )

        if distance is not None:
            cv2.putText(
                frame,
                f"Distance: {distance:.3f}",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

        cv2.imshow("Recognition with YOLOv8n Gate", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
