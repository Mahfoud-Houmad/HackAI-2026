from ultralytics import YOLO
import cv2

# Load YOLOv8 nano model
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

# Animal classes we want
TARGET_CLASSES = ["cow", "sheep"]

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Run YOLO
    results = model(frame)

    cow_count = 0
    sheep_count = 0

    # Process detections
    for result in results:
        boxes = result.boxes

        for box in boxes:
            cls_id = int(box.cls[0])

            class_name = model.names[cls_id]

            if class_name in TARGET_CLASSES:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                confidence = float(box.conf[0])

                # Count animals
                if class_name == "cow":
                    cow_count += 1

                elif class_name == "sheep":
                    sheep_count += 1

                # Draw rectangle
                color = (0, 255, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                label = f"{class_name} {confidence:.2f}"

                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

    # Show counts
    cv2.putText(
        frame,
        f"Cows: {cow_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    cv2.putText(
        frame,
        f"Sheep: {sheep_count}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    cv2.imshow("Animal Detection", frame)

    # Exit with Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()