from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

#cap = cv2.VideoCapture(0)

cap = cv2.VideoCapture("Herding Tame Cows Every Afternoon - Dancing Cows, Cows Sounds, Cows, Funny Cows, Cows Eating Grass.2.mp4")

print("Camera opened:", cap.isOpened())

TARGET_CLASSES = ["cow", "sheep"]

LINE_Y = 300
counted_ids = set()

cow_count = 0
sheep_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1024, 768))

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml"
    )

    cv2.line(frame, (0, LINE_Y), (1024, LINE_Y), (0, 0, 255), 2)

    for result in results:

        if result.boxes.id is None:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()
        track_ids = result.boxes.id.cpu().numpy()

        for box, cls_id, track_id in zip(boxes, class_ids, track_ids):

            class_name = model.names[int(cls_id)]

            if class_name not in TARGET_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box)

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(frame,
                        f"{class_name} ID:{int(track_id)}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2)

            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

            # COUNTING LOGIC
            if cy > LINE_Y and track_id not in counted_ids:

                counted_ids.add(track_id)

                if class_name == "cow":
                    cow_count += 1
                elif class_name == "sheep":
                    sheep_count += 1

    cv2.putText(frame, f"Cows: {cow_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.putText(frame, f"Sheep: {sheep_count}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("Animal Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

