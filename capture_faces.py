import cv2
import os

save_path = "faces/person1"

cap = cv2.VideoCapture(0)

count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow("Capture Faces", frame)

    key = cv2.waitKey(1)

    # Press S to save image
    if key == ord('s'):
        img_path = os.path.join(save_path, f"{count}.jpg")
        cv2.imwrite(img_path, frame)
        print(f"Saved {img_path}")
        count += 1

    # Press Q to quit
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()