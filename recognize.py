import cv2
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

device = 'cuda' if torch.cuda.is_available() else 'cpu'

mtcnn = MTCNN(image_size=160, margin=20, device=device)

model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load known embedding
known_embedding = np.load("embeddings/person1.npy")

cap = cv2.VideoCapture(0)

THRESHOLD = 0.9

while True:

    ret, frame = cap.read()

    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    img = Image.fromarray(rgb)

    face = mtcnn(img)

    label = "Unknown"

    if face is not None:

        face = face.unsqueeze(0).to(device)

        embedding = model(face)

        embedding = embedding.detach().cpu().numpy()[0]

        distance = np.linalg.norm(
            embedding - known_embedding
        )

        print("Distance:", distance)

        if distance < THRESHOLD:
            label = "Person1"

    cv2.putText(
        frame,
        label,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.imshow("Recognition", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()