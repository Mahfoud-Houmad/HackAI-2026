import os
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Face detector
mtcnn = MTCNN(image_size=160, margin=20, device=device)

# FaceNet model
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

embeddings = []

folder = "faces/person1"

for file in os.listdir(folder):

    path = os.path.join(folder, file)

    img = Image.open(path)

    face = mtcnn(img)

    if face is not None:

        face = face.unsqueeze(0).to(device)

        embedding = model(face)

        embeddings.append(
            embedding.detach().cpu().numpy()[0]
        )

# Average embedding
mean_embedding = np.mean(embeddings, axis=0)

# Save embedding
np.save("embeddings/person1.npy", mean_embedding)

print("Embedding saved!")