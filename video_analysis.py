# video_analysis.py

import os
import cv2
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms
from discriminator import Discriminator
from facenet_pytorch import MTCNN

transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

def detect_faces_mtcnn(frame, conf_threshold=0.8):
    """
    Detect faces in a frame using MTCNN face detection model.

    Parameters:
        frame (numpy.ndarray): Input frame in BGR format.
        conf_threshold (float): Confidence threshold for filtering weak detections.

    Returns:
        List of bounding boxes [x, y, width, height].
    """
    # Initialize the MTCNN detector
    mtcnn = MTCNN(keep_all=True, min_face_size=50)

    # Convert the frame to RGB (MTCNN expects RGB input)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    boxes, probs = mtcnn.detect(frame_rgb)

    faces = []
    if boxes is not None:
        for i, box in enumerate(boxes):
            if probs[i] > conf_threshold:  # Only consider detections with a high confidence score
                x1, y1, x2, y2 = box
                width = int(x2 - x1)
                height = int(y2 - y1)
                faces.append((int(x1), int(y1), width, height))  # [x, y, width, height]

    return faces


def display_frames(frame):

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Normalize the frame to [0, 1] for displaying
    frame_rgb = frame_rgb / 255.0

    # Display the image
    plt.figure(figsize=(5, 5))
    plt.imshow(frame_rgb)
    plt.axis('off')
    plt.title("Frame")
    plt.show()


def predict_deepfake_for_video(face, transform, discriminator, device):

    face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
    face_tensor = transform(face_pil).unsqueeze(0).to(device)  # Add batch dimension
    discriminator.eval()  # Set discriminator to evaluation mode
    with torch.no_grad():
        output = discriminator(face_tensor)
        prediction = torch.sigmoid(output).item()  # Apply sigmoid to get probability
        print(f"Face classified as {'Real' if prediction > 0.5 else 'Fake'}, Probability: {prediction:.4f}")

    display_frames(face)

def process_video(video_path, transform, discriminator, device):

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = total_frames // 5

    for i in range(5):

        cap.set(cv2.CAP_PROP_POS_FRAMES, i * frame_interval)
        ret, frame = cap.read()
        if not ret:
            break

        faces = detect_faces_mtcnn(frame)

        for face in faces:
            # Extract the face region from the frame using the bounding box
            x, y, width, height = face

            x2 = min(x + width, frame.shape[1])
            y2 = min(y + height, frame.shape[0])

            face_image = frame[y:y + height, x:x + width]

            if face_image.size == 0:
                print(f"Skipping empty face crop at frame {i}")
                continue

            # Pass the cropped face to the deepfake prediction function
            predict_deepfake_for_video(face_image, transform, discriminator, device)

    cap.release()


def analyze_all_videos_in_directory(videos_directory, discriminator, transform, device):

    video_files = [f for f in os.listdir(videos_directory) if f.endswith('.mp4')]

    for video_file in video_files:
        video_path = os.path.join(videos_directory, video_file)
        print(f"Analyzing video: {video_path}")
        process_video(video_path, transform, discriminator, device)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    discriminator = Discriminator().to(device)
    discriminator.load_state_dict(torch.load('./discriminator.pth', map_location=device))
    analyze_all_videos_in_directory("./test_videos/Celeb-synthesis", discriminator, transform, device)

if __name__ == "__main__":
    main()