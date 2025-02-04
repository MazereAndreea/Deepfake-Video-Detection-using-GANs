
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import cv2
import uuid
import torch
from PIL import Image
from video_analysis import detect_faces_mtcnn
from discriminator import Discriminator
from torchvision import transforms

app = Flask(__name__)

# Path for storing uploaded videos and frames
UPLOAD_FOLDER = 'static/videos'
FRAMES_FOLDER = 'static/frames'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['FRAMES_FOLDER'] = FRAMES_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAMES_FOLDER, exist_ok=True)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
discriminator = Discriminator().to(device)
discriminator.load_state_dict(torch.load('discriminator_30epochs_valid.pth', map_location=device))

transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return redirect(request.url)

    video = request.files['video']

    if video.filename == '':
        return redirect(request.url)

    if video and video.filename.endswith('.mp4'):
        # Save the video file
        video_filename = str(uuid.uuid4()) + '.mp4'
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
        video.save(video_path)

        frames_with_predictions, avg_probability = analyze_video(video_path)

        return render_template(
            'index.html',
            video_url=video_filename,
            frames_with_predictions=frames_with_predictions,
	        avg_probability=avg_probability
        )
    return redirect(url_for('index'))

def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_index = 0
    frames_with_predictions = []

    total_probability = 0  # Track sum of probabilities
    frame_count = 0  # Track total frames processed

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faces = detect_faces_mtcnn(frame)
        for face in faces:
            x, y, width, height = face
            face_image = frame[y:y + height, x:x + width]

            face_image_filename = f'{uuid.uuid4()}.jpg'
            face_image_path = os.path.join(app.config['FRAMES_FOLDER'], face_image_filename)
            #face_image_path where is saved, face_image fata detectata
            cv2.imwrite(face_image_path, face_image)

            prediction = predict_deepfake_for_video(face_image, transform, discriminator, device)
            save_frame_result(face_image_filename, prediction)

            frames_with_predictions.append({
                'frame_index': frame_index,
                'filename': face_image_filename,
                'prediction': prediction
            })

            total_probability += prediction
            frame_count += 1

        frame_index += 1

    cap.release()

    avg_probability = (total_probability / frame_count) if frame_count > 0 else 0
    return frames_with_predictions, avg_probability

def predict_deepfake_for_video(face_image, transform, discriminator, device):
    #converts the numpy array to a image
    face_pil = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
    #Face batch of size 1
    face_tensor = transform(face_pil).unsqueeze(0).to(device)  # Add batch dimension
    discriminator.eval()

    with torch.no_grad():
        output = discriminator(face_tensor)
        prediction = torch.sigmoid(output).item()  # Apply sigmoid to get probability
    return prediction


def save_frame_result(filename, prediction):
    result_filename = f'{filename.split(".")[0]}_result.txt'
    with open(os.path.join(app.config['FRAMES_FOLDER'], result_filename), 'w') as f:
        f.write(f'Prediction: {"Fake" if prediction < 0.5 else "Real"}, Probability: {prediction:.4f}')


@app.route('/frames/<filename>')
def get_frame(filename):
    return send_from_directory(app.config['FRAMES_FOLDER'], filename)


@app.route('/videos/<filename>')
def get_video(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

