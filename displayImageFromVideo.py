import os

import cv2 as cv #Part of the OpenCV library, useful for reading, writing, and processing images and videos
import matplotlib.pyplot as plt #plotting library used here to display images extracted from videos
from IPython.display import HTML # Used to render HTML content in Jupyter notebooks
from base64 import b64encode #Encodes binary data into a base64 string, which is useful for embedding video files in HTML

def display_image(video_path):
    capture_image = cv.VideoCapture(video_path) #capture image
    #ret is a boolean value that returns true if the frame was captured successfully
    #and false otherwise
    ret, frame = capture_image.read() #reading image
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
    else:
        fig = plt.figure(figsize=(10 , 10)) #displaying the image in a 10x10 inch figure
        ax = fig.add_subplot(111)
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB) #converting the frame from BGR to RGB to be able to plot it
        ax.imshow(frame)
        plt.show()

def display_images_from_video_list(video_path_list, data_folder, video_folder):
    plt.figure()
    fig,ax = plt.subplots(2,3,figsize=(16,8)) #prepared a 2x3 grid for plotting images using Matplotlib
    for i, video_file in enumerate(video_path_list[0:6]): #iterates over the first six video files in video_path_list
        video_path = os.path.join(data_folder, video_folder, video_file) #constructs path to video
        capture_image = cv.VideoCapture(video_path)
        ret,frame = capture_image.read()
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        ax[i//3, i%3].imshow(frame)
        ax[i//3, i%3].set_title(f"Video {i+1}")
        ax[i//3, i%3].axis('on')

def play_video(video_file, data_folder, subset):
    # Reading the video file located at the given path in binary mode
    video_url = open(os.path.join(data_folder, subset, video_file),'rb').read()
    # Encoding the video content into a base64 string
    # Embeds this base64-encoded video into an HTML video tag with controls to play it inside a Jupyter notebook
    data_url = b64encode(video_url).decode()
    # Returns the HTML object for display
    return HTML("""<video width=500 controls><source src="%s" type="video/mp4"></video>""" % data_url)