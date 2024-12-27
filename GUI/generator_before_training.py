import numpy as np
from flask import Flask, send_file
from PIL import Image
import io

from GUI.app import app
from main import sendData

@app.route('/generated-image')
def get_generated_image():
    # Step 1: Generate image
    img_array = sendData()  # This should return a generated array
    img = Image.fromarray(img_array.astype('uint8'))

    # Step 2: Convert to in-memory file
    img_io = io.BytesIO()
    img.save(img_io, 'PNG')
    img_io.seek(0)

    # Step 3: Serve image as response
    return send_file(img_io, mimetype='image/png')