import os
import json
from flask import Flask, request, jsonify, Response, send_file, make_response
from flask_cors import CORS
import torch
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from utils import audio2img, image2image, text2img, audiotext2image, audioimage2image, textimage2image, audioimagetext2image

# Instantiate Flask app
app = Flask(__name__)

# Enable CORS
CORS(app)

# Paths for storing images and JSON data
IMAGE_DIR = "images"
QUERY_DIR = "query"
JSON_FILE = "image_data.json"
GENERATE_DIR = "generate"   

# Check if CUDA is available
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Instantiate model
models = imagebind_model.imagebind_huge(pretrained=True)
models.eval()
models.to(device)

# Load JSON data if it exists
if os.path.exists(JSON_FILE):
    with open(JSON_FILE, "r") as json_file:
        image_data = json.load(json_file)
else:
    image_data = {}
    
def calculate_embeddings(text, image_path, audio_path):
    if text != "":
        text_list = [text]
        inputs = {ModalityType.TEXT: data.load_and_transform_text(text_list, device)}
    elif image_path != "":
        image_paths = [image_path]
        inputs = {ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device)}
    elif audio_path != "":
        audio_paths = [audio_path]
        inputs = {ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, device)}
    else:
        return None

    with torch.no_grad():
        embeddings = models(inputs)
    
    return embeddings

# print out the URL where the app will be running   
print("Running on http://localhost:5001")

@app.route("/store_image", methods=["POST"])
def store_image():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        filename = os.path.join(IMAGE_DIR, file.filename)
        file.save(filename)
        print("Image stored successfully")
        # Calculate image embedding and store image metadata
        embeddings = calculate_embeddings("", filename, "")
        image_data[filename] = embeddings[ModalityType.VISION].numpy().tolist()
        # Update the JSON file
        with open(JSON_FILE, "w") as json_file:
            json.dump(image_data, json_file)
        return jsonify({"message": "Image stored successfully"}), 200
    
@app.route("/search_image", methods=["POST"])
def search_image():
    text, image_file, audio_file = "", None, None
    # Get the query data from the request
    text = request.form.get("text")
    # image_file = request.files.get("image")
    # audio_file = request.files.get("audio")
    if not text:
        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                # Check the file type (image or audio)
                if file.mimetype == 'image/jpeg':
                    image_file = file
                elif file.mimetype == 'audio/wav':
                    audio_file = file
                else:
                    return jsonify({"error": "Unsupported file type"}), 600
    
    # Calculate embeddings for the query
    if image_file:
        image_filename = os.path.join(QUERY_DIR, image_file.filename)
        image_file.save(image_filename)
        query_embeddings = calculate_embeddings("", image_filename, "")[ModalityType.VISION]
    elif audio_file:
        audio_filename = os.path.join(QUERY_DIR, audio_file.filename)
        audio_file.save(audio_filename)
        query_embeddings = calculate_embeddings("", "", audio_filename)[ModalityType.AUDIO]
    elif text:
        query_embeddings = calculate_embeddings(text, "", "")[ModalityType.TEXT]
    else:
        return jsonify({"error": "No query data provided"}), 400
        
    # Compare query embeddings to stored image embeddings
    best_match = None
    image_embeddings = list(image_data.values())
    image_embeddings_all = torch.cat([torch.tensor(image_embeddings[i]) for i in range(len(image_embeddings))])
    scores = torch.softmax(image_embeddings_all @ query_embeddings.T, dim=0)
    max_index = torch.argmax(scores)
    best_match = list(image_data.keys())[max_index]
            
    if best_match:
        # Load the best match image and send it to the frontend
        with open(best_match, "rb") as image_file:
            image_content = image_file.read()
        return Response(image_content, content_type="image/jpeg"), 200
    else:
        return jsonify({"error": "No match found"}), 404
    
@app.route("/generate_image", methods=["POST"])
def generate_image():
    text, image_file, audio_file = "", None, None
    # Get the query data from the request
    text = request.form.get("text")
    image_file = request.files.get("image")
    audio_file = request.files.get("audio")
    # if all are empty, return error
    if not text and not image_file and not audio_file:
        return jsonify({"error": "No attribute data provided"}), 400
    
    # Calculate embeddings for the attributes
    image_signal = False
    audio_signal = False
    text_signal = False
    if image_file:
        image_filename = os.path.join(GENERATE_DIR, image_file.filename)
        image_file.save(image_filename)
        image_embeddings = calculate_embeddings("", image_filename, "")[ModalityType.VISION]
        image_signal = True
    if audio_file:
        audio_filename = os.path.join(GENERATE_DIR, audio_file.filename)
        audio_file.save(audio_filename)
        audio_embeddings = calculate_embeddings("", "", audio_filename)[ModalityType.AUDIO]
        audio_signal = True
    if text:
        text_signal = True
        
    # Generate image
    if image_signal and audio_signal and text_signal:
        result = audioimagetext2image(image_embeddings, audio_embeddings, text)
    elif image_signal and audio_signal:
        result = audioimage2image(image_embeddings, audio_embeddings)
    elif image_signal and text_signal:
        result = textimage2image(image_embeddings, text)
    elif audio_signal and text_signal:
        result = audiotext2image(audio_embeddings, text)
    elif image_signal:
        result = image2image(image_embeddings)
    elif audio_signal:
        result = audio2img(audio_embeddings)
    elif text_signal:
        result = text2img(text)
    else:
        return jsonify({"error": "No attribute data provided"}), 400
    
    # Send the generated image to the frontend
    result.save("generate/result.jpg")
    with open("generate/result.jpg", "rb") as image_file:
        image_content = image_file.read()
    return Response(image_content, content_type="image/jpeg"), 200
        
if __name__ == "__main__":
    app.run(host="localhost", port=5001)