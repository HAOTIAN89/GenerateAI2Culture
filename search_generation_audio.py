import torch
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
import os
import json

# save the given image data to a JSON file
def save_image_data(image_data, json_file):
    with open(json_file, "w") as outfile:
        json.dump(image_data, outfile)

# calculate embeddings compute the embeddings for the given text, image, or audio
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

# load the audio in the given directory and compute their embeddings, save them to a JSON file
def load_audios_and_compute_embeddings(audio_dir):
    audio_data = {}
    count = 0
    for audio_file in os.listdir(audio_dir):
        if audio_file.endswith(".wav"):
            audio_path = os.path.join(audio_dir, audio_file)
            embedding = calculate_embeddings("", "", audio_path)
            audio_data[audio_file] = embedding[ModalityType.AUDIO].numpy().tolist()
            count += 1
    print("Loaded {} audios".format(count))
    return audio_data
        
# Check if CUDA is available
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Instantiate model
models = imagebind_model.imagebind_huge(pretrained=True)
models.eval()
models.to(device)

AUDIO_DIR = '../../../audioG/'
JSON_FILE = 'generation_audio.json'

audio_data = load_audios_and_compute_embeddings(AUDIO_DIR)    
save_image_data(audio_data, JSON_FILE)