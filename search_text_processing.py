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
        
# Check if CUDA is available
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Instantiate model
models = imagebind_model.imagebind_huge(pretrained=True)
models.eval()
models.to(device)

text_list = ["a 17th to 19th Century clothing", "a 20th century clothing", "an ancient clothing", "a medieval clothing"]
text_data = {}

for text in text_list:
    embedding = calculate_embeddings(text, "", "")
    text_data[text] = embedding[ModalityType.TEXT].numpy().tolist()
    
save_image_data(text_data, "search_text_clothing_time.json")
    
