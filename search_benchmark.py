import torch
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
import os
import json

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

# load the images in the given directory and compute their embeddings, save them to a JSON file
def load_images_and_compute_embeddings(image_dir):
    image_data = {}
    count = 0
    for image_file in os.listdir(image_dir):
        if image_file.endswith(".jpg"):
            image_path = os.path.join(image_dir, image_file)
            embedding = calculate_embeddings("", image_path, "")
            image_data[image_file] = embedding[ModalityType.VISION].numpy().tolist()
            count += 1
    print("Loaded {} images".format(count))
    return image_data

# save the given image data to a JSON file
def save_image_data(image_data, json_file):
    with open(json_file, "w") as outfile:
        json.dump(image_data, outfile)
        

if __name__ == "__main__":
    # Check if CUDA is available
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Instantiate model
    models = imagebind_model.imagebind_huge(pretrained=True)
    models.eval()
    models.to(device)

    # load the images in the given directory and compute their embeddings, save them to a JSON file
    IMAGE_DIR = '../../../datasets/painting_dataset'
    JSON_FILE = 'search_paintings.json'
    image_data = load_images_and_compute_embeddings(IMAGE_DIR)
    save_image_data(image_data, JSON_FILE)