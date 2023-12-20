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

text_list = ["chinese newyear hotpot", "Indian family gathering eating traditional Indian cuisine. Introduce elements like a plate of Basmati rice, a bowl of rich, creamy Dal Makhani, vibrant vegetable Biryani, and freshly baked Naan bread. Decorate with garnishes like coriander leaves and slices of lemon", 
            "Jpanese family gathering eating traditional Japanese cuisine. Introduce elements like a plate of sushi and a bowl of ramen", 
            "Imagine a cozy Japanese family gathering in a traditional Japanese dining setting. The family members are seated around a low wooden dining table, comfortably settled on cushioned seating. The table is adorned with an array of exquisite Japanese dishes: a colorful plate of sushi, featuring delectable tuna, salmon, and cucumber rolls; alongside, a steaming bowl of ramen emits an inviting aroma, filled with tender slices of chashu pork, boiled eggs, green vegetables, and soft noodles. Traditional Japanese paintings hang on the walls, contributing to a warm and harmonious atmosphere. The family members are smiling, enjoying the delicious food and the joyous moments of togetherness",
            "Imagine a cozy Japanese family gathering in a traditional Japanese dining setting",
            "Envision a lively Mexican family gathering, embracing the rich culinary traditions of Mexico. The family is congregated around a large, rustic wooden table in a vibrantly decorated room, echoing the spirit of Mexican culture. The table is a feast for the eyes, laden with traditional Mexican fare: a large, colorful bowl of homemade guacamole, surrounded by crispy tortilla chips; steaming plates of tacos filled with seasoned carne asada, fresh lettuce, diced tomatoes, and topped with melting cheese; a pot of aromatic, spicy black bean soup garnished with cilantro; and a pitcher of refreshing agua fresca with slices of lime and watermelon. Brightly colored papel picado banners flutter overhead, adding to the festive atmosphere. Laughter and lively conversation fill the air as the family enjoys the delicious flavors and the warmth of each other's company",
            "Envision a lively Mexican family gathering, embracing the rich culinary traditions of Mexico",
            "Korean families are sitting at tables, enjoying crispy Korean fried chicken and various side dishes like kimchi and pickled radishes",
            "a oil painting", "a Chinese traditional freehand brushwork landscape painting", "a rococo style oil painting",
            "a chinese emporer of Qing Dynasty wearing Qing Clothing", 
            "clothing of a chinese emporer of Qing Dynasty",
            "a cloth during China's Qing Dynasty",
            "korean traditonal clothing",
            "Japanese traditonal clothing",
            "Mexican traditonal clothing"]

text_data = {}

for text in text_list:
    print(len(text))
    embedding = calculate_embeddings(text, "", "")
    text_data[text] = embedding[ModalityType.TEXT].numpy().tolist()
    
save_image_data(text_data, "generation_text.json")