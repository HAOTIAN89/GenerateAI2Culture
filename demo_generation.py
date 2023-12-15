import torch
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from utils import textimage2image

# Check if CUDA is available
device1 = "cuda:0" if torch.cuda.is_available() else "cpu"
device2 = "cpu"

#input
text = "a painting"
image = "images/car_image.jpg"
audio = None

# Instantiate model
models = imagebind_model.imagebind_huge(pretrained=True)
models.eval()
models.to(device1)

def calculate_image_embeddings(image_path):
    if image_path != "":
        image_paths = [image_path]
        inputs = {ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device1)}

        with torch.no_grad():
            embeddings_dict = models(inputs)
        
        return embeddings_dict['vision']
    else:
        return None

imageembeddings = calculate_image_embeddings(image).to(device2)
promptembeddings = calculate_image_embeddings(text).to(device2)
# result = textimage2image(imageembeddings,text)
result = textimage2image(promptembeddings,imageembeddings)
result.save("generate/8.png")