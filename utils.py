from diffusers import DiffusionPipeline
import torch

def audio2img(audioembeddings: torch.Tensor):
    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float32)
    image = pipe(image_embeds=audioembeddings).images[0]
    return image

def image2image(imageembeddings: torch.Tensor):
    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float32)
    image = pipe(image_embeds=imageembeddings).images[0]
    return image

def text2img(textembeddings: torch.Tensor):
    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float32)
    image = pipe(image_embeds=textembeddings).images[0]
    return image

def audiotext2image(audioembeddings: torch.Tensor, text: str):
    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float32)
    image = pipe(image_embeds=audioembeddings, prompt=text).images[0]
    return image

def audioimage2image(audioembeddings: torch.Tensor, imageembeddings: torch.Tensor):
    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float32)
    embeddings =(audioembeddings + imageembeddings)/2
    image = pipe(image_embeds=embeddings).images[0]
    return image

def textimage2image(imageembeddings: torch.Tensor, text: str):
    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float32)
    image = pipe(image_embeds=imageembeddings, prompt=text).images[0]
    return image

def audioimagetext2image(imageembeddings: torch.Tensor, audioembeddings: torch.Tensor, text: str):
    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float32)
    embeddings =(audioembeddings + imageembeddings)/2
    image = pipe(image_embeds=embeddings, prompt=text).images[0]
    return image