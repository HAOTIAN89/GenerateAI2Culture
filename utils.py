from diffusers import DiffusionPipeline
import torch

def audio2img(audioembeddings: torch.Tensor):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pipe = DiffusionPipeline.from_pretrain("stabilityai/stable-diffusion-2-1-unclip-small", torch_dtype=torch.float16).to(device)
    image = pipe(image_embeds=audioembeddings.half()).images[0]
    return image

def image2image(imageembeddings: torch.Tensor):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pipe = DiffusionPipeline.from_pretrain("stabilityai/stable-diffusion-2-1-unclip-small", torch_dtype=torch.float16).to(device)
    image = pipe(image_embeds=imageembeddings.half()).images[0]
    return image

def text2img(textembeddings: torch.Tensor):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pipe = DiffusionPipeline.from_pretrain("stabilityai/stable-diffusion-2-1-unclip-small", torch_dtype=torch.float16).to(device)
    image = pipe(image_embeds=textembeddings.half()).images[0]
    return image

def audiotext2image(audioembeddings: torch.Tensor, text: str):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pipe = DiffusionPipeline.from_pretrain("stabilityai/stable-diffusion-2-1-unclip-small", torch_dtype=torch.float16).to(device)
    image = pipe(image_embeds=audioembeddings.half(), prompt=text).images[0]
    return image

def audioimage2image(audioembeddings: torch.Tensor, imageembeddings: torch.Tensor):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pipe = DiffusionPipeline.from_pretrain("stabilityai/stable-diffusion-2-1-unclip-small", torch_dtype=torch.float16).to(device)
    embeddings =(audioembeddings + imageembeddings)/2
    image = pipe(image_embeds=embeddings).images[0]
    return image

def textimage2image(imageembeddings: torch.Tensor, text: str):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pipe = DiffusionPipeline.from_pretrain("stabilityai/stable-diffusion-2-1-unclip-small", torch_dtype=torch.float16).to(device)
    image = pipe(image_embeds=imageembeddings, prompt=text).images[0]
    return image

def audioimagetext2image(imageembeddings: torch.Tensor, audioembeddings: torch.Tensor, text: str):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pipe = DiffusionPipeline.from_pretrain("stabilityai/stable-diffusion-2-1-unclip-small", torch_dtype=torch.float16).to(device)
    embeddings =(audioembeddings + imageembeddings)/2
    image = pipe(image_embeds=embeddings, prompt=text).images[0]
    return image