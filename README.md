# Anything to image, learning about cultures

## Our model

Our model is the combination of ImageBind and Stable Diffusion, more specifically, we use the unified latent space of ImageBind and the inference part of Stable-Diffusion-2-1-unclip. We use the ImageBind to transform multimodal (including text, image and audio) into one latent space, and then put the embeddings from the same space into Stable Diffusion to generate images. Because the ImageBind and Stable-Diffusion-2-1-unclip are all trained on the same text-image dataset OpenCLIP, they should have the similar latent space, and that's why we can directly consider the output of ImageBind as the input of Stable Diffusion. Our model should be one of the most powerful multimedia search and generation open-source model which can run on the normal computers nowadays. We also create a website to show the search and generation function of our model.

## Usage

First, you need to set up the ImageBind model.

Install pytorch 1.13+ and other 3rd party dependencies.

```shell
conda create --name imagebind python=3.8 -y
conda activate imagebind

pip install .
```

For windows users, you might need to install `soundfile` for reading/writing audio files. 

```shell
pip install soundfile
```

Then, you should set up the Stable Diffusion model.

```shell
pip install diffusers transformers accelerate scipy safetensors
```

Finally, you can open the `Home.html` to start the frontend and run the following command to start the backend.

```shell
python app.py
```

## Results

If you are interested in our project, please refer to our [Wiki page](https://fdh.epfl.ch/index.php/Extending_Text2Image_Models_to_Accept_Multi-Modal_Conditions_by_Encoding_to_the_CLIP_Latent_Space) for detailed methods and analysis. (Notice: You may need to use EPFL network to open it)

## References
[ImageBind](https://github.com/facebookresearch/ImageBind)
[Stable Diffusion](https://github.com/huggingface/diffusers)
