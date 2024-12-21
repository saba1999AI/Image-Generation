## Outputs


## Diffusers Integration [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/omerbt/MultiDiffusion/blob/master/MultiDiffusion_Panorama.ipynb)

```
import torch
from diffusers import StableDiffusionPanoramaPipeline, DDIMScheduler

model_ckpt = "stabilityai/stable-diffusion-2-base"
scheduler = DDIMScheduler.from_pretrained(model_ckpt, subfolder="scheduler")
pipe = StableDiffusionPanoramaPipeline.from_pretrained(
     model_ckpt, scheduler=scheduler, torch_dtype=torch.float16
)

pipe = pipe.to("cuda")

prompt = "a photo of the dolomites"
image = pipe(prompt).images[0]
```

## Gradio Demo 
We provide a gradio UI for our method. Running the following command in a terminal will launch the demo:
```
python app_gradio.py
```
This demo is also hosted on HuggingFace [here](https://huggingface.co/spaces/weizmannscience/MultiDiffusion)


