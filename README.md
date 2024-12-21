## Outputs
![Fairy garden glowing with oversized flowers and fluttering lights](https://github.com/user-attachments/assets/1a61c24a-6fc0-47a4-a433-8f9f9d3c0569)  Fairy garden glowing with oversized flowers and fluttering lights

![Desert oasis at sunset, with crystal-clear water and palm trees](https://github.com/user-attachments/assets/c3535af2-5345-4bb9-bc11-15f4bcd88e2e)

Desert oasis at sunset, with crystal-clear water and palm trees


![Ice castle under northern lights, frost-covered sculptures inside](https://github.com/user-attachments/assets/886f60bd-ab27-480b-8ac5-67635d60173c)

Ice castle under northern lights, frost-covered sculptures inside

## [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/omerbt/MultiDiffusion/blob/master/MultiDiffusion_Panorama.ipynb)

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
