from beam import App, Runtime, Image, Output, Volume

import os
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from PIL import Image as PImage
from io import BytesIO
import base64

cache_path = "./models"
model_id = "runwayml/stable-diffusion-v1-5"

# The environment your code will run on

app = App(
    name="TweetDreams",
    runtime=Runtime(
        cpu=8,
        memory="32Gi",
        gpu="A10G",
        image=Image(
            python_version="python3.10",
            python_packages=[
                "diffusers[torch]>=0.10",
                "transformers",
                "torch",
                "pillow",
                "accelerate",
                "safetensors",
                "xformers",
            ],
        ),
    ),
    volumes=[Volume(name="models", path="./models")],
)

@app.task_queue(
    # File to store image outputs
    outputs=[Output(path="output.png")]
)


@app.rest_api(
    # File to store image outputs
    outputs=[Output(path="output.png"),
             Output(path="orig_image.png"),
             ],
)
def generate_image(origimage, myprompt, stoi, gs):
    prompt = myprompt
    im_binary = origimage.encode('utf-8')
    png_recovered = base64.b64decode(im_binary)
    init_image = PImage.open(BytesIO(png_recovered)).convert('RGB')
    init_image = init_image.resize((768, 512))
    init_image.save("orig_image.png")

    torch.backends.cuda.matmul.allow_tf32 = True

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id,
        revision="fp16",
        torch_dtype=torch.float16,
        cache_dir=cache_path,
    ).to("cuda")

    with torch.inference_mode():
        with torch.autocast("cuda"):
            image = pipe(prompt=prompt, image=init_image, strength=stoi, guidance_scale=gs).images

    print(f"Saved Image: {image[0]}")
    print(myprompt)
    image[0].save("output.png")
    input_image = "output.png"
    with open(input_image, "rb") as image_file:
        encoded_gen_image = base64.b64encode(image_file.read()).decode("utf-8")
    return {"prompt" : myprompt,
            "guidance_scale": gs,
            "strenght_originalimage": stoi,
            "gen_image" : encoded_gen_image}
