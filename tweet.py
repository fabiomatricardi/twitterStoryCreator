from beam import App, Runtime, Image, Output, Volume, VolumeType

import os
import torch
from io import BytesIO
import base64
from transformers import LlamaForCausalLM, LlamaTokenizer
import sentencepiece

# The environment your code will run on

app = App(
    name="TweetGenerator",
    runtime=Runtime(
        cpu=8,
        memory="32Gi",
        gpu="A10G",
        image=Image(
            python_version="python3.10",
            python_packages=[
                "accelerate>=0.16.0,<1",
                "transformers[torch]>=4.28.1,<5",
                "torch>=1.13.1,<2",
                "langchain",
                "sentencepiece",
                "xformers",
                "protobuf"
            ],
        ),
    ),
    volumes=[
        Volume(
            name="model_weights",
            path="./model_weights",
            volume_type=VolumeType.Persistent,
        )
    ],
)

# Cached model
cache_path = "./model_weights"

# Huggingface model
model_id = "psmathur/orca_mini_3b"

def load_models():
    tokenizer = LlamaTokenizer.from_pretrained(
        model_id, cache_dir=cache_path, legacy=False, use_fast=False
    )
    model = LlamaForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=cache_path,
    )

    return tokenizer, model

@app.rest_api(loader=load_models)
def generate(**inputs):
    # Retrieve cached model from the loader
    tokenizer, model = inputs["context"]
    myprompt = inputs["myprompt"]

    # Generate output
    you = myprompt

    instruction = f"Write a creative twitter post about '{you}'."
    system = "You are an AI assistant that follows instruction extremely well. Help as much as you can."

    if input:
        prompt = f"### System:\n{system}\n\n### User:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    else:
        prompt = f"### System:\n{system}\n\n### User:\n{instruction}\n\n### Response:\n"

    tokens = tokenizer.encode(prompt)
    tokens = torch.LongTensor(tokens).unsqueeze(0)
    tokens = tokens.to("cuda")

    instance = {
        "input_ids": tokens,
        "top_p": 1.0,
        "temperature": 0.7,
        "generate_len": 1024,
        "top_k": 50,
    }

    length = len(tokens[0])
    with torch.no_grad():
        rest = model.generate(
            input_ids=tokens,
            max_length=length + instance["generate_len"],
            use_cache=True,
            do_sample=True,
            top_p=instance["top_p"],
            temperature=instance["temperature"],
            top_k=instance["top_k"],
        )
    output = rest[0][length:]
    blog = tokenizer.decode(output, skip_special_tokens=True)

    # print the results
    print("Suggested tweet content:\n")
    print("Generated with psmathur/orca_mini_3b\n")
    print("-------")
    print(blog)
    print("-------")
    # return the json with the results
    return {"blogpost": blog}

