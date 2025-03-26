# pip install git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3

from transformers import pipeline
import torch

pipe = pipeline(
    "image-text-to-text",
    model="google/gemma-3-27b-it",
    device="cuda",
    torch_dtype=torch.bfloat16
)
messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
            {"type": "text", "text": "What animal is on the candy?"}
        ]
    }
]

output = pipe(text=messages, max_new_tokens=200)
# print(output[0][0]["generated_text"][-1]["content"])
print(output)
# Okay, let's take a look! 
# Based on the image, the animal on the candy is a **turtle**. 
# You can see the shell shape and the head and legs.

