import multiprocessing
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import torch
from PIL import Image
import requests

model_id = "google/gemma-3-27b-it"

def run_model_instance(output_file, image_url, prompt_text):
    # Load model and processor
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id, device_map="auto"
    ).eval()
    
    processor = AutoProcessor.from_pretrained(model_id)

    # Construct the input messages
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_url},
                {"type": "text", "text": prompt_text}
            ]
        }
    ]

    # Preprocess the inputs
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]

    # Generate output
    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        generation = generation[0][input_len:]

    decoded = processor.decode(generation, skip_special_tokens=True)

    # Save output to file
    with open(output_file, "w") as file:
        file.write(decoded)

def main():
    # Define image URLs and prompts for two different instances
    image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
    prompt_text = "Describe this image in detail."

    # Create two separate processes for model instances
    process1 = multiprocessing.Process(target=run_model_instance, args=("model1-output.txt", image_url, prompt_text))
    process2 = multiprocessing.Process(target=run_model_instance, args=("model2-output.txt", image_url, prompt_text))

    # Start both processes
    process1.start()
    process2.start()

    # Wait for both processes to complete
    process1.join()
    process2.join()

if __name__ == "__main__":
    main()
