import os
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import torch
from tqdm import tqdm
import json
import csv
from collections import deque

model_id = "google/gemma-3-27b-it"
num_iterations = 3000 
history_limit = 6  

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id, device_map={"": "cuda:0"}
).eval()

processor = AutoProcessor.from_pretrained(model_id)

with open("prompts/prompt-generate.txt", "r") as file:
    base_prompt_text = file.read().strip()

history = deque(maxlen=history_limit)

csv_filename = "output.csv"
file_exists = os.path.isfile(csv_filename)

with open(csv_filename, "a", newline="") as csvfile:
    fieldnames = ["program", "fault"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()

    for _ in tqdm(range(num_iterations), desc="Running Inference"):
        messages = list(history)
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": base_prompt_text}
            ]
        })

        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = model.generate(
                **inputs, max_new_tokens=4000, do_sample=True
            )
            generation = generation[0][input_len:]

        decoded = processor.decode(generation, skip_special_tokens=True)
        decoded = decoded.lstrip("```json").rstrip("```")
        print("THE DECODED", decoded)

        try:
            output_json = json.loads(decoded)
            program = output_json.get("Program", "").strip()
            fault = output_json.get("Fault", "").strip()
        except json.JSONDecodeError as e:
            print("JSON Decode Error:", e)
            program = ""
            fault = ""

        
        writer.writerow({
            "program": program,
            "fault": fault
        })
        csvfile.flush()

        history.append({
            "role": "user",
            "content": [
                {"type": "text", "text": base_prompt_text}
            ]
        })
        history.append({
            "role": "assistant",
            "content": [
                {"type": "text", "text": decoded}
            ]
        })
        
