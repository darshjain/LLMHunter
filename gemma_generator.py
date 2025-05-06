import os
import csv
import json
import torch
from collections import deque
from tqdm import tqdm
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

# Configuration
MODEL_ID    = "google/gemma-3-12b-it"
OUTPUT_CSV  = "generated_programs.csv"
NUM_SAMPLES = 5000        # total programs to generate
CONTEXT_SIZE= 10          # how many past outputs to remember

# Load model and processor once
model     = Gemma3ForConditionalGeneration.from_pretrained(MODEL_ID, device_map="auto").eval()
processor = AutoProcessor.from_pretrained(MODEL_ID)

# Determine if we're resuming or starting fresh
if os.path.exists(OUTPUT_CSV):
    # Load existing rows
    existing = []
    with open(OUTPUT_CSV, newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            existing.append({"Program": row["Program"], "Fault": row["Fault"]})
    start_idx = len(existing)
    mode = "a"  # append
    recent_programs = deque(existing[-CONTEXT_SIZE:], maxlen=CONTEXT_SIZE)
    print(f"Resuming from sample #{start_idx} out of {NUM_SAMPLES}")
else:
    # No file yet → start at zero
    start_idx = 0
    mode = "w"  # write (will create and write header)
    recent_programs = deque(maxlen=CONTEXT_SIZE)
    print("Starting fresh generation from sample #0")

# Open CSV (write header if fresh, else append)
with open(OUTPUT_CSV, mode, newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["Program", "Fault"])
    if mode == "w":
        writer.writeheader()

    for idx in tqdm(range(start_idx, NUM_SAMPLES), desc="Generating programs"):
        # Build “avoid repeats” context
        if recent_programs:
            avoid_list = "\n".join(
                f"- {i+1}. {entry['Program'].splitlines()[0]}…"
                for i, entry in enumerate(recent_programs)
            )
            repeat_warning = (
                "⚠️ Ensure this program is highly diverse—"
                f"different from last {len(recent_programs)}:\n{avoid_list}\n\n"
            )
        else:
            repeat_warning = ""

        # Build prompt messages (same as before)…
        messages = [
            {"role":"system","content":[{"type":"text","text":"You are a helpful assistant."}]},
            {"role":"user","content":[{"type":"text","text":
                repeat_warning +
                "Example Input:\n"
                "{\n"
                '  "Program": "#include <iostream>\\n'
                'int divide(int a, int b) {\\n'
                '    return a / b;\\n'
                '}\\n'
                'int main() {\\n'
                '    int x = 10;\\n'
                '    int y = 0;  // division by zero\\n'
                '    std::cout << divide(x, y) << std::endl;\\n'
                '    return 0;\\n'
                '}",\n'
                '  "Fault": "int y = 0;  // division by zero"\n'
                "}\n\n"
                "Now, Task: Generate a brand-new C++ program containing a subtle logical or "
                "syntactical fault. Return **only** a single parsable JSON object with two fields:\n"
                "{\n"
                '  "Program": "<full C++ program here>",\n'
                '  "Fault": "<exact faulty code segment here>"\n'
                "}\n"
                "Constraints:\n"
                "- Program and fault must be novel, highly diverse, and non-trivial.\n"
                "- Fault field must include **only** the faulty segment, no extra text.\n"
                "- Do **NOT** include any text outside the JSON.\n"
            }]}
        ]

        # Tokenize & generate (same as before)…
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True,
            tokenize=True, return_dict=True,
            return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs, max_new_tokens=800,
                do_sample=True, top_p=0.95,
                temperature=1.0
            )[0]

        generated_text = processor.decode(
            output_ids[input_len:], skip_special_tokens=True
        ).strip().lstrip("```json").rstrip("```").strip()
        print(f"[DEBUG] Sample #{idx} generated text: {generated_text}")
        try:
            result = json.loads(generated_text)
            writer.writerow({
                "Program": result["Program"],
                "Fault": result["Fault"]
            })
            recent_programs.append(result)
        except json.JSONDecodeError:
            tqdm.write(f"[Warning] Sample #{idx} invalid JSON, skipping.")