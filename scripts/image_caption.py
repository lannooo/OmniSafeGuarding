import os
import json
import torch

from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor

MODEL_NAME = "Qwen/Qwen3-VL-30B-A3B-Instruct"


# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="cuda",
)

processor = AutoProcessor.from_pretrained(MODEL_NAME)


def generate_from_image(model, processor, img_path, text_prompt, max_new_tokens=256):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": img_path,
                },
                {"type": "text", "text": text_prompt},
            ],
        }
    ]
    # Preparation for inference
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0].strip()


def caption_image(model, processor, img_path):
    return generate_from_image(
        model,
        processor,
        img_path,
        "Describe this image briefly.",
    )

caption_file = "llavaguard_caption.jsonl"

img_dir = "./data/LlavaGuard/img"
for img_file in os.listdir(img_dir):
    img_name = os.path.splitext(img_file)[0]
    img_path = os.path.join(img_dir, img_file)
    caption = caption_image(model, processor, img_path)
    print(img_path, " Caption: ", caption)
    with open(caption_file, 'a') as f:
        f.write(json.dumps({"img_id": img_name, "caption": caption}, ensure_ascii=False) + '\n')
        
    
    
