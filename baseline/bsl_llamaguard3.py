import os
import torch
import argparse
import json

from transformers import AutoModelForImageTextToText, AutoModelForCausalLM, AutoProcessor
from module.util import load_PIL_image, report_metrics, get_model_name
from module.configs import ConfigItem
# from module.predefined_datasets import load_modality_dataset



def inference_llamaguard(model, processor, dataset, include_img=False):
    results = []
    for i, item in enumerate(dataset):
        if include_img:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": item['txt']
                        },
                        {
                            "type": "image",
                        },
                    ],
                }
            ]
            input_prompt = processor.apply_chat_template(
                conversation, return_tensors="pt",
                # TODO: add categories
            )
            image = load_PIL_image(item['img']).convert("RGB") if isinstance(item['img'], str) else item['img']
            input_ids = processor(text=input_prompt, images=image, return_tensors="pt").to(model.device)
            prompt_len = len(input_ids['input_ids'][0])
            output = model.generate(
                **input_ids,
                max_new_tokens=25,
                pad_token_id=0,
            )
        else:
            conversation = [
                {
                    "role": "user",
                    "content": item['txt'],
                }
            ]
            input_ids = processor.apply_chat_template(conversation, return_tensors="pt").to(model.device)
            # print(input_ids)
            prompt_len = len(input_ids['input_ids'][0])
            # prompt_len = input_ids.shape[-1]
            output = model.generate(**input_ids, max_new_tokens=100, pad_token_id=0)
        

        generated_tokens = output[:, prompt_len:]
        output_str = processor.decode(generated_tokens[0], skip_special_tokens=True)
        # print(output_str)
        # parse output_str: typically it is in two lines, e.g., unsafe\nS1
        parse_res = output_str.split()
        print(f"{i+1}/{len(dataset)}:", " ".join(parse_res), item['txt'][:128])
        if len(parse_res) <= 2 and parse_res[0].lower() in ['unsafe', 'safe']:
            results.append({
                "pred_toxicity": parse_res[0].lower(),
                "pred_risk": parse_res[1] if len(parse_res) > 1 else "NaN",
                "gt_toxicity": item['toxicity'],
                "gt_risk": item['risk']
            })
        else:
            print(f"[Error] Invalid output: {output_str}")

    return results

def load_modality_dataset(modality:str, dataset:str):
    from module.predefined_modality import (
        TextModality as TM,
        VisionModality as VM
    )
    if modality == 'text':
        return getattr(TM, dataset)
    elif modality == 'vision':
        return getattr(VM, dataset)
    else:
        raise ValueError("invalid modality")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="models/guard/Llama-Guard-3-11B-Vision")
    parser.add_argument('--output_dir', type=str, default='output/metric.logs')
    parser.add_argument('--cuda', type=str, default='2')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--modality', type=str, default='text')
    parser.add_argument('--limit', type=int, default=-1)
    args = parser.parse_args()

    # get the class property by name
    modality, dataset = args.dataset.split(".") if "." in args.dataset else (args.modality, args.dataset)
    dataset_instance: ConfigItem = load_modality_dataset(modality, dataset)
    
    model_id = args.model_path
    model_name = get_model_name(args.model_path)
    is_vision_guard = 'vision' in model_name.lower()
    if is_vision_guard: # Llama-Guard3-Vision
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=f"cuda:{args.cuda}",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=f"cuda:{args.cuda}",
        )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id)
    
    dataset = dataset_instance.load_unified()
    if args.limit > 0:
        # shuffle dataset with random seed
        import random
        random.seed(42)
        random.shuffle(dataset)
        dataset = dataset[:args.limit]

    results = inference_llamaguard(model, processor, dataset, include_img=is_vision_guard)

    total = len(dataset)
    valid = len(results)
    print(f"Total: {total}, Valid: {valid}, Valid Rate: {valid/total:.2%}")

    os.makedirs(os.path.join(args.output_dir, model_name), exist_ok=True)
    output_file = os.path.join(args.output_dir, model_name, f"{'llamaguard3v' if is_vision_guard else 'llamaguard3'}_{args.dataset}.jsonl")
    with open(output_file, 'w') as f:
        for res in results:
            f.write(json.dumps(res) + '\n')
    
    report_metrics(results, display=True)
    
