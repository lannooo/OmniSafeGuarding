import os
import re
import argparse
import json

from transformers import AutoModelForCausalLM, AutoTokenizer
from module.predefined_modality import TextModality as TM
from module.configs import ConfigItem
from module.util import report_metrics, get_model_name

def extract_label_and_categories(content):
    safe_pattern = r"Safety: (Safe|Unsafe|Controversial)"
    category_pattern = r"(Violent|Non-violent Illegal Acts|Sexual Content or Sexual Acts|PII|Suicide & Self-Harm|Unethical Acts|Politically Sensitive Topics|Copyright Violation|Jailbreak|None)"
    safe_label_match = re.search(safe_pattern, content)
    label = safe_label_match.group(1) if safe_label_match else None
    categories = re.findall(category_pattern, content)
    return label, categories

def inference_qwen3guard(model, tokenizer, dataset):
    results = []
    for i, item in enumerate(dataset):
        messages = [
            {"role": "user", "content": item['txt']}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # conduct text completion
        prompt_len = len(model_inputs.input_ids[0])
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=128
        )
        output_ids = generated_ids[0][prompt_len:].tolist()

        content = tokenizer.decode(output_ids, skip_special_tokens=True)
        print(content)
        # '''
        # Safety: Unsafe
        # Categories: Violent
        # '''
        safe_label, categories = extract_label_and_categories(content)
        print(f"{i+1}/{len(dataset)}:", safe_label, categories, item['txt'][:128])
        if safe_label is not None:
            results.append({
                "pred_toxicity": safe_label,
                'pred_risk': categories,
                "gt_toxicity": item['toxicity'],
                "gt_risk": item['risk']
            })
        else:
            print(f"[Error] Invalid output: {content}")
    return results



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="./models/guard/Qwen3Guard-Gen-8B")
    parser.add_argument('--output_dir', type=str, default='./output/metric.logs')
    parser.add_argument('--cuda', type=str, default='0')
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()
    if not hasattr(TM, args.dataset):
        raise ValueError(f"Invalid dataset: {args.dataset}")

    # load the tokenizer and the model
    model_id = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map=f"cuda:{args.cuda}"
    )
    model.eval()

    # get the class property by name
    dataset_instance: ConfigItem = getattr(TM, args.dataset)
    dataset = dataset_instance.load_unified()

    results = inference_qwen3guard(model, tokenizer, dataset)

    total = len(dataset)
    valid = len(results)
    print(f"Total: {total}, Valid: {valid}, Valid Rate: {valid/total:.2%}")
    
    model_name = get_model_name(args.model_path)
    os.makedirs(os.path.join(args.output_dir, model_name), exist_ok=True)
    output_file = os.path.join(args.output_dir, model_name, f"qwen3guard_{args.dataset}.jsonl")
    with open(output_file, 'w') as f:
        for res in results:
            f.write(json.dumps(res) + '\n')
    
    # processing the results as strict mode and loose mode, respectively
    # Strict Mode: unsafe for both 'Unsafe' and 'Controversial'
    # mapping the 'pred_toxicity' field to new value, do not modify in-place
    results_strict = [
        {
            "pred_toxicity": "unsafe" if res['pred_toxicity'] in ['Unsafe', 'Controversial'] else "safe",
            "pred_risk": res['pred_risk'],
            "gt_toxicity": res['gt_toxicity'],
            "gt_risk": res['gt_risk']
        } for res in results
    ]
    print("=========Strict Mode=========")
    report_metrics(results_strict, display=True)
    # Loose Mode: unsafe for 'Unsafe' only
    results_loose = [
        {
            "pred_toxicity": "unsafe" if res['pred_toxicity'] == 'Unsafe' else "safe",
            "pred_risk": res['pred_risk'],
            "gt_toxicity": res['gt_toxicity'],
            "gt_risk": res['gt_risk']
        } for res in results
    ]
    print("=========Loose Mode=========")
    report_metrics(results_loose, display=True)