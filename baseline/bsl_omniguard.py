#!/usr/bin/env python3
"""
OmniGuard Main Inference Script

This script provides simple inference capabilities for safety evaluation
on BeaverTails and VLGuard datasets.
"""

import sys
import os
import argparse
import re
import json

from baseline.omniguard.omniguard_evaluator import OmniGuardEvaluator
from baseline.omniguard.utils import LLAMA_GUARD_3_CATEGORY
from module.predefined_modality import (
    TextModality,
    VisionModality,
    AudioModality,
    OmniModality
)
from module.util import report_metrics, get_model_name
# from tqdm import tqdm

def parse_safety_output(text):
    # parse results
    pattern = re.compile(
        r'-\s*Safety Assessment:\s*(?P<safety_assessment>.*?)\s*\n'
        r'-\s*Categories:\s*(?P<categories>.*?)\s*\n'
        r'-\s*Assessment:\s*(?P<assessment>.+)',
        re.DOTALL
    )
    
    match = pattern.search(text.strip())
    if match:
        return {
            'safety': match.group('safety_assessment').strip(),
            'category': match.group('categories').strip(),
            'reasoning': match.group('assessment').strip()
        }
    return None

def inference_omniguard(model_path, dataset, media_type=None, num_samples=None, device=None):
    # Initialize evaluator
    print(f"\n[2/3] Loading model from: {model_path}")
    evaluator = OmniGuardEvaluator(model_id=model_path, device=device)
    print("✓ Model loaded successfully")
    
    total_samples = len(dataset)
    num_eval = num_samples if num_samples else total_samples
    print(f"✓ Dataset loaded: {total_samples} samples")
    print(f"✓ Will process: {num_eval} samples")
    
    # Run inference
    print(f"\n[3/3] Running inference...")
    results = []
    n = len(dataset)
    for i, sample in enumerate(dataset):
        if num_samples and i >= num_samples:
            break
        # print(sample)
        # Perform evaluation
        if media_type == 'text':
            result = evaluator.evaluate(
                prompt_1=sample["txt"],
                prompt_2='',
                categories=LLAMA_GUARD_3_CATEGORY,
                max_token=512,
            )
        else:
            if media_type == 'omni': 
                media = {}
                if sample.get('img') is not None: media['image'] = sample.get('img')
                if sample.get('audio') is not None: media['audio'] = sample.get('audio')
                if sample.get('video') is not None: media['video'] = sample.get('video')
            elif media_type == 'image': media = sample['img']
            elif media_type == 'audio': media = sample['audio']
            else: media = sample['video']
            query = sample.get('txt', None)
            if query is None: query = ''
            result = evaluator.evaluate(
                prompt_1=query,
                prompt_2='',
                categories=LLAMA_GUARD_3_CATEGORY,
                media=media,
                media_type=media_type,
                max_token=512,
            )
        parsed_result = parse_safety_output(result)
        if parsed_result is not None and parsed_result['safety'] is not None:
            print(f"{i+1}/{n}:", parsed_result['safety'], parsed_result['category'], parsed_result['reasoning'][:256])
            results.append({
                "pred_toxicity": parsed_result['safety'],
                "pred_risk": parsed_result['category'],
                "gt_toxicity": sample['toxicity'],
                "gt_risk": sample['risk']
            })
        else:
            print(f"[Error] Invalid output: {result}")
    
    # Print results
    print(f"\n[4/4] Results Summary")
    print("-" * 70)
    print(f"Total Processed: {len(results)}")
    print("-" * 70)
    print("\n✓ Inference completed successfully!\n")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="models/guard/Omniguard-7B")
    parser.add_argument('--output_dir', type=str, default='./output/metric.logs')
    parser.add_argument('--cuda', type=str, default='0')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--preload', action='store_true', default=False)
    args = parser.parse_args()
    modality, dataset_key = args.dataset.split(".")
    if modality == 'text':
        dataset_instance = getattr(TextModality, dataset_key)
    elif modality in ['image', 'video']:
        dataset_instance = getattr(VisionModality, dataset_key)
    elif modality == 'audio':
        dataset_instance = getattr(AudioModality, dataset_key)
    elif modality == 'omni':
        dataset_instance = getattr(OmniModality, dataset_key)
    else:
        raise ValueError("invalid modality")
    
    dataset = dataset_instance.load_unified(preload=args.preload)
    results = inference_omniguard(
        model_path=args.model_path,
        dataset=dataset,
        media_type=modality,
        # num_samples=10,
        device=f'cuda:{args.cuda}'
    )
    # print(results[0]['result'])

    total = len(dataset)
    valid = len(results)
    print(f"Total: {total}, Valid: {valid}, Valid Rate: {valid/total:.2%}")

    model_name = get_model_name(args.model_path)
    os.makedirs(os.path.join(args.output_dir, model_name), exist_ok=True)
    output_file = os.path.join(args.output_dir, model_name, f"omniguard_{args.dataset}.jsonl")
    with open(output_file, 'w') as f:
        for res in results:
            f.write(json.dumps(res) + '\n')
    print("Metrics of", args.dataset)
    report_metrics(results, display=True)

if __name__ == "__main__":
    sys.exit(main())
