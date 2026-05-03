import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["NCCL_DEBUG"] = "warn"

import argparse
import json

from module.mllm import load_llm
from module.configs import ConfigItem
from module.predefined_modality import load_modality_dataset
from module.util import report_metrics, get_model_name
from module.infer import inference_omniguard_sft

import logging
# disable Qwen2.5-Omni logging for modifed system prompts
logging.getLogger().setLevel(logging.ERROR)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="./models/guard/OmGuard-Qwen2.5-7B-Enhance")
    parser.add_argument('--output_dir', type=str, default='./output/metric.logs')
    parser.add_argument('--cuda', type=str, default='0')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--modality', type=str, default='text')
    parser.add_argument('--limit', type=int, default=-1)
    parser.add_argument('--report_error', action='store_true')
    parser.add_argument('--version', type=str, default='v3-1')
    parser.add_argument('--enable_vllm', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()

    modality, dataset = args.dataset.split(".") if "." in args.dataset else (args.modality, args.dataset)
    dataset_instance: ConfigItem = load_modality_dataset(modality, dataset)
     
    llm = load_llm(args.model_path, 
                   device=f"cuda:{args.cuda}", 
                   use_vllm=args.enable_vllm)
    dataset = dataset_instance.load_unified(preload=False)
    if args.limit > 0:
        # shuffle dataset with random seed
        import random
        random.seed(42)
        random.shuffle(dataset)
        dataset = dataset[:args.limit]

    results = inference_omniguard_sft(llm, dataset, version=args.version, report_detail=args.report_error, batch_size=args.batch_size)

    total = len(dataset)
    valid = len(results)
    print(f"Results for model: {args.model_path}, dataset: {args.dataset}")
    print(f"Total: {total}, Valid: {valid}, Valid Rate: {valid/total:.2%}")

    model_name = get_model_name(args.model_path)
    os.makedirs(os.path.join(args.output_dir, model_name), exist_ok=True)
    
    # save results
    output_file = os.path.join(args.output_dir, model_name, f"omniguard_sft_{args.dataset}.jsonl")
    print(f"Saving results to {output_file}")
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    # save error results
    if args.report_error:
        error_file = output_file.replace('.jsonl', '_error.jsonl')
        with open(error_file, 'w') as f:
            for result in results:
                if (result['gt_toxicity'] == 1 and result['pred_toxicity'] == 'safe') \
                    or (result['gt_toxicity'] == 0 and result['pred_toxicity'] == 'unsafe'):
                    f.write(json.dumps(result) + '\n')
    
    report_metrics(results, display=True)
    llm.close()
