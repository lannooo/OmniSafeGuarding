import os
import random
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["NCCL_DEBUG"] = "warn"

import argparse
import json

from module.mllm import load_llm
from module.configs import ConfigItem
from module.predefined_modality import (
    TextModality, VisionModality, AudioModality, OmniModality, CustomOmniModality
)
from module.predefined_datasets import *
from module.util import report_metrics, get_model_name
from module.infer import inference_omniguard_sft

import logging
# disable Qwen2.5-Omni logging for modifed system prompts
logging.getLogger().setLevel(logging.ERROR)

def get_config_item_attributes(cls):
    attrs = {}
    for name, value in cls.__dict__.items():
        if isinstance(value, ConfigItem):
            attrs[value] = name
    return attrs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="./models/base/Qwen2.5-Omni-7B")
    parser.add_argument('--output_dir', type=str, default='./output/metric.logs')
    parser.add_argument('--cuda', type=str, default='0')
    parser.add_argument('--report_error', action='store_true')
    parser.add_argument('--version', type=str, default='v3-1')
    parser.add_argument('--enable_vllm', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--reload_llm_each_iter', action='store_true', default=False)
    parser.add_argument('--disable_sys', action='store_true', default=False)

    args = parser.parse_args()
    
    model_name = get_model_name(args.model_path)
    os.makedirs(os.path.join(args.output_dir, f"{model_name}.{args.version}"), exist_ok=True)
    eval_sets = {
        # For fast evaluate the whole inference framework
        "fast": eval_fast_validate,
        
        # Uncomment the following lines for complete evaluation, which may take a long time to run (especially for vision and audio datasets)
        # For complete evaluation for safety-utility assessment
        # "utility": eval_general,
        # "safety": eval_basic_safety,
        # "jailbk": eval_ext_jailbreaks,

        # For modality-bias evaluation
        # Note that this dataset contains only a subset, we provide a anyonmous download link in the README for full download 
        # "mb": eval_custom_modality, 

        # For False-Rejection to text short-cuts
        # "fr": eval_false_reject,
    }
    dataset_names_map = get_config_item_attributes(TextModality) \
        | get_config_item_attributes(VisionModality) \
            | get_config_item_attributes(AudioModality) \
                | get_config_item_attributes(OmniModality) \
                    | get_config_item_attributes(CustomOmniModality)

    if not args.reload_llm_each_iter:
        llm = load_llm(args.model_path, device=f"cuda:{args.cuda}", use_vllm=args.enable_vllm)
    
    eval_results, complete_eval_results = [], []
    log_file_name = f"./logs/{model_name.replace('-', '_')}.{args.version}.metric.log"
    os.makedirs(os.path.dirname(log_file_name), exist_ok=True)

    for modality, config_list in eval_sets.items():
        for config_name, dataset_instance in config_list.items():
            dataset_name = dataset_names_map[dataset_instance.data_config]
            print(f"{modality} - {dataset_name}")

            dataset = dataset_instance.load_unified(preload=False)
            dataset = random.sample(dataset, 10) # just for debug

            if args.reload_llm_each_iter:
                llm = load_llm(args.model_path, device=f"cuda:{args.cuda}", use_vllm=args.enable_vllm)
            
            results = inference_omniguard_sft(llm, dataset, version=args.version, 
                                              report_detail=args.report_error, 
                                              batch_size=args.batch_size,
                                              use_sys_prompt=True if not args.disable_sys else False)
            if args.reload_llm_each_iter:
                llm.close()
            
            total = len(dataset)
            valid = len(results)
            valid_rate = valid / total
            print(f"Total: {total}, Valid: {valid}, Valid Rate: {valid_rate:.2%}")

            output_file = os.path.join(args.output_dir, f"{model_name}.{args.version}", f"omniguard_sft_{modality}.{dataset_name}.jsonl")
            with open(output_file, 'w') as f:
                for result in results:
                    f.write(json.dumps(result) + '\n')
            
            metrics, _ = report_metrics(results, display=True)
            metric_val = metrics['f1']
            eval_results.append((f"{modality}.{dataset_name}", metric_val, valid_rate))
            complete_eval_results.append((f"{modality}.{dataset_name}", metrics, valid_rate))
            # write to logging file
            with open(log_file_name, 'a') as f:
                f.write(f"{modality}.{dataset_name}: {metric_val:.4f} ({valid_rate:.2%})\n")

    if not args.reload_llm_each_iter:
        llm.close()
    
    # print overall eval results
    print("Overall F1 Results:")
    for name, metric_val, valid_rate in eval_results:
        print(f"{name}: {metric_val:.4f} ({valid_rate:.2%})")
    print("----------------")

    # print complete eval results
    for name, metrics, valid_rate in complete_eval_results:
        print(f"{name}: valid=({valid_rate:.2%})")
        for metric_name, metric_val in metrics.items():
            print(f"  {metric_name}: {metric_val:.4f}")
        print("----------------")
