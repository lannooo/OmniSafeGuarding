import os
import json
import argparse
import pandas as pd
from module.util import report_metrics, find_files
from module.predefined_modality import (
    TextModality, VisionModality, AudioModality, OmniModality, CustomOmniModality
)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 200)
pd.set_option('display.width', 1000)
pd.set_option('expand_frame_repr', False)


report_settings = {
    "truthfulQA": TextModality.truthfulQA,
    "mme": VisionModality.mme,
    "voicebench_alpacaeval": AudioModality.voicebench_alpacaeval,
    "mmbench_video": VisionModality.mmbench_video,

    "advbench": TextModality.advbench,
    "jbv_redteam_2k": TextModality.jbv_redteam_2k,
    "beavertails_30k_test": TextModality.beavertails_30k_test,
    "aegis2_test": TextModality.aegis2_test,
    "openai_moderation": TextModality.openai_moderation,
    "wildguardtest": TextModality.wildguardtest,
    "toxicchat_test": TextModality.toxicchat_test,
    "xstest": TextModality.xstest,
    
    "vlsafe": VisionModality.vlsafe,
    "vlsbench": VisionModality.vlsbench,
    "vlguard": VisionModality.vlguard,
    "siuo": VisionModality.siuo,
    "aiah": AudioModality.aiah,
    "safewatch_real": VisionModality.safewatch_real,
    "video_safetybench_ben": VisionModality.video_safetybench_ben,
    "safebench_t": TextModality.safebench_t,
    "safebench_ti": VisionModality.safebench_ti,
    "safebench_ta": AudioModality.safebench_ta,
    "safebench_tia": OmniModality.safebench_tia,

    "forbidden_question_dan": TextModality.forbidden_question_dan,
    "harmbench_contextual": TextModality.harmbench_contextual,
    "jailbreakbench": TextModality.jailbreakbench,
    "cipherchat": TextModality.cipherchat,
    "rtvlm": VisionModality.rtvlm,
    "mm_safetybench": VisionModality.mm_safetybench,
    "jbv_jailbreak_mini": VisionModality.jbv_jailbreak_mini,
    "figstep": VisionModality.figstep,
    "mml_hades": VisionModality.mml_hades,
    "ajailbench": AudioModality.ajailbench,

    "omni_safetybench_unimodal_t": TextModality.omni_safetybench_unimodal_t,
    "omni_safetybench_unimodal_a": AudioModality.omni_safetybench_unimodal_a,
    "omni_safetybench_unimodal_i": VisionModality.omni_safetybench_unimodal_i,
    "omni_safetybench_unimodal_v": VisionModality.omni_safetybench_unimodal_v,
    "omni_safetybench_dual_ta": AudioModality.omni_safetybench_dual_ta,
    "omni_safetybench_dual_ti": VisionModality.omni_safetybench_dual_ti,
    "omni_safetybench_dual_tv": VisionModality.omni_safetybench_dual_tv,
    "omni_safetybench_omni_tia": OmniModality.omni_safetybench_omni_tia,
    "omni_safetybench_omni_tva": OmniModality.omni_safetybench_omni_tva,

    # "false_reject_mme": VisionModality.false_reject_mme,
    # "false_reject_alpacaeval": AudioModality.false_reject_alpacaeval,
    # "false_reject_mmbench_video": VisionModality.false_reject_mmbench_video,
}

def get_dataset_stats(dataset_fn, stat_file:str):
    if stat_file is not None and os.path.exists(stat_file):
        with open(stat_file, 'r') as f:
            return json.load(f)
    
    # use python counter util
    from collections import Counter
    dataset = dataset_fn.load_unified(preload=False)
    total_count = len(dataset)
    benign_count = len([item for item in dataset if item['toxicity'] == 0])
    harmful_count = len([item for item in dataset if item['toxicity'] == 1])
    risk_counter = Counter([item['risk'] for item in dataset])
    risk_counts = {risk: risk_counter[risk] for risk in risk_counter}
    assert benign_count == risk_counts.get('BNI', 0), "benign_count != risk_counts.get('BNI', 0)"
    stats = {
        "total_count": total_count,
        "benign_count": benign_count,
        "harmful_count": harmful_count,
        "risk_counts": risk_counts,
    }
    if stat_file is not None:
        with open(stat_file, 'w') as f:
            json.dump(stats, f)
    return stats

def preprocess_results(results, strict:bool):
    if len(results) == 0:
        return results
    # compatible for Qwen3Guard-Gen (strict / loose modes)
    if isinstance(results[0]['pred_toxicity'], str):
        if strict:
            results_strict = [
                {
                    "pred_toxicity": "unsafe" if res['pred_toxicity'].lower() in ['unsafe', 'controversial'] else "safe",
                    "pred_risk": res['pred_risk'],
                    "gt_toxicity": res['gt_toxicity'],
                    "gt_risk": res['gt_risk']
                } for res in results
            ]
            return results_strict
        else:
            results_loose = [
                {
                    "pred_toxicity": "unsafe" if res['pred_toxicity'].lower() == 'unsafe' else "safe",
                    "pred_risk": res['pred_risk'],
                    "gt_toxicity": res['gt_toxicity'],
                    "gt_risk": res['gt_risk']
                } for res in results
            ]
            return results_loose
    return results

def compare_results(compare_models, compare_datasets, compare_metrics, strict=False, metric_tag:str='_fast.'):
    all_metrics = {} # model -> {dataset_1 -> {'accuracy': xxx, ...}}
    all_results = {} # model -> [dataset_metrics]
    all_counts = {}  # model -> (valid_cnt, total_cnt)
    for dataset_name, dataset_fn in compare_datasets:
        dataset_stat = get_dataset_stats(dataset_fn, f"./output/metric.logs/dataset_stats/{dataset_name}.json")
        total_count = dataset_stat["total_count"]
        print(f"Dataset: {dataset_name}")
        print(dataset_stat)

        columns = compare_metrics + ["valid_ratio", "+/-"]
        df = pd.DataFrame(columns=columns)
        for dir in compare_models:
            model_name = dir.split("/")[-1]
            # search matching json files
            
            json_files = find_files(dir, f"*.{dataset_name}.jsonl")
            if len(json_files) == 0:
                continue
            json_file = None
            if len(json_files) >= 2:
                print(f"Warning: multiple json files found for {model_name}")
                for j in json_files: print(f"\t{j}")
                for j in json_files:
                    if metric_tag in j:
                        json_file = j
                        break
                
            if json_file is None: json_file = json_files[0]
            print(f"Use json file: {json_file}")
            # read jsonl file into results list
            with open(json_file, 'r') as f:
                results = [json.loads(line) for line in f]
            results = preprocess_results(results, strict)

            if model_name not in all_results:
                all_results[model_name] = []
                all_counts[model_name] = [0, 0] # [valid, total]
                all_metrics[model_name] = {}
            
            valid_count = len(results)
            all_results[model_name].extend(results)
            all_counts[model_name][0] += valid_count
            all_counts[model_name][1] += total_count

            m, gm = report_metrics(results, display=False, dataset_stat=dataset_stat if strict else None)
            all_metrics[model_name][dataset_name] = m
            # write to df
            metrics = [m[k] for k in compare_metrics]
            metrics.append(valid_count / total_count)
            metrics.append(f"{m['count-benign']}/{m['count-harmful']}")
            df.loc[model_name] = metrics
        print(df)
        print("-"*10)
    
    print('='*10)
    for model_name, results in all_results.items(): 
        counts = all_counts[model_name]
        valid, total = counts[0], counts[1]
        print(f"{model_name}: {valid}/{total} = {valid/total:.2%}")
        report_metrics(results, display=True)
    print('='*10)

    for model_name, model_metrics in all_metrics.items():
        print(f"---------{model_name}---------")
        dataset_keys = [item[0] for item in compare_datasets]
        print("\t\t" + " ".join(dataset_keys))
        for m in compare_metrics:
            m_values = [model_metrics[dk][m] for dk in dataset_keys]
            m_values = map(str, [round(v, 4) for v in m_values])
            m_tab = '\t' if len(m)>6 else '\t\t'
            m_str = '\t'.join(m_values)
            print(f"{m}:{m_tab}{m_str}")

    print("==============Done================")
    return all_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dimension', '-d', type=str, default="General", choices=["General", "Safety", "Jailbreak", "FalseReject"])
    # parser.add_argument('--modality', '-m', type=str, default="Text", choices=["Text", "Image", "Audio", "Video"])
    parser.add_argument('--tag', type=str, default="fast")
    parser.add_argument('--models', type=str, nargs='+', default=["Qwen2.5-Omni-7B"])
    parser.add_argument('--datasets', type=str, nargs='+', default=["truthfulQA"])
    parser.add_argument('--loose', action='store_true', default=False)
    args = parser.parse_args()
    print(args)

    compare_model_dirs = [f"./output/metric.logs/{model}" for model in args.models]
    compare_datasets = [(dataset_name, report_settings[dataset_name]) for dataset_name in args.datasets]
    compare_metrics = ["accuracy", "precision", "recall", "f1", "fpr"]
    all_metrics = compare_results(compare_model_dirs, compare_datasets, compare_metrics, strict=not args.loose, metric_tag=f"_{args.tag}.")
    exit(0)