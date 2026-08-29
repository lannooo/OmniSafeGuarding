import os
import time
import torch
import argparse
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

from module.mllm import load_llm
from module.hidden import mm_prepare_embeddings
from module.policy import UNIFIED_POLICY_TOXINOMY
from module.classifier import OmniGuardClassifier, LinearProbing
from module.predefined_datasets import load_dataset_configs
from module.util import (
    convert_labels, 
    convert_risk_labels, 
    convert_risk_id_to_category,
    get_model_name,
    calculate_metrics
)

import logging
# disable Qwen2.5-Omni logging for modifed system prompts
logging.getLogger().setLevel(logging.ERROR)


def evaluate_classifier(model, dataset, device, batch_size=32):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds, labels, risks = [], [], []
    with torch.no_grad():
        for batch_x, batch_y, batch_z in dataloader:
            batch_x = batch_x.to(device).float()
            batch_y = batch_y.to(device).long()
            batch_z = batch_z.to(device).long()
            
            logits = model(batch_x)
            pred = torch.argmax(logits, dim=1)
            # collect the predictions, ground_truth labels and risk labels, as a list
            preds.extend(pred.tolist())
            labels.extend(batch_y.tolist())
            risks.extend(batch_z.tolist())
    preds, labels, risks = np.array(preds), np.array(labels), np.array(risks)
    pos_label = 0 if (labels == 0).all() else 1
    return calculate_metrics(preds, labels, risks, pos_label=pos_label)
    


def do_evaluate(configs, model, device, layer_idx, metric_name='accuracy'):
    details = pd.DataFrame(columns=["Dataset", "Overall",] + list(UNIFIED_POLICY_TOXINOMY.keys()))
    for name, config in configs.items():
        emb_pt_path = config.embedding_file
        risk_npy_path = config.risk_label_file
        if not os.path.exists(emb_pt_path) or not os.path.exists(risk_npy_path):
            print(f"Warning! No embedding file / Risk file for {name}, Skipping")
            continue

        print(f"Evaluating {name}:")
        emb_data = torch.load(emb_pt_path, weights_only=False)
        risk_data = np.load(risk_npy_path)
        X, Y = emb_data.tensors
        # use the embeddings at layer_idx
        X = X[:, layer_idx, :]
        Z = convert_risk_labels(risk_data) # convert risk labels (str -> id)
        Y = convert_labels(Y)

        assert X.shape[0] == Y.shape[0], "X and Y should have the same size"

        dataset = TensorDataset(X, Y, Z)
        
        metrics, g_metrics = evaluate_classifier(model, dataset, device)
        overall_metric = metrics[metric_name]
        metric_per_risk = {convert_risk_id_to_category(k): v.get(metric_name, 0.0) for k, v in g_metrics.items()}
        count_per_risk = {convert_risk_id_to_category(k): v['count'] for k, v in g_metrics.items()}

        print(f"{name}: (Overall) {overall_metric:.4f}, ", end="")
        # print acc per risk in one line
        for k, acc in metric_per_risk.items():
            print(f"[{k}] {acc * 100:.2f}% (#{count_per_risk[k]}), ", end="")
        print("\n-------------------------")

        details.loc[len(details)] = {"Dataset": name, "Overall": overall_metric, **metric_per_risk}
        # print(f"Inference time on {name}: {total_t:.2f}s total | {avg_t * 1000:.2f} ms/sample")
    return details


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="models/base/Qwen2.5-Omni-7B")
    parser.add_argument('--output_dir', type=str, default="output/head")
    parser.add_argument('--layer', type=int, default=19)
    parser.add_argument('--hidden_size', type=int, default=3584)
    parser.add_argument('--exp_id', type=str, default='default')
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--collection', type=str, required=True)
    parser.add_argument('--metric', type=str, default='f1', choices=['accuracy', 'f1', 'precision', 'recall', 'fpr'])
    parser.add_argument('--probing', action='store_true', default=False, help='probing mode')
    parser.add_argument('--skip_load_llm', action='store_true', default=False)
    parser.add_argument('--tag', type=str, default='')
    args = parser.parse_args()

    model_path = args.model_path
    best_layer_idx = args.layer
    hidden_size = args.hidden_size
    exp_id = args.exp_id

    model_name = get_model_name(model_path)
    emb_out_dir = f"{args.output_dir}/embeddings_{model_name}{args.tag}"
    if not os.path.exists(emb_out_dir):
        os.makedirs(emb_out_dir)
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    # load llm and prepare embeddings
    llm = load_llm(model_path, device) if not args.skip_load_llm else None
    dataset_configs = load_dataset_configs(stage='eval', collection=args.collection)
    mm_prepare_embeddings(llm, dataset_configs=dataset_configs, out_dir=emb_out_dir)


    if args.repeat <= 1: # only evaluate on the global best version
        checkpoints = [f"{args.output_dir}/weights_{model_name}_layer{best_layer_idx}_{exp_id}/omniguard.pt"]
    else:
        checkpoints = [f"{args.output_dir}/weights_{model_name}_layer{best_layer_idx}_{exp_id}/{i}/omniguard.pt" for i in range(args.repeat)]
        checkpoints = [ckpt for ckpt in checkpoints if os.path.exists(ckpt)] # check existing ckpts

    # load omniguard ckpt
    results_list = []
    for ckpt_path in checkpoints:
        print(f"Loading omniguard ckpt from {ckpt_path}")
        if args.probing:
            model = LinearProbing(input_dim=hidden_size)
        else:
            model = OmniGuardClassifier(input_dim=hidden_size)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device)
        model.eval()

        print("Evaluation start...")
        print(f"----------Use Metric: {args.metric}--------------")
        result = do_evaluate(dataset_configs, model, device, best_layer_idx, metric_name=args.metric)
        results_list.append(result)
    
    # print details: DataFrame
    dataset_names = result['Dataset'].to_list()
    overall_acc_pd = pd.DataFrame(columns=dataset_names)
    for i, result in enumerate(results_list):
        print(f"Result for ckpt {i}:")
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 
                                'display.width', 1000, 'display.expand_frame_repr', True, 
                                'display.float_format', '{:.2%}'.format):
            print(result)
        # insert a new row to overall_acc_pd
        overall_acc_pd.loc[len(overall_acc_pd)] = result['Overall'].to_list()
    
    # insert a new row to overall_acc_pd to calculate the average overall performance per dataset/Column
    overall_acc_pd.loc[len(overall_acc_pd)] = overall_acc_pd.mean(axis=0)
    print("\nOverall:")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 
                            'display.width', 1000, 'display.expand_frame_repr', True, 
                            'display.float_format', '{:.2%}'.format):
        print(overall_acc_pd.T)
    print('\nBrief:', ' '.join(map(lambda x: f'{x:.2f}', overall_acc_pd.iloc[-1].to_list())))




        
    