import os
import torch
import numpy as np
import time

from tqdm import tqdm
from torch.utils.data import TensorDataset
from module.configs import DatasetConfigs
from module.mllm import BaseLLM


def extract_risk_labels(dataset):
    risk_labels = []
    for sample in tqdm(dataset, desc="Extracting Risk Labels"):
        risk = sample['risk']
        risk_labels.append(risk)
    # return as np.array
    return np.array(risk_labels)


def extract_embeddings(llm:BaseLLM, dataset, include_response=False):
    embeddings, labels = [], []

    for sample in tqdm(dataset, desc="Extracting embeddings"):
        txt_prompt = sample.get('txt', None)
        img_prompt = sample.get('img', None)
        audio_prompt = sample.get('audio', None)
        video_prompt = sample.get('video', None)
        start = time.time()
        outputs = llm.inference(txt_prompt=txt_prompt, img_prompt=img_prompt, 
                                audio_prompt=audio_prompt, video_prompt=video_prompt, 
                                response=sample.get('answer', None) if include_response else None)
        end1 = time.time()
        next_hidden_state_list = []
        
        for layer_i in range(1, len(outputs.hidden_states)): # discard the first one
            hidden_states_i = outputs.hidden_states[layer_i] # (1, S, H)
            next_hidden_state = hidden_states_i[:, -1, :] # (1, H)
            next_hidden_state_list.append(next_hidden_state)
        # concatenate the hidden states at the second dimension (1, H) * L -> (L, H)
        repr_last_all_layers = torch.cat(next_hidden_state_list, dim=0).detach().cpu() # hidden from layer 1 to last (-1)
        end2 = time.time()
        if end1 - start > 5:
            print(f"Extracting embeddings: {end1 - start:.2f}s, {end2 - end1:.2f}s")
        # # use the hidden state of the specified layer
        # best_hidden_state = outputs.hidden_states[emb_layer + 1] # add 1 to remove the first redundant one
        # # use the next token value: (B, S, H) -> (1, 1, H)
        # repr_last = best_hidden_state[:, -1, :].cpu()

        embeddings.append(repr_last_all_layers)
        labels.append(sample["toxicity"])
    # X = torch.cat(embeddings, dim=0)

    # stack the embeddings: B * (L, H) -> (B, L, H)
    X = torch.stack(embeddings, dim=0)
    y = torch.tensor(labels).unsqueeze(1).float()
    return TensorDataset(X, y)

def mm_prepare_embeddings(llm:BaseLLM, dataset_configs:DatasetConfigs, 
                          out_dir, include_response=False):
    # cache embeddings of multiple datasets
    total_datasets = len(dataset_configs.items())
    print(f"Start caching embeddings for {total_datasets} datasets...")
    for idx, (key, config) in enumerate(dataset_configs.items()):
        print(f"Processing dataset {idx+1}/{total_datasets}: {key}")
        embedding_file = f"{out_dir}/{key}.pt"
        risk_label_file = f"{out_dir}/{key}_risk.npy"
        has_embedding = os.path.exists(embedding_file)
        has_risk_label = os.path.exists(risk_label_file)

        if has_embedding and has_risk_label:
            config.embedding_file = embedding_file
            config.risk_label_file = risk_label_file
            print(f"Embedding & Risk labels for {key} already exists. Skipping...")
            continue
        
        try:
            if not has_embedding or not has_risk_label:
                dataset = config.load_unified()
                if not has_embedding:
                    # item in dataset: {'txt':..., 'img':..., 'audio':..., 'toxicity':..., 'risk': ...}
                    embeddings = extract_embeddings(llm, dataset=dataset, include_response=include_response)
                    torch.save(embeddings, embedding_file)
                if not has_risk_label:
                    risk_labels = extract_risk_labels(dataset=dataset)
                    np.save(risk_label_file, risk_labels)
            config.embedding_file = embedding_file
            config.risk_label_file = risk_label_file
        except Exception as e:
            print(f"Error processing {key}: {str(e)}")
            raise e
    return dataset_configs