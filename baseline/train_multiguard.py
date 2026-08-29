import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
from torch.utils.data import TensorDataset, ConcatDataset
from torch.utils.data import DataLoader, random_split

from module.util import (
    convert_labels, convert_risk_labels, get_model_name
)

from module.mllm import load_llm
from module.classifier import OmniGuardClassifier, LinearProbing
from module.hidden import mm_prepare_embeddings
from module.predefined_datasets import load_dataset_configs

import logging
# disable Qwen2.5-Omni logging for modifed system prompts
logging.getLogger().setLevel(logging.ERROR)

def split_dataset(dataset_configs, layer_idx, mode='fix', seed=42):
    train_val_datasets = []
    test_datasets = []
    for key, config in dataset_configs.items():
        emb_pt = config.embedding_file
        risk_npy = config.risk_label_file
        if not os.path.exists(emb_pt) or not os.path.exists(risk_npy):
            raise ValueError(f"embedding & Risk label file of {key} not found")
        
        emb_data = torch.load(emb_pt, weights_only=False)
        risk_data = np.load(risk_npy)

        X, Y = emb_data.tensors
        # use the embeddings at layer_idx
        X = X[:, layer_idx, :]
        Y = convert_labels(Y)
        Z = convert_risk_labels(risk_data)

        print(f"{key} total dataset size: {X.shape[0]}")
        
        # filter by risk labels
        if config.allowed_risks() is not None:
            allowed_z_values = convert_risk_labels(config.allowed_risks())
            flags = torch.isin(Z, allowed_z_values)
            Z, X, Y = Z[flags], X[flags], Y[flags]
            print(f"{key} filtered dataset size: {X.shape[0]}")


        train_split = config.train_split
        if train_split <= 1:    # fraction
            n_train_val = int(train_split * X.shape[0])
        else:
            n_train_val = train_split
        n_train_val = min(n_train_val, X.shape[0])

        test_split = config.test_split
        if test_split == -1: # default use the remaining as test
            n_test = max(0, X.shape[0] - n_train_val)
        elif test_split <= 1:
            n_test = int(test_split * X.shape[0])
        else:
            n_test = test_split

        if mode == 'fix': # fix mode splitting
            train_val_set = TensorDataset(X[:n_train_val], Y[:n_train_val], Z[:n_train_val])
            test_set = TensorDataset(X[n_train_val:n_train_val+n_test], 
                                     Y[n_train_val:n_train_val+n_test], 
                                     Z[n_train_val:n_train_val+n_test])
        else:     # random mode splitting: got Subset type
            if n_train_val + n_test < X.shape[0]:
                train_val_set, test_set, _ = random_split(TensorDataset(X, Y, Z), 
                                                           [n_train_val, n_test, X.shape[0] - n_train_val - n_test],
                                                           generator=torch.Generator().manual_seed(seed))
            else:
                train_val_set, test_set = random_split(TensorDataset(X, Y, Z), 
                                                        [n_train_val, n_test],
                                                        generator=torch.Generator().manual_seed(seed))
        
        print(f"{key} train/val dataset size: {len(train_val_set)}, test dataset size: {len(test_set)}")
        if len(train_val_set) > 0:
            train_val_datasets.append(train_val_set)
        if len(test_set) > 0 and config.include_in_test:
            test_datasets.append((key, test_set))

    # combine train_val datasets
    train_val_dataset_combined = ConcatDataset(train_val_datasets)
    train_dataset_final, val_dataset_final = random_split(train_val_dataset_combined, [0.8, 0.2])
    print(f"Train: {len(train_dataset_final)} | Val: {len(val_dataset_final)}")
    
    # for test datasets, we return each key-dataset pairs for fine-grid testing
    test_datasets = {key: test_dataset for key, test_dataset in test_datasets}
    # print each length of test datasets
    for key, test_dataset in test_datasets.items():
        print(f"Test: {key} = {len(test_dataset)}")
    
    return train_dataset_final, val_dataset_final, test_datasets

def test_omniguard(model, test_sets, device, batch_size=32):
    correct_all, total_all = 0, 0
    for key, test_set in test_sets.items():
        statistic_dataset(test_set)
        test_loader = DataLoader(test_set, batch_size=batch_size)
        model.eval()
        correct, total = 0, 0
        for batch_x, batch_y, batch_z in test_loader:
            batch_x, batch_y = batch_x.to(device).float(), batch_y.to(device).long()
            with torch.no_grad():
                logits = model(batch_x)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)
        print(f"[{key}] Accuracy: {correct / total:.4f}")
        total_all += total
        correct_all += correct
    print(f"Overall Accuracy: {correct_all / total_all:.4f}")
    return correct_all / total_all

def statistic_dataset(dataset):
    # print the statistic of the dataset: count the benign and malicious samples
    counter = {}
    for x, y, z in dataset:
        y = y.item()
        if y not in counter:
            counter[y] = 0
        counter[y] += 1
    print(f"Staticstic of the dataset: {counter}")

def train_omniguard(model, train_set, val_set, device='cuda', save_dir='output', batch_size=32, epochs=3, lr=1e-4):
    statistic_dataset(train_set)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses = [], []
    ckpts = []
    for epoch in range(epochs):
        # Training 
        model.train()
        total_loss = 0
        for batch_x, batch_y, batch_z in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
            batch_x, batch_y = batch_x.to(device).float(), batch_y.to(device).long()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_losses.append(total_loss / len(train_loader))

        # Validation
        model.eval()
        val_loss, correct, total = 0, 0, 0
        for batch_x, batch_y, batch_z in val_loader:
            batch_x, batch_y = batch_x.to(device).float(), batch_y.to(device).long()
            with torch.no_grad():
                logits = model(batch_x)
                val_loss += criterion(logits, batch_y).item()
                preds = torch.argmax(logits, dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)
        val_losses.append(val_loss / len(val_loader))
        print(f"[Epoch {epoch + 1}] Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f} | Val Acc: {correct / total:.4f}")
        
        # saving checkpoint
        ckpt_path = f"{save_dir}/omniguard-{epoch}-{val_loss/len(val_loader):.5f}.pt"
        torch.save(model.state_dict(), ckpt_path)
        ckpts.append(ckpt_path)
        # print(f"Saved classifier ckpt to {ckpt_path}")

    # find and return the best checkpoint with minimal val loss
    best_idx = np.argmin(val_losses)
    best_val_loss = val_losses[best_idx]
    best_ckpt = ckpts[best_idx]
    print(f"Best checkpoint: {best_ckpt} with val loss {best_val_loss:.5f}")
    return best_ckpt, best_val_loss
    

def main(args):
    model_path = args.model_path
    best_layer_idx = args.layer
    hidden_size = args.hidden_size
    exp_id = args.exp_id

    model_name = get_model_name(model_path)
    weight_out_dir = f"{args.output_dir}/weights_{model_name}_layer{best_layer_idx}_{exp_id}" # not shared for each classifier experiment
    if not os.path.exists(weight_out_dir):
        os.makedirs(weight_out_dir)
    emb_out_dir = f"{args.output_dir}/embeddings_{model_name}" # global for same model
    if not os.path.exists(emb_out_dir):
        os.makedirs(emb_out_dir)

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    dataset_configs = load_dataset_configs(stage='train' if not args.probing else 'probe', collection=args.collection)
    
    llm = load_llm(model_path, device=device) if not args.skip_load_llm else None
    mm_prepare_embeddings(llm, dataset_configs=dataset_configs, out_dir=emb_out_dir)

    # train multiple versions
    min_loss = float('inf')
    best_ckpt = None
    weight_best_path = None
    test_acc = []
    for version in range(args.repeat):
        weight_version_dir = f"{weight_out_dir}/{version}"
        if not os.path.exists(weight_version_dir):
            os.makedirs(weight_version_dir)
        if args.probing:
            model = LinearProbing(input_dim=hidden_size).to(device)
        else:
            model = OmniGuardClassifier(input_dim=hidden_size).to(device)
        train_set, val_set, test_sets = split_dataset(dataset_configs, best_layer_idx, mode=args.split_mode, seed=args.seed+version)
        v_best_ckpt, v_best_loss = train_omniguard(model, train_set, val_set, device=device, 
                                                   save_dir=weight_version_dir, epochs=args.epochs, lr=args.lr)
        
        # copy the best checkpoint as an ultimate version of trained model (Tip: use shutil)
        shutil.copyfile(v_best_ckpt, f"{weight_version_dir}/omniguard.pt")

        # update the global best checkpoint (by comparing the validate loss)
        if v_best_loss < min_loss:
            min_loss = v_best_loss
            best_ckpt = v_best_ckpt
            weight_best_path = f"{weight_out_dir}/omniguard.pt"
            shutil.copyfile(v_best_ckpt, weight_best_path)
        
        # re-load a best ckpt of all versions, test the performance on the splitted test datasets
        if len(test_sets) > 0:
            model.load_state_dict(state_dict=torch.load(best_ckpt, map_location=device))
            t_acc = test_omniguard(model, test_sets, device=device)
            test_acc.append(t_acc)
    print(f"Global best checkpoint: {best_ckpt} with val loss {min_loss:.5f}, saved to {weight_best_path}")
    print(f"Test accuracy (average): {np.mean(test_acc):.3f}")
    return test_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="models/base/Qwen2.5-Omni-7B")
    parser.add_argument('--output_dir', type=str, default="output/head")
    parser.add_argument('--layer', type=int, default=19)
    parser.add_argument('--hidden_size', type=int, default=3584) # 3584 for Omni2.5, 2048 for Omni3
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--exp_id', type=str, default='default')
    parser.add_argument('--collection', type=str, default='multimodal_v3')
    parser.add_argument('--split_mode', type=str, default='random', choices=['random', 'fix'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--probing', action='store_true', default=False, help='probing mode')
    parser.add_argument('--skip_load_llm', action='store_true', default=False)
    args = parser.parse_args()
    main(args)
    