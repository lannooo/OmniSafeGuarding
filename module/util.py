import os
import re
import shutil
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import requests
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix
)
import tomlkit
from io import BytesIO
from PIL import Image
from pathlib import Path

from module.policy import PROMPT_SAFE
from module.policy import PolicyHub

def get_model_name(model_path) -> str:
    # extract the model name from model path, e.g., path/to/Qwen2-VL-7B-Instruct -> Qwen2-VL-7B-Instruct
    # if path has path seprator at last, remove it
    model_name = os.path.basename(os.path.abspath(model_path))
    return model_name

def not_empty_str(v:str):
    return v is not None and v.strip() != ""

def not_empty(item:dict, key:str):
    if key not in item:
        return False
    if item[key] is None:
        return False
    if isinstance(item[key], str) and item[key] == "":
        return False
    return True


def mask_textual_query(query:str, tokenizer, mask_token:str)->str:
    tokenizer = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
    query_length = len(tokenizer.encode(query, add_special_tokens=False))
    return ' '.join([mask_token] * query_length)

def mask_image(ori_img:Image.Image, background_color=(255, 255, 255)):
    return Image.new("RGB", ori_img.size, background_color)

def mask_video(ori_video:list, background_color=(255, 255, 255)):
    return [Image.new("RGB", frame.size, background_color) for frame in ori_video]

def mask_audio(ori_audio:np.ndarray):
    return np.zeros(ori_audio.shape)


def build_omniguard_input_alpaca(item:dict, roleplay="User's question: "):
    input = roleplay
    images, audios, videos = [], [], []
    # <audio><image><video> Bob's question: ...
    if item['video']:
        input = "<video>" + input
        videos.append(item['video'])
    if item['image']:
        input = "<image>" + input
        images.append(item['image'])
    if item['audio']:
        input = "<audio>" + input
        audios.append(item['audio'])
    if item['txt']:
        input = input + item['txt'] # as suffix of input ('<image>Bob: balabala')

    return input, images, audios, videos

def build_omniguard_output(toxicity, risk_label, reasoning:str=None, sources:str=None,
                           perception:dict=None, ind_risks:dict=None, formatting='simple'):
    if formatting == 'simple':
        if toxicity == PROMPT_SAFE:
            output = "safe"
        else:
            output = f"unsafe\n{risk_label}\n{reasoning}"
    elif formatting == 'toml':
        output = ""
        if perception is not None:
            output += "[summary]\n"
            if 'text' in perception:
                output += f"text = \"{perception['text']}\"\n"
            else:
                output += "text = \"N/A\"\n"
            if "vision" in perception:
                output += f"vision = \"{perception['vision']}\"\n"
            else:
                output += "vision = \"N/A\"\n"
            if "audio" in perception:
                output += f"auditory = \"{perception['audio']}\"\n"
            else:
                output += "auditory = \"N/A\"\n"
        if ind_risks is not None:
            output += "\n[analysis]\n"
            if 'text' in ind_risks:
                output += f"text_safety = \"{ind_risks['text']}\"\n"
            if "vision" in ind_risks:
                output += f"vision_safety = \"{ind_risks['vision']}\"\n"
            if "audio" in ind_risks:
                output += f"auditory_safety = \"{ind_risks['audio']}\"\n"
        output += "\n[verdict]\n"
        if reasoning is not None: # first or last? first is absolute better
            output += f"reasoning = \"{reasoning}\"\n"
        output += f"safety = \"{'safe' if toxicity == PROMPT_SAFE else 'unsafe'}\"\n"
        output += f"category = \"{'N/A' if toxicity == PROMPT_SAFE else risk_label}\"\n"
        if sources is not None:
            # unsafe_modality = "N/A" or "text, image, audio, video"
            output += f"unsafe_modality = \"{'N/A' if toxicity == PROMPT_SAFE else sources}\"\n"
    else:
        raise ValueError(f"Invalid formatting: {formatting}")
    return output

def parse_output(output:str, formatting='simple') -> dict:
    if formatting == 'simple':
        segs = output.split("\n")
        n_seg = len(segs)
        if n_seg <= 3 and segs[0].lower() in ['safe', 'unsafe']:
            info = {
                "pred_toxicity": segs[0].lower(),
                "pred_risk": segs[1] if len(segs) > 1 else "NaN",
                "pred_extra": " | ".join(segs[2:]) if len(segs) > 2 else "NaN"
            }
            return info
        else:
            raise ValueError(f"Invalid output in simple parsing: {output}")
    elif formatting == 'toml':
        try:
            output_clean = preprocess_toml_str(output)
            tomldata = tomlkit.parse(output_clean.strip()) # TODO maybe need to handle exception
            verdict_info = tomldata['verdict']       # basic information, must be present
        except KeyError:
            verdict_info = preprocess_toml_str_2(output)
        
        pred_toxicity = verdict_info['safety'].lower()
        if pred_toxicity not in ['safe', 'unsafe']: raise ValueError(f"Invalid pred_toxicity: {pred_toxicity}")
        pred_risk = verdict_info['category']
        # more information can be extracted
        return {
            "pred_toxicity": pred_toxicity,
            "pred_risk": pred_risk,
            "pred_extra": {
                "reasoning": verdict_info['reasoning'] if 'reasoning' in verdict_info else 'NaN',
                "perception": tomldata['summary'] if 'summary' in tomldata else 'NaN',
                "independent_risks": tomldata['analysis'] if 'analysis' in tomldata else 'NaN'
            }
        }
    else:
        raise ValueError(f"Invalid formatting: {formatting}")


def preprocess_toml_str_2(toml_str:str):
    pattern = r'\[verdict\]\s*safety\s*=\s*"([^"]*)"\s*category\s*=\s*"([^"]*)"'
    match = re.search(pattern, toml_str)
    if match:
        return {
            "safety": match.group(1),
            "category": match.group(2)
        }
    else:
        raise ValueError(f"failed to parse output {toml_str}")

def preprocess_toml_str(toml_str:str):
    # print('======================================')
    # print(toml_str)
    # print('--------------------------------------')
    toml_str = toml_str.strip()
    if toml_str.startswith('```'):
        toml_str = toml_str.removeprefix('```toml').removesuffix('```')
    lines = toml_str.split('\n')
    cleaned_lines = []
    
    # valid patterns
    valid_patterns = [
        r'^\s*\[.*\]\s*$',  # table definition [table]
        r'^\s*\[\[.*\]\]\s*$',  # array table definition [[table]]
        r'^\s*[a-zA-Z_][a-zA-Z0-9_-]*\s*=.*$',  # simple key-value pair
        r'^\s*"[^"]+"\s*=.*$',  # string key
        r'^\s*\'[^\']+\'\s*=.*$',  # single-quoted string key
        r'^\s*#.*$',  # comment
        r'^\s*$',  # empty line
    ]
    compiled_patterns = [re.compile(pattern) for pattern in valid_patterns]
    
    for line in lines:
        # check if the line matches any of the valid patterns
        is_valid = any(pattern.match(line) for pattern in compiled_patterns)
        
        if is_valid:
            cleaned_lines.append(line)
        else:
            print(f"Skip row: {line.strip()}")
    
    output = '\n'.join(cleaned_lines)
    # print(output)
    # print('======================================')
    return output


def convert_risk_labels(risk_labels):
    # mapping risk category labels (str) to ids (int), e.g., Illegal Activity (IA) => 1
    policy_convertor = PolicyHub.default
    risk_label_ids = [policy_convertor.risk_id(risk) for risk in risk_labels]
    # return as torch tensor (long)
    return torch.tensor(risk_label_ids)

def convert_risk_id_to_category(risk_id):
    policy_convertor = PolicyHub.default
    return policy_convertor.risk_name(risk_id)
    

def convert_labels(y):
    """
    Convert the original y to the format required by CrossEntropyLoss
    """
    if y.ndim > 1 and y.shape[1] == 1:
        y = y.squeeze(1)
    y = y.long()
    return y


def find_files(directory:str, file_pattern:str):
    files = []
    for file in Path(directory).glob(file_pattern):
        files.append(str(file))
    return sorted(files)


def calculate_statistics(dataset, show_statistics=True):
    # statistics for dataset (calculate the count for each risk type)
    risk_count = {}
    for item in dataset:
        risk = item['risk']
        if risk not in risk_count:
            risk_count[risk] = 0
        risk_count[risk] += 1
    if show_statistics:
        print("-----------Risk Count------------")
        for risk in sorted(list(risk_count.keys())):
            print(f"'{risk}': {risk_count[risk]},")
        print("---------------------------------")
    return risk_count



def calculate_metrics(y_pred: np.ndarray, y_true: np.ndarray, y_risk: np.ndarray, pos_label=1):
    # Overall Metrics
    # 1. Accuracy
    acc = accuracy_score(y_true, y_pred)
    # 2. Precision, Recall, F1 (for harmful samples: y_true == 1)
    prec = precision_score(y_true, y_pred, pos_label=pos_label)
    rec = recall_score(y_true, y_pred, pos_label=pos_label)
    f1 = f1_score(y_true, y_pred, pos_label=pos_label)

    # 3. Disturbance Rate = FPR = FP / (FP + TN), only for dataset that has benign samples (y_true==0)
    if (y_true == 0).sum() > 0:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        fpr = fp / (fp + tn) # benign samples: y_true == 0 misclassified as harmful y_pred == 1
    else:
        fpr = np.nan
    
    metrics = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'fpr': fpr,
        'count-benign': (y_true == 0).sum(),
        'count-harmful': (y_true == 1).sum()
    }

    # Metrics per risk category
    group_metrics = {}
    risks = np.unique(y_risk[y_true == 1])
    for risk in risks:
        risk_mask = (y_risk == risk)
        risk_pred = y_pred[risk_mask]
        risk_true = y_true[risk_mask]
        group_metrics[risk] = {
            'accuracy': accuracy_score(risk_true, risk_pred),
            'precision': precision_score(risk_true, risk_pred, pos_label=pos_label),
            'recall': recall_score(risk_true, risk_pred, pos_label=pos_label),
            'f1': f1_score(risk_true, risk_pred, pos_label=pos_label),
            'count': risk_mask.sum()
        }
    return metrics, group_metrics


def report_metrics(results, display=False, dataset_stat=None):
    # result: list of dict, each item is a dict with keys 
    #   'pred_toxicity' ('str': safe|unsafe)
    #   'pred_risk' ('str')
    #   'gt_toxicity' (0/1)
    #   'gt_risk' ('str' or int)
    # here, the positive sample is that the sample is unsafe/toxic, and the negative sample is that the sample is safe/benign
    # calculate the accuracy, precision, recall, f1 score, false-positive-rate (Disturbance Rate) with sklearn.metrics
    y_true = [item['gt_toxicity'] for item in results] # 0 / 1
    y_risk = [item['gt_risk'] for item in results]
    y_pred = [item['pred_toxicity'] for item in results] # 'safe'/'unsafe' or 0/1

    if len(y_pred) == 0: # no valid prediction results
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'fpr': 0.0,
            'count-benign': 0,
            'count-harmful': 0
        }, None
    if isinstance(y_pred[0], str):
        assert all(pred.lower() in ['safe', 'unsafe'] for pred in np.unique(y_pred).tolist()), f"Invalid y_pred: {np.unique(y_pred)}"
        y_pred = [1 if pred.lower() == 'unsafe' else 0 for pred in y_pred]
    
    # invalid dataset_stat: {'benign_count': 100, 'harmful_count': 100, 'risk_counts': {'IA': 100, ...}}
    # count the number of safe and unsafe samples that are not detected correctly
    # if invalid_counts is not None, simulate the failures in the y_pred, and calculate the metrics
    if dataset_stat is not None:
        from collections import Counter

        benign_count = dataset_stat['benign_count']
        harmful_count = dataset_stat['harmful_count']
        risk_counts = dataset_stat['risk_counts']
        valid_benign_count = (np.array(y_true)==0).sum()
        valid_harmful_count = (np.array(y_true)==1).sum()
        valid_risk_count = {risk: count for risk, count in Counter(y_risk).items()}
        invalid_y_true = [0] * (benign_count-valid_benign_count) + [1] * (harmful_count-valid_harmful_count)
        invalid_y_pred = [1] * (benign_count-valid_benign_count) + [0] * (harmful_count-valid_harmful_count)
        invalid_y_risk = ['BNI'] * (benign_count-valid_benign_count)
        for risk, count in risk_counts.items():
            if risk == 'BNI': continue
            invalid_y_risk.extend([risk] * (count - valid_risk_count.get(risk, 0)))
            
        y_true.extend(invalid_y_true)
        y_pred.extend(invalid_y_pred)
        y_risk.extend(invalid_y_risk)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_risk = np.array(y_risk)
    # if all samples are benign, set pos_label to 0
    pos_label = 0 if (y_true == 0).all() else 1
    metrics, group_metrics = calculate_metrics(y_pred, y_true, y_risk, pos_label=pos_label)

    if display:
        print("=========Metrics=========")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        if group_metrics:
            print("-------------------------")
            for risk, mc in group_metrics.items():
                print(f"{risk}[{mc['count']}]: Accuracy: {mc['accuracy']:.4f}, Precision: {mc['precision']:.4f}, Recall: {mc['recall']:.4f}, F1: {mc['f1']:.4f}")
        print("=========================")

    return metrics, group_metrics


def load_PIL_image(img):
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, str):
        url_or_path = img
        if url_or_path.startswith('http'):
            return Image.open(requests.get(url_or_path, stream=True).raw)
        elif os.path.exists(url_or_path):
            return Image.open(url_or_path)
        else:
            raise ValueError(f"Invalid image path: {url_or_path}")
    elif isinstance(img, bytes):
        return Image.open(BytesIO(img))
    else:
        raise ValueError(f"Invalid image type: {type(img)}")

def save_PIL_image(img, path):
    # get basedir of path, if the path is not exist, create it
    tgt_dir = os.path.dirname(path)
    filename = os.path.basename(path)
    if not os.path.exists(tgt_dir):
        os.makedirs(tgt_dir)
    # Error 95 Operation not supported?
    tmp_dir = "./temp/pil_tmp"
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    tmp_path = os.path.join(tmp_dir, filename)

    # First, save to a savable temp directory, then copy to target path
    img.save(tmp_path)
    shutil.copy(tmp_path, path)
    # remove temp file
    os.remove(tmp_path)

    

def PIL_img_ext(img: Image.Image):
    if img.format == 'PNG' or img.format is None:
        return 'png'
    elif img.format == 'JPEG':
        return 'jpg'
    elif img.format == 'WEBP':
        return 'webp'
    elif img.format == 'BMP':
        return 'bmp'
    elif img.format == 'GIF':
        return 'gif'
    else:
        raise ValueError(f"Invalid image format: {img},{img.format}")


def to_rgb(pil_image: Image.Image) -> Image.Image:
    if pil_image.mode == "RGBA":
        white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
        white_background.paste(pil_image, mask=pil_image.split()[3])  # Use alpha channel as mask
        return white_background
    else:
        return pil_image.convert("RGB")



def has_audio_track(video_path):
    from moviepy import VideoFileClip
    try:
        clip = VideoFileClip(video_path)
        # Check if the video has an audio stream
        return clip.audio is not None
    except Exception as e:
        print(f"Error: {e}")
        return False



def cosine_similarity_matrix(x1, x2):
    # x1, x2 shape: (N, D)
    # normalize the embeddings first
    x1_norm = F.normalize(x1, p=2, dim=1)  # (N, D)
    x2_norm = F.normalize(x2, p=2, dim=1)  # (N, D)
    
    # calculate cosine similarity matrix
    similarity_matrix = torch.mm(x1_norm, x2_norm.t())  # (N, N)
    return similarity_matrix


def calculate_u_score(x1, x2):
    # x1, x2 shape (N, D)
    # calculate the cosine similarity matrix (N, N)
    sim_matrix = cosine_similarity_matrix(x1.float(), x2.float()) # conver to float32 by .float()
    
    # U-score defines as the average score of diagonal elements minus the average of all the non-diagonal elements
    # i.e., u-score = (1/N) * sum(sim_matrix[i][i] for i in range(N)) - (1/(N*(N-1))) * sum(sim_matrix[i][j] for i in range(N) for j in range(N) if i != j)
    diag_sim_scores = torch.diag(sim_matrix)
    N = sim_matrix.shape[0]
    u_score = (1 / N) * torch.sum(diag_sim_scores) - (1 / (N * (N - 1))) * (torch.sum(sim_matrix) - torch.sum(diag_sim_scores))
    return u_score.item()  # return as a scalar value


def plot_uscore(n_layers, u_scores, model_name):
    layers = np.arange(n_layers)
    
    plt.figure(figsize=(10, 5))
    plt.plot(layers, u_scores, marker='o', linestyle='-', color='b')
    plt.title('U-scores for each layer')
    plt.xlabel('Layer Index')
    plt.ylabel('U-score')
    plt.xticks(layers)
    plt.savefig(f'u_scores_{model_name}.png')
    plt.close()


def display_multimedia(sample, show_labels=True, max_width="800px"):
    """
    Display multimedia sample data in Jupyter Notebook
    
    Args:
        sample (dict): A dictionary containing multimedia data, supporting the following fields:
            - 'txt': Text string
            - 'image': PIL.Image.Image object or image path (str/Path)
            - 'audio': Audio file path (str/Path)
            - 'video': Video file path (str/Path)
        show_labels (bool):  Whether to display field labels (default: True)
        max_width (str): Maximum width of multimedia elements (default: "800px")
    """
    from IPython.display import (
        display, 
        HTML, Audio, Video, 
        Image as IPImage
    )

    if not isinstance(sample, dict):
        raise TypeError(f"sample must be of type dict, current type: {type(sample)}")
    
    displayed = False
    # Text display
    if 'txt' in sample and sample['txt'] is not None:
        if show_labels:
            display(HTML(f"<div style='margin: 10px 0; padding: 8px; background-color: #f0f8ff; border-left: 4px solid #2196F3;'><strong>📝 Text:</strong></div>"))
        display(HTML(f"<div style='margin: 5px 0; white-space: pre-wrap;'>{sample['txt']}</div>"))
        displayed = True
    
    # Image display
    if 'img' in sample and sample['img'] is not None:
        img = sample['img']
        try:
            if isinstance(img, Image.Image):
                # Display PIL.Image directly
                if show_labels:
                    display(HTML(f"<div style='margin: 15px 0 5px 0;'><strong>🖼️ Image (PIL):</strong></div>"))
                display(img)
            elif isinstance(img, (str, Path)):
                path = Path(img)
                if not path.exists():
                    raise FileNotFoundError(f"Image path does not exist: {path}")
                if show_labels:
                    display(HTML(f"<div style='margin: 15px 0 5px 0;'><strong>🖼️ Image:</strong> {path.name}</div>"))
                display(IPImage(filename=str(path), width=max_width))
            else:
                raise TypeError(f"Unsupported image type: {type(img)}, only PIL.Image or path string is supported")
            displayed = True
        except Exception as e:
            display(HTML(f"<div style='color: #d32f2f; margin: 10px 0;'>❌ Image Error: {e}</div>"))
    
    # Audio display
    if 'audio' in sample and sample['audio'] is not None:
        try:
            path = Path(sample['audio'])
            if not path.exists():
                raise FileNotFoundError(f"Audio path does not exist: {path}")
            if show_labels:
                display(HTML(f"<div style='margin: 15px 0 5px 0;'><strong>🔊 Audio:</strong> {path.name}</div>"))
            display(Audio(filename=str(path), autoplay=False))
            displayed = True
        except Exception as e:
            display(HTML(f"<div style='color: #d32f2f; margin: 10px 0;'>❌ Audio Error: {e}</div>"))
    
    # Video display
    if 'video' in sample and sample['video'] is not None:
        try:
            path = Path(sample['video'])
            if not path.exists():
                raise FileNotFoundError(f"Video path does not exist: {path}")
            if show_labels:
                display(HTML(f"<div style='margin: 15px 0 5px 0;'><strong>🎬 Video:</strong> {path.name}</div>"))
            # Use Video component (supports controls)
            display(Video.from_file(str(path), embed=True, width=max_width, html_attributes="controls muted"))
            displayed = True
        except Exception as e:
            # Fallback: use HTML5 video tag
            try:
                rel_path = os.path.relpath(path)
                html = f"""
                <video width="{max_width}" controls>
                    <source src="{rel_path}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
                """
                display(HTML(html))
                displayed = True
            except Exception as e2:
                display(HTML(f"<div style='color: #d32f2f; margin: 10px 0;'>❌ Video Error: {e} | Fallback Error: {e2}</div>"))
    
    # Empty sample warning
    if not displayed:
        display(HTML("<div style='color: #ff9800; margin: 10px 0;'>⚠️ No valid multimedia fields in the sample (txt/image/audio/video)</div>"))