import json
import glob
import math
import librosa
import torch

from collections import Counter
from typing import List, Union, Callable, Dict
import numpy as np
from PIL import Image

from pathos.pools import ThreadPool
from module.policy import *


class ConfigItem:
    def __init__(self, key:str, 
                 load_fn:Union[Callable, List[Callable]],
                 policy:Policy=None,
                 callback:Callable=None,
                 risk_mapper:Callable=None):
        self.key = key
        self.load_fn = load_fn
        self.load_extra_info_fn = risk_mapper
        self.callback = callback
        
        if policy is None:
            policy = Policy() # default Policy, mapping every risk to UNK
        self.policy = policy
    
    def load_unified(self, preload=True):
        # apply the initial risk mapping
        load_fns = self.load_fn if isinstance(self.load_fn, list) else [self.load_fn]
        dataset = []

        def preload_resource(item:dict):
            
            if preload and isinstance(item.get("img", None), str):
                item["img"] = Image.open(item["img"]) # no need to convert RGB here
            
            # Video cannot be preloaded because the logic in process_mm_info only accepts video paths
            if preload and isinstance(item.get('video', None), str):
                item['video'] = sample_fixed_frames_torch(item['video'], total_frames=12) # maximum up to 12 frames

            if preload and isinstance(item.get("audio", None), str):
                # TODO: use audio in video, extract the audio track from video
                item['audio'] = librosa.load(item['audio'], sr=16000)[0]
            
            return item

        for fn in load_fns:
            # results = [preload_resource(item) for item in fn()]
            results = ThreadPool().map(preload_resource, fn())
            dataset.extend(results)
        
        # apply risk policy mapping
        dataset = apply_policies(dataset, self.policy)
        
        # relabel the dataset with extra informations
        if self.load_extra_info_fn is not None:
            dataset = apply_extra_policies(dataset, self.load_extra_info_fn(), 
                                           drop_policy=self.policy.relabel_drop_policy, 
                                           agreement_policy=self.policy.relabel_agreement_policy)
        
        # filter the dataset by allowed risks
        n = len(dataset)
        dataset = apply_risk_filters(dataset, self.policy.allowed_risks)
        print(f"Filtered {len(dataset)} out of {n} samples")
        
        # more processing steps to do, for example, replace the text prompt
        if self.callback is not None:
            dataset = self.callback(dataset)
            print(f"Processed {len(dataset)} samples")
        return dataset


def sample_fixed_frames_torch(video_path, total_frames=64):
    # import torch
    from torchcodec.decoders import VideoDecoder

    decoder = VideoDecoder(video_path, num_ffmpeg_threads=8)
    video_fps = decoder.metadata.average_fps
    v_len = decoder.metadata.num_frames
    # print(f"{video_path} Video fps: {video_fps}, video length: {v_len}")
    if v_len < total_frames:
        total_frames = v_len
    if decoder.metadata.duration_seconds > 3:
        seconds = math.floor(decoder.metadata.end_stream_seconds - decoder.metadata.begin_stream_seconds)
        actual_frames = int(seconds * video_fps)
    else:
        actual_frames = v_len
    idx = torch.linspace(0, actual_frames - 1, total_frames).round().long().tolist()
    try:
        frames = decoder.get_frames_at(indices=idx).data.numpy()
        return [Image.fromarray(frame.transpose(1,2,0)) for frame in frames]
    except Exception as e:
        print(f"Failed to load video {video_path}")
        raise e


def sample_fixed_frames(video_path, total_frames=64):
    from decord import VideoReader, cpu
    """
    only extract fixed number of frames from the video, no matter how long the video is.    
    """
    vr = VideoReader(video_path, ctx=cpu(0))
    v_len = len(vr)

    if v_len < total_frames:
        total_frames = v_len

    indices = np.linspace(0, v_len - 1, num=total_frames, dtype=int)
    
    frames = vr.get_batch(indices).asnumpy()
    return [Image.fromarray(frame) for frame in frames]


class Splitable:
    def __init__(self, config: ConfigItem, 
                 train_split:Union[float, int]=1.0, 
                 test_split:Union[float, int]=-1,
                 include_in_test:bool=True) -> None:
        self.data_config = config
        self.embedding_file = None   # to be filled later
        self.risk_label_file = None  # to be filled later
        self.train_split = train_split
        self.test_split = test_split
        self.include_in_test = include_in_test

    
    def allowed_risks(self):
        return self.data_config.policy.allowed_risks

    def load_unified(self, preload=True):
        return self.data_config.load_unified(preload=preload)

    def key(self):
        return self.data_config.key



class DatasetConfigs:
    def __init__(self, dataset_configs: List[Splitable]):
        self.configs = {item.key(): item for item in dataset_configs}

    def items(self):
        return self.configs.items()
    
    def __len__(self):
        return len(self.configs)
    
    def __add__(self, other: 'DatasetConfigs') -> 'DatasetConfigs':
        if not isinstance(other, DatasetConfigs):
            raise TypeError("Can only add another DatasetConfigs instance")
        
        combined_configs = list(self.configs.values()) + list(other.configs.values())
        return DatasetConfigs(combined_configs)


def load_extra_label_from_files(filter_pattern:str)->Dict[str, List[str]]:
    extra_infos = {}
    for file in glob.glob(filter_pattern):
        print(f"Loading extra label from {file}")
        with open(file, "r") as f:
            for line in f:
                item = json.loads(line)
                request_prompt = item['prompt']
                if request_prompt not in extra_infos:
                    extra_infos[request_prompt] = []
                if item['safety'] == 'safe':
                    risk_type = Const.BENIGN
                else:
                    risk_type = item['category'].split("/")[1]
                extra_infos[request_prompt].append(risk_type)
    return extra_infos


def apply_risk_filters(dataset, risks: List[str]):
    risks = set(risks)
    filter_dataset = [item for item in dataset if item['risk'] in risks]
    return filter_dataset


def apply_policies(dataset, policy:Policy):
    for item in dataset:
        if 'risk' not in item:
            item['risk'] = policy.unified_risk('unknown')
        else:
            if item['risk'] in [Const.BENIGN, Const.UNKNOWN, Const.CTRL_RELABEL, Const.CTRL_DROP]:
                continue # skip benign, unknown, ctrl_relabel, ctrl_drop
            else:
                item['risk'] = policy.unified_risk(item['risk'])
    return dataset


def apply_extra_policies(dataset, extra_infos: Dict, drop_policy:int, agreement_policy:float):
    for item in dataset:
        if item['risk'] == Const.CTRL_RELABEL: # only relabel this type of items
            if 'request_key' in item and item['request_key'] in extra_infos:
                # reset risk category
                new_risk, agreement = apply_vote(extra_infos[item['request_key']])
                if agreement < agreement_policy: # drop some items that LLM relabel has no agreement
                    item['risk'] = Const.CTRL_DROP
                    continue

                item['risk'] = new_risk
                
                # reset toxicity
                new_toxic = PROMPT_SAFE if item['risk'] == Const.BENIGN else PROMPT_UNSAFE
                if item['toxicity'] != new_toxic:
                    if new_toxic == PROMPT_SAFE and drop_policy in [RelabelPolicy.DROP_TOXIC_TO_BENIGN, RelabelPolicy.DROP_ALL_CHANGE]:
                        item['risk'] = Const.CTRL_DROP
                    elif new_toxic == PROMPT_UNSAFE and drop_policy in [RelabelPolicy.DROP_BENIGN_TO_TOXIC, RelabelPolicy.DROP_ALL_CHANGE]:
                        item['risk'] = Const.CTRL_DROP
                    else:  # DO NOT Drop this chaning, and relabel the toxicity
                        item['toxicity'] = new_toxic
            else:
                item['risk'] = Const.CTRL_DROP
    return dataset


def apply_vote(alist: List[str]):
    counter = Counter(alist)
    total = counter.total()
    risk, agreement = counter.most_common(1)[0]
    return risk, round(agreement/total, 2)
    # return max(alist, key=alist.count)


def apply_drop(dataset):
    return [item for item in dataset if item['risk'] != Const.CTRL_DROP]


def relabel_from(filter_pattern:str):
    return lambda: load_extra_label_from_files(filter_pattern)
