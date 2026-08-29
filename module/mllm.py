import os
import time
import copy
import importlib
import torch
import torch.distributed as dist
import numpy as np
import soundfile as sf
from PIL import Image
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
)

from typing import NamedTuple
from transformers.dynamic_module_utils import get_class_from_dynamic_module


class BaseLLM:
    def __init__(self, model_path, device="cuda", enable_flash_attention=False):
        self.model_path = model_path
        self.device = str(device) if isinstance(device, torch.device) else device
        self.enable_flash_attention = enable_flash_attention

        # implement in specific LLM model interfaces
        model, processor = self.load()
        self.model = model
        self.processor = processor

        self.inference_count = 0
        self.consume_time = 0.0
    
    def load(self):
        # Load from pretrained model
        raise NotImplementedError
    
    def close(self):
        pass

    def calculate_query_token_length(self, text):
        assert self.processor is not None
        tokenizer = self.processor.tokenizer if hasattr(self.processor, "tokenizer") else self.processor
        return len(tokenizer.encode(text, add_special_tokens=False))
    
    def generate(self, messages, max_token=1024):
        raise NotImplementedError
    
    def generate_llm(self, txt_prompt, max_token=512, sys_prompt=None):
        messages = self.build_messages(txt=txt_prompt, sys_prompt=sys_prompt)
        return self.generate(messages, max_token=max_token)

    def generate_vlm(self, txt_prompt, img_prompt, max_token=256, sys_prompt=None):
        messages = self.build_messages(txt=txt_prompt, img=img_prompt, sys_prompt=sys_prompt)
        return self.generate(messages, max_token=max_token)

    def generate_olm(self, txt_prompt, img_promt, audio_prompt, video_prompt, max_token=256, user_inst=None, sys_prompt=None):
        messages = self.build_messages(txt=txt_prompt, img=img_promt, audio=audio_prompt, video=video_prompt, user_inst=user_inst, sys_prompt=sys_prompt)
        return self.generate(messages, max_token=max_token)
    
    ## Inference(*): Obtain the hidden states
    def inference(self, txt_prompt=None, img_prompt=None, audio_prompt=None, video_prompt=None, response=None):
        raise NotImplementedError
    
    def build_messages(self, txt=None, img=None, audio=None, video=None, user_inst=None, sys_prompt=None, response=None):
        message = []
        if sys_prompt: message.append({ "role": "system", "content": [{ "type": "text", "text": sys_prompt }]})
        
        content = []
        if user_inst is not None: content.append({"type": "text", "text": user_inst}) # user instruction: before all the other content (image, etc), for Omniguard only
        # better in this order
        if audio is not None: content.append({"type": "audio", "audio": audio})
        if img is not None: content.append({"type": "image", "image": img})
        if video is not None:
            content.append({"type": "video", "video": video, "fps": 1.0, "max_pixels": 100352})
        if txt is not None: content.append({"type": "text", "text": txt}) # other user text

        assert len(content) > 0, "No content to generate"
        message.append({ "role": "user", "content": content })

        # if response: message.append({ "role": "assistant", "content": [ {"type": "text", "text": response}]})
        # print(message)
        return message

def get_sampling_params(
    temperature=0.0,
    top_p=1.0,
    top_k=50,
    max_new_tokens=256,
    tokenizer=None,
    stop_token_ids=None,
):
    from vllm import SamplingParams
    params = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_tokens": max_new_tokens,
    }
    if stop_token_ids is not None:
        params["stop_token_ids"] = stop_token_ids
    elif tokenizer is not None:
        params["stop_token_ids"] = [tokenizer.eos_token_id]
    return SamplingParams(**params)



class Qwen2_5Omni_vllm(BaseLLM):

    def __init__(self, model_path, max_num_seqs=5,):
        self.max_num_seqs = max_num_seqs
        super().__init__(model_path, enable_flash_attention=True)

    def load(self):
        from transformers import Qwen2_5OmniProcessor
        from vllm import LLM
        
        processor = Qwen2_5OmniProcessor.from_pretrained(self.model_path, fix_mistral_regex=True)
        llm = LLM(
            model = self.model_path,
            # tokenizer='./model/fixed_tokenizer',
            dtype='bfloat16',
            trust_remote_code=True, 
            gpu_memory_utilization=0.8,
            max_model_len = 32768, # 32768
            max_num_seqs=self.max_num_seqs,
            limit_mm_per_prompt={"audio": 1, "image": 1, "video": 1},
            seed=42,
        )
        return llm, processor
    
    def close(self):
        m = self.model
        if m is not None:
            m.llm_engine.engine_core.shutdown()
            if dist.is_initialized():
                dist.destroy_process_group()
            del m

    def message_to_vllm_input(self, message):
        from qwen_omni_utils import process_mm_info

        prompt = self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        audios, images, videos = process_mm_info(message, use_audio_in_video=False)
        inputs = {
            "prompt": prompt,
            "multi_modal_data": {},
            "mm_processor_kwargs": {
                "use_audio_in_video": False,
            }
        }
        if images is not None:
            inputs['multi_modal_data']['image'] = images
        if videos is not None:
            inputs['multi_modal_data']['video'] = videos
        if audios is not None:
            inputs['multi_modal_data']['audio'] = audios
        return inputs
    
    def generate(self, messages, max_token=1024):
        if isinstance(messages[0], dict):
            messages = [messages]
        sampling_params = get_sampling_params(top_k=50, max_new_tokens=max_token)
        inputs_batch = []
        for message in messages:
            inputs = self.message_to_vllm_input(message)
            inputs_batch.append(inputs)
        try :
            outputs = self.model.generate(inputs_batch, sampling_params=sampling_params)
            outputs_batch = []
            for output in outputs:
                outputs_batch.append(output.outputs[0].text)
        except ValueError as e:
            if 'longer than the maximum model length' in str(e):
                outputs_batch = ["I can't assist with such a long prompt."] * len(messages)
            else:
                raise e
        return outputs_batch


class Qwen3_Omni_vllm(Qwen2_5Omni_vllm):
    def load(self):
        from transformers import Qwen3OmniMoeProcessor
        from vllm import LLM
        processor = Qwen3OmniMoeProcessor.from_pretrained(self.model_path)
        llm = LLM(
            model = self.model_path,
            # tokenizer='./model/fixed_tokenizer',
            dtype='bfloat16',
            trust_remote_code=True, 
            gpu_memory_utilization=0.9,
            max_model_len = 32768, # 32768
            max_num_seqs=self.max_num_seqs,
            limit_mm_per_prompt={"audio": 1, "image": 1, "video": 1},
            seed=42,
        )
        return llm, processor
        

class Qwen2VL(BaseLLM):
    def __init__(self, model_path, device="cuda"):
        super().__init__(model_path, device, enable_flash_attention=False)

    def load(self):
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            attn_implementation="flash_attention_2" if self.enable_flash_attention else None,
            device_map=self.device,   # set to cuda, not auto
            trust_remote_code=True,
        )
        model.eval()
        return model, processor

    def generate(self, messages, max_token=1024):
        from qwen_vl_utils import process_vision_info
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_token)
            output_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output_ids)]
            output_text = self.processor.batch_decode(output_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return output_text

    def inference(self, txt_prompt=None, img_prompt=None, audio_prompt=None, video_prompt=None, response=None):
        from qwen_vl_utils import process_vision_info

        messages = self.build_messages(txt=txt_prompt, img=img_prompt)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)
        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True)
        return outputs


class Qwen2_5Omni(BaseLLM):
    Default_System_Prompt = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."

    def __init__(self, model_path, device="cuda", enable_flash_attention=False):
        super().__init__(model_path, device, enable_flash_attention=enable_flash_attention)
        self.reload_tps = 999.0
    
    def load(self):
        from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor

        processor = Qwen2_5OmniProcessor.from_pretrained(self.model_path)
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            attn_implementation="flash_attention_2" if self.enable_flash_attention else None,
            device_map=self.device,   # set to cuda, not auto
            trust_remote_code=True,
        )
        if hasattr(model, 'disable_talker'):
            model.disable_talker()
        model.eval()
        return model, processor
    
    def thinker_model(self):
        # return self.model
        if hasattr(self.model, 'thinker'):
            return self.model.thinker
        else:
            return self.model

    def generate(self, messages, max_token=1024):
        from qwen_omni_utils import process_mm_info
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # print(text)
        audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
        inputs = self.processor(
            text=text, 
            audio=audios, 
            images=images, 
            videos=videos,
            return_tensors="pt", 
            padding=True, 
            use_audio_in_video=False
        ).to(self.model.device).to(self.model.dtype)
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, use_audio_in_video=False, max_new_tokens=max_token)
            output_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output_ids)]
            output_text = self.processor.batch_decode(output_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return output_text

    def inference(self, txt_prompt=None, img_prompt=None, audio_prompt=None, video_prompt=None, response=None):
        from qwen_omni_utils import process_mm_info
        # TODO add system prompt maybe
        messages = self.build_messages(txt=txt_prompt, img=img_prompt, audio=audio_prompt, video=video_prompt, response=response)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
        inputs = self.processor(
            text=text, 
            audio=audios, 
            images=images, 
            videos=videos,
            return_tensors="pt", 
            padding=True, 
            use_audio_in_video=False
        ).to(self.model.device).to(self.model.dtype)
        t0 = time.time()
        with torch.inference_mode():
            outputs = self.thinker_model()(**inputs, output_hidden_states=True, use_audio_in_video=False)
        t1 = time.time()
        self.consume_time += (t1 - t0)
        self.inference_count += 1

        if self.consume_time / self.inference_count > self.reload_tps:
            print(f"Reloading model, consume time: {self.consume_time / self.inference_count}s")
            old_model = self.model
            old_processor = self.processor
            self.model = None
            self.processor = None
            del old_model, old_processor
            model, processor = self.load()
            self.model = model
            self.processor = processor
            self.consume_time = 0.0
            self.inference_count = 0
        return outputs


class Qwen3_Omni(Qwen2_5Omni):
    def __init__(self, model_path, device="cuda", enable_flash_attention=False):
        super().__init__(model_path, device, enable_flash_attention)
        self.reload_tps = 999


    def load(self):
        from transformers import Qwen3OmniMoeThinkerForConditionalGeneration, AutoProcessor
        # processor = Qwen3OmniMoeProcessor.from_pretrained(self.model_path)
        # min_pixels, max_pixels = 128*28*28, 768*28*28
        max_pixels = 384*28*28
        
        processor = AutoProcessor.from_pretrained(self.model_path, max_pixels=max_pixels)
        # print(processor.max_pixels, processor.min_pixels)
        model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2" if self.enable_flash_attention else None,
            device_map=self.device,   # set to auto maybe
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        model.eval()
        return model, processor
    
    def thinker_model(self):
        return self.model


def patch_transformers_dynamic_cache_for_minicpm():
    try:
        from transformers.cache_utils import DynamicCache
        if not hasattr(DynamicCache, "seen_tokens"):
            DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
    except Exception:
        pass


class MiniCPM_o_4_5(BaseLLM):
    def __init__(self, model_path, device="cuda", enable_flash_attention=False):
        super().__init__(model_path, device, enable_flash_attention=False)
        self.reload_tps = 999.0

    def load(self):
        patch_transformers_dynamic_cache_for_minicpm()
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if self.enable_flash_attention else "sdpa",
            torch_dtype=torch.bfloat16,
            init_vision=True,
            init_audio=True,
            init_tts=False,
            device_map=self.device,
        )
        model.eval()
        model.prepare_processor(tokenizer=tokenizer)
        # self.tokenizer = tokenizer
        self.normalize_content = getattr(importlib.import_module(type(model).__module__), "normalize_content")
        return model, model.processor

    def build_messages(self, txt=None, img=None, audio=None, video=None, user_inst=None, sys_prompt=None, response=None):
        """Build MiniCPM messages while preserving native multimodal objects.

        Align with the Qwen-side design philosophy: do not force every input into
        a path-based schema too early. MiniCPM's normalize_content(...) can accept
        both OpenAI-style *_url dicts and native objects such as PIL.Image /
        np.ndarray / lists of them.
        """
        message = []
        if sys_prompt: message.append({"role": "system", "content": [sys_prompt]})

        content = []
        if user_inst is not None: content.append(user_inst)

        if audio is not None:
            assert isinstance(audio, (str, np.ndarray)), "audio data type not in str (url) or np.ndarray"
            if isinstance(audio, str): content.append({"type": "audio_url", "audio_url": {"url": audio}})
            else: content.append(audio)

        if img is not None:
            assert isinstance(img, (str, Image.Image)), "image data type not in str (url) or Image.Image"
            if isinstance(img, str): content.append({"type": "image_url", "image_url": {"url": img}})
            else: content.append(img)

        if video is not None:
            assert isinstance(video, (str, list, tuple)), "video data type not in str (url) or list/tuple of Image.Image"
            if isinstance(video, str):
                content.append({"type": "video_url", "video_url": {"url": video, "use_audio": False}}) # default not using audio track in video
            else:
                assert all([isinstance(it, Image.Image) for it in video]), "video data type not in str (url) or list/tuple of Image.Image"
                content.extend(video)

        if txt is not None: content.append(txt) # append the text query at last

        assert len(content) > 0, "No content to generate"
        message.append({"role": "user", "content": content})

        # if response: message.append({"role": "assistant", "content": [response]})
        return message

    def _append_minicpm_content(self, item, prompt_parts, images, audios, audio_parts, msg_idx):
        if isinstance(item, Image.Image):
            images.append(item)
            prompt_parts.append("(<image>./</image>)")
        elif isinstance(item, np.ndarray):
            audios.append(item)
            audio_parts.append(msg_idx)
            prompt_parts.append("(<audio>./</audio>)")
        elif isinstance(item, str):
            prompt_parts.append(item)
        else:
            raise TypeError(f"Unsupported MiniCPM content type: {type(item)}")

    def _prepare_prompt_and_mm_inputs(self, messages, add_generation_prompt=True, enable_thinking=False, omni_mode=False):
        copy_msgs = copy.deepcopy(messages)
        input_images = []
        input_audios = []
        audio_parts = []

        for i, msg in enumerate(copy_msgs):
            content = self.normalize_content(msg["content"])
            prompt_parts = []
            for item in content:
                self._append_minicpm_content(item, prompt_parts, input_images, input_audios, audio_parts, i)
            msg["content"] = "\n".join(prompt_parts) if not omni_mode else "".join(prompt_parts)

        chat_tokenizer = self.processor.tokenizer if self.processor is not None and hasattr(self.processor, "tokenizer") else self.tokenizer
        prompt = chat_tokenizer.apply_chat_template(
            copy_msgs,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            use_tts_template=False,
            enable_thinking=enable_thinking,
        )
        return prompt, input_images, input_audios, audio_parts

    def generate(self, messages, max_token=1024):
        if isinstance(messages[0], dict):
            messages = [messages]

        outputs = []
        for message in messages:
            answer = self.model.chat(
                msgs=message,
                # tokenizer=self.tokenizer,
                # processor=self.processor,
                use_image_id=False,
                max_slice_nums=1,
                max_new_tokens=max_token,
                use_tts_template=False,
                generate_audio=False,
                enable_thinking=False,
                do_sample=False,
            )
            outputs.append(answer)
        return outputs

    def inference(self, txt_prompt=None, img_prompt=None, audio_prompt=None, video_prompt=None, response=None):
        messages = self.build_messages(
            txt=txt_prompt,
            img=img_prompt,
            audio=audio_prompt,
            video=video_prompt,
            response=response,
        )
        prompt, input_images, input_audios, audio_parts = self._prepare_prompt_and_mm_inputs(
            messages,
            add_generation_prompt=True,
            enable_thinking=False,
            omni_mode=False,
        )
        model_inputs = self.processor(
            [prompt],
            [input_images],
            [input_audios],
            # [audio_parts] if len(input_audios) > 0 else None,
            [audio_parts],
            max_slice_nums=1, # 2 for HD
            use_image_id=None,
            stream_input=False,
            return_tensors="pt",
            max_length=8192,
        ).to(self.model.device)
        model_inputs = model_inputs.to(self.model.device)
        if "image_sizes" in model_inputs:
            model_inputs.pop("image_sizes")
        if "position_ids" not in model_inputs and "attention_mask" in model_inputs:
            position_ids = model_inputs["attention_mask"].long().cumsum(-1) - 1
            position_ids.masked_fill_(model_inputs["attention_mask"] == 0, 1)
            model_inputs["position_ids"] = position_ids.clone(memory_format=torch.contiguous_format)
        t0 = time.time()
        # sample a sequence of hidden states (not used)
        # generation_config = self.model.prepare_generation_config(do_sample=False, max_new_tokens=4096, min_new_tokens=0)
        # generation_config.pop("max_new_tokens", None)
        # model_inputs.pop("image_sizes")
        # with torch.inference_mode():
            # res, outputs = self.model.generate(
            #     **model_inputs,
            #     tokenizer=self.processor.tokenizer,
            #     max_new_tokens=4096,
            #     **generation_config,
            # )
        # only extract the next hidden states [recommend]
        with torch.inference_mode():
            outputs = self.model(data=model_inputs, output_hidden_states=True, return_dict=True)
        t1 = time.time()
        self.consume_time += (t1 - t0)
        self.inference_count += 1
        return outputs


class MiniCPM_o_4_5_vllm(MiniCPM_o_4_5):
    def __init__(self, model_path, max_num_seqs=5):
        self.max_num_seqs = max_num_seqs
        super().__init__(model_path, enable_flash_attention=True)

    def load(self):
        from vllm import LLM
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        llm = LLM(
            model=self.model_path,
            trust_remote_code=True,
            gpu_memory_utilization=0.8,
            max_model_len=8196,  # 32768
            max_num_seqs=self.max_num_seqs,
            limit_mm_per_prompt={"image": 1, "audio": 1, "video": 1},
            seed=42,
        )
        self.tokenizer = tokenizer
        stop_tokens = ['<|im_end|>', '<|endoftext|>']
        self.stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
        try:
            model_cls = get_class_from_dynamic_module("modeling_minicpmo.MiniCPMO", self.model_path)
            self.normalize_content = getattr(importlib.import_module(model_cls.__module__), "normalize_content")
        except Exception:
            self.normalize_content = None
        return llm, tokenizer

    def close(self):
        m = self.model
        if m is not None:
            m.llm_engine.engine_core.shutdown()
            if dist.is_initialized():
                dist.destroy_process_group()
            del m

    def message_to_vllm_input(self, message):
        prompt, input_images, input_audios, _ = self._prepare_prompt_and_mm_inputs(
            message,
            add_generation_prompt=True,
            enable_thinking=False,
            omni_mode=False,
        )
        inputs = {
            "prompt": prompt,
            "multi_modal_data": {},
            "mm_processor_kwargs": {
                "max_slice_nums": 1,
                "use_image_id": False,
            }
        }
        if len(input_images) > 0:
            inputs["multi_modal_data"]["image"] = input_images
        if len(input_audios) > 0:
            inputs["multi_modal_data"]["audio"] = input_audios
        return inputs

    def generate(self, messages, max_token=1024):
        if isinstance(messages[0], dict):
            messages = [messages]
        # sampling_params = get_sampling_params(top_k=50, max_new_tokens=max_token, stop_token_ids=self.stop_token_ids)
        sampling_params = get_sampling_params(top_k=50, max_new_tokens=max_token, tokenizer=self.tokenizer)
        inputs_batch = [self.message_to_vllm_input(message) for message in messages]
        try:
            outputs = self.model.generate(inputs_batch, sampling_params=sampling_params)
            return [output.outputs[0].text for output in outputs]
        except ValueError as e:
            if "longer than the maximum model length" in str(e):
                return ["I can't assist with such a long prompt."] * len(messages)
            raise e


class Phi4Multimodal(BaseLLM):
    def __init__(self, model_path, device="cuda", enable_flash_attention=True):
        super().__init__(model_path, device=device, enable_flash_attention=enable_flash_attention)

    def _patch_phi4mm_for_peft(self):
        try:
            phi4mm_model_cls = get_class_from_dynamic_module("modeling_phi4mm.Phi4MMModel", self.model_path)
        except Exception:
            return

        if hasattr(phi4mm_model_cls, "prepare_inputs_for_generation"):
            return

        def _prepare_inputs_for_generation(self, *args, **kwargs):
            return kwargs

        phi4mm_model_cls.prepare_inputs_for_generation = _prepare_inputs_for_generation

    def load(self):
        # self._patch_phi4mm_for_peft()
        processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype="auto", # torch.bfloat16,
            _attn_implementation="flash_attention_2" if self.enable_flash_attention else None,
            device_map=self.device,
        )
        model.eval()
        self.generation_config = GenerationConfig.from_pretrained(self.model_path, "generation_config.json")
        # self.generation_config = GenerationConfig.from_pretrained(self.model_path)
        return model, processor

    def _ensure_list(self, value):
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    def _load_phi_image(self, img):
        if isinstance(img, Image.Image):
            return img.convert("RGB")
        if isinstance(img, str):
            # if img.startswith("http://") or img.startswith("https://"):
            #     import requests
            #     return Image.open(requests.get(img, stream=True).raw).convert("RGB")
            return Image.open(img).convert("RGB")
        raise ValueError(f"Unsupported Phi-4 image input: {type(img)}")

    def _load_phi_audio(self, audio):
        # be sure in the format of (array, sample_rate: int)
        if isinstance(audio, tuple) and isinstance(audio[0], np.ndarray) and isinstance(audio[1], int):
            return audio
        if isinstance(audio, str):
            return sf.read(audio)
        if isinstance(audio, np.ndarray):
            return (audio, 16000) # default 16000 (by librosa)
        raise ValueError(f"Unsupported Phi-4 audio input: {type(audio)}")

    def build_messages(self, txt=None, img=None, audio=None, video=None, user_inst=None, sys_prompt=None, response=None):
        if video is not None:
            raise NotImplementedError("Phi-4-multimodal-instruct does not support video in this wrapper.")
        messages = []
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})

        content = []
        if user_inst is not None: content.append(user_inst)

        images = [self._load_phi_image(item) for item in self._ensure_list(img)]
        audios = [self._load_phi_audio(item) for item in self._ensure_list(audio)]

        image_idx = 0
        audio_idx = 0
        for _ in images:
            image_idx += 1
            content.append(f"<|image_{image_idx}|>")
        for _ in audios:
            audio_idx += 1
            content.append(f"<|audio_{audio_idx}|>")
        if txt is not None:
            content.append(txt)

        assert len(content) > 0, "No content to generate"
        messages.append({
            "role": "user",
            "content": "".join(content),
            "images": images,
            "audios": audios,
        })

        # if response: messages.append({"role": "assistant", "content": response})
        return messages

    def _prepare_prompt_and_mm_inputs(self, messages):
        chat = []
        images, audios = [], []
        for message in messages:
            chat.append({
                "role": message["role"],
                "content": message["content"],
            })
            images.extend(message.get("images", []))
            audios.extend(message.get("audios", []))

        prompt = self.processor.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        # print(prompt)
        if prompt.endswith('<|endoftext|>'):
            print("Warning! Encounter remove end of text", prompt)
            prompt = prompt.rstrip('<|endoftext|>')
        # if prompt.endswith(self.processor.tokenizer.eos_token):
        #     prompt = prompt[: -len(self.processor.tokenizer.eos_token)]
        return prompt, images, audios

    def _processor_inputs(self, messages):
        prompt, images, audios = self._prepare_prompt_and_mm_inputs(messages)
        return self.processor(
            text=prompt,
            images=images if len(images) > 0 else None,
            audios=audios if len(audios) > 0 else None,
            return_tensors="pt",
        ).to(self.model.device)

    def generate(self, messages, max_token=1024):
        if isinstance(messages[0], dict):
            messages = [messages]

        outputs = []
        for message in messages:
            inputs = self._processor_inputs(message)
            # generation_kwargs = {
            #     "max_new_tokens": max_token,
            #     "do_sample": False,
            # }
            # if self.generation_config is not None:
            #     generation_kwargs["generation_config"] = self.generation_config
            with torch.inference_mode():
                output_ids = self.model.generate(**inputs, max_new_tokens=max_token, 
                    do_sample=False, generation_config=self.generation_config, num_logits_to_keep=1)
            input_len = inputs["input_ids"].shape[1]
            generate_ids = output_ids[:, input_len:]
            response = self.processor.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            outputs.append(response)
        return outputs

    def inference(self, txt_prompt=None, img_prompt=None, audio_prompt=None, video_prompt=None, response=None):
        messages = self.build_messages(
            txt=txt_prompt,
            img=img_prompt,
            audio=audio_prompt,
            video=video_prompt,
            response=response,
        )
        inputs = self._processor_inputs(messages)
        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
        return outputs


class Phi4Multimodal_vllm(Phi4Multimodal):
    def __init__(self, model_path, max_num_seqs=5):
        self.max_num_seqs = max_num_seqs
        super().__init__(model_path, enable_flash_attention=True)

    def load(self):
        processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        from vllm import LLM
        llm = LLM(
            model=self.model_path,
            dtype="bfloat16",
            trust_remote_code=True,
            gpu_memory_utilization=0.8,
            max_model_len=32768,
            max_num_seqs=self.max_num_seqs,
            # enforce_eager=True,
            limit_mm_per_prompt={"image": 16, "audio": 32},
            seed=42,
        )
        self.tokenizer = tokenizer
        return llm, processor

    def close(self):
        m = self.model
        if m is not None:
            m.llm_engine.engine_core.shutdown()
            if dist.is_initialized():
                dist.destroy_process_group()
            del m

    def message_to_vllm_input(self, message):
        prompt, images, audios = self._prepare_prompt_and_mm_inputs(message)
        inputs = {
            "prompt": prompt,
            "multi_modal_data": {},
        }
        if len(images) > 0:
            # resample images if necessary
            for i, image in enumerate(images):
                if isinstance(image, Image.Image):
                    if image.width > 1024 or image.height > 1024:
                        # resize image
                        print("Warning! Encounter large image", image.size, "resized to <1024x1024")
                        image.thumbnail((1024, 1024))
            inputs["multi_modal_data"]["image"] = images
        if len(audios) > 0:
            inputs["multi_modal_data"]["audio"] = [audio[0] if isinstance(audio, tuple) else audio for audio in audios]
        return inputs

    def generate(self, messages, max_token=1024):
        if isinstance(messages[0], dict):
            messages = [messages]
        sampling_params = get_sampling_params(top_k=50, max_new_tokens=max_token, tokenizer=self.tokenizer)
        inputs_batch = [self.message_to_vllm_input(message) for message in messages]
        try:
            outputs = self.model.generate(inputs_batch, sampling_params=sampling_params)
            return [output.outputs[0].text for output in outputs]
        except ValueError as e:
            if "longer than the maximum model length" in str(e):
                return ["I can't assist with such a long prompt."] * len(messages)
            raise e


def load_llm(model_path, device="cuda", use_vllm=False, parallel_size=4, enable_flash_attn=False) -> BaseLLM:
    # parse the last part of the model_path as the model_name
    base_name = os.path.basename(model_path).lower()
    if "phi" in base_name:
        if use_vllm:
            return Phi4Multimodal_vllm(model_path, max_num_seqs=parallel_size)
        return Phi4Multimodal(model_path, device=device, enable_flash_attention=enable_flash_attn)
    if "minicpm" in base_name:
        if use_vllm:
            return MiniCPM_o_4_5_vllm(model_path, max_num_seqs=parallel_size)
        return MiniCPM_o_4_5(model_path, device=device, enable_flash_attention=enable_flash_attn)
    if "qwen" in base_name:
        if 'qwen3' in base_name:
            if use_vllm:
                return Qwen3_Omni_vllm(model_path, max_num_seqs=parallel_size)
            return Qwen3_Omni(model_path, device=device, enable_flash_attention=enable_flash_attn)
        else:
            if use_vllm:
                return Qwen2_5Omni_vllm(model_path, max_num_seqs=parallel_size)
            return Qwen2_5Omni(model_path, device=device, enable_flash_attention=enable_flash_attn)
    raise ValueError(f"Unknown model: {model_path}") 
