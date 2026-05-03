
from typing import Any, Dict, Optional, List
from .safety_datasets.base import SafetyEvaluator
from transformers import AutoConfig
import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
try:
    from qwen_omni_utils import process_mm_info
except ImportError:
    # Fallback implementation if qwen_omni_utils is not available
    def process_mm_info(conversation, use_audio_in_video=False):
        """
        Extract multimedia information from conversation.
        Returns: (audios, images, videos)
        """
        audios, images, videos = [], [], []
        for msg in conversation:
            if isinstance(msg, dict) and "content" in msg:
                for item in msg["content"]:
                    if isinstance(item, dict):
                        if item.get("type") == "audio" and "audio" in item:
                            audios.append(item["audio"])
                        elif item.get("type") == "image" and "image" in item:
                            images.append(item["image"])
                        elif item.get("type") == "video" and "video" in item:
                            videos.append(item["video"])
        return audios, images, videos

from .utils.prompt import (
    LLAMA_GUARD_3_CATEGORY_SHORT_NAME_PREFIX,
    build_custom_prompt,
    create_conversation,
    PROMPT_TEMPLATE_2,
    PROMPT_TEMPLATE_3,
    PROMPT_TEMPLATE_audio,
    PROMPT_TEMPLATE_image,
    AgentType,
    SafetyCategory,
)
from PIL import Image
from typing import Union, Optional


def make_conversation(
    text: str,
    role: str = "user",
    media: Optional[Union[str, Image.Image]] = None,
    media_type: Optional[str] = None,
) -> list[dict]:
    """Create a single-message conversation optionally containing media."""
    content = [{"type": "text", "text": text}]
    if media is not None:
        if media_type == 'omni':
            assert isinstance(media, dict), "for omni setting, media should be dict"
            if 'video' in media:
                content.append({"type": "video", "video": media['video']})
            if 'image' in media:
                content.append({"type": "image", "image": media['image']})
            if 'audio' in media:
                content.append({"type": "audio", "audio": media['audio']})
        elif media_type == "video":
            content.append({"type": "video", "video": media})
        elif media_type == "image":
            content.append({"type": "image", "image": media})
        else:
            content.append({"type": "audio", "audio": media})
    return [{"role": role, "content": content}]


class OmniGuardEvaluator(SafetyEvaluator):
    def __init__(self, model_id, device: Optional[str] = None):
        self.model_id = model_id
        self.tokenizer = Qwen2_5OmniProcessor.from_pretrained(model_id)
        config = AutoConfig.from_pretrained(self.model_id)
        config.thinker_config.text_config.use_cache = True
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                model_id, 
                torch_dtype="auto", 
                device_map=self.device, 
                attn_implementation="flash_attention_2", 
                config=config
        ).eval()
        print(self.model.generation_config)
        print(f"Model parameter dtype: {self.model.dtype}")

    def evaluate(
        self,
        prompt_1: str,
        prompt_2: str,
        *,
        categories: Optional[List[Any]] = None,
        **kwargs,
    ) -> str:
        # Get media type to select appropriate prompt template
        media_type = kwargs.get("media_type")
        max_token = kwargs.get("max_token", 1024)
        # For images (VLGuard), use image-specific template without conversation
        if media_type == "image":
            # Build prompt for image evaluation
            policies = ""
            for i, category in enumerate(categories, 1):
                category_name = f"{LLAMA_GUARD_3_CATEGORY_SHORT_NAME_PREFIX}{i}"
                policies += f"{category_name}: {category.name}\n{category.description}\n\n"
            
            formatted_prompt = PROMPT_TEMPLATE_image.format(policies=policies.strip())
        else:
            # For text/audio, use standard conversation-based template
            prompt = [([prompt_1]),([prompt_2])]
            formatted_prompt = build_custom_prompt(
                agent_type=AgentType.AGENT,
                conversations=create_conversation(prompt),
                categories=categories,
                category_short_name_prefix=LLAMA_GUARD_3_CATEGORY_SHORT_NAME_PREFIX,
                prompt_template=PROMPT_TEMPLATE_audio,
                with_policy=True,
            )

        # print(formatted_prompt)

        conversation = make_conversation(
            text=formatted_prompt,
            media=kwargs.get("media"),
            media_type=media_type,
        )
        text = self.tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
        inputs = self.tokenizer(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=False)
        inputs = inputs.to(self.model.device).to(self.model.dtype)
        prompt_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            text_ids = self.model.generate(**inputs, max_new_tokens=max_token, return_audio=False)
            result = self.tokenizer.decode(text_ids[0][prompt_len:], skip_special_tokens=True).strip() 
        return result