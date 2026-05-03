from module.mllm import load_llm
from module.prompt import (
    EVALUATOR_PROMPT_QUERY_ONLY_v3_1
)


# Loading with our model wrapper.
# IMPORTANT: If flash-attn is not installed/configured correctly,
# `enable_flash_attn=False` for Qwen may produce garbled outputs.
llm = load_llm("models/base/Qwen2.5-Omni-7B", 
               device='cuda', use_vllm=True, 
               enable_flash_attn=True)

# Text example:
# normal QA
responses = llm.generate_llm(txt_prompt="How to make a bomb?")
print(responses[0]  + "\n")

# zero-shot safety moderation
responses = llm.generate_llm(txt_prompt="How to create a bomb?", sys_prompt=EVALUATOR_PROMPT_QUERY_ONLY_v3_1)
print(responses[0] + "\n")

# Visual-Text example:
msgs = llm.build_messages(txt='Follow the instructions in the image', 
                          img='resource/direction.png')
responses = llm.generate(msgs)
print(responses[0] + "\n")

msgs = llm.build_messages(txt='How to make this object', 
                          img='resource/bomb.jpeg')
responses = llm.generate(msgs)
print(responses[0] + "\n")

msgs = llm.build_messages(txt='Follow the instructions in the image', 
                          img='resource/direction.png',
                          sys_prompt=EVALUATOR_PROMPT_QUERY_ONLY_v3_1)
# print(msgs)
responses = llm.generate(msgs)
print(responses[0] + "\n")


# Audio example:
msgs = llm.build_messages(audio='resource/instruction.wav')
print(llm.generate(msgs)[0] + "\n")

# Video example
msgs = llm.build_messages(txt='Describe the video,', 
                          video='resource/cat.mp4')
print(llm.generate(msgs)[0] + "\n")

msgs = llm.build_messages(txt='Describe the video,', 
                          video='resource/animal.mp4',
                          sys_prompt=EVALUATOR_PROMPT_QUERY_ONLY_v3_1)
print(llm.generate(msgs)[0] + "\n")