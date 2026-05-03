import argparse
import os
import dashscope
import json
import re

from transformers import pipeline
from openai import OpenAI
from module.load_datasets import *
from module.prompt import (
    POLICY_V1, 
    POLICY_V1_BRIEF, 
    RELABEL_GUIDE_PROMPT_v2, 
    RELABEL_TASK_PROMPT_QUERY_ONLY
)
from module.policy import PROMPT_UNSAFE

# You are a powerful safety guardrail for large language models, designed to help filter malicious and harmful user queries. Your task is to examine user inputs to determine whether they violate the given safety policies, and if so, identify which specific category has been violated.
# Note that the criterion for judging harmful input is whether the input might cause the model to output guidance and descriptions for implementing behaviors in that risk category, or to generate content on topics within that category.

def call_qwen_max(messages)->str:
    response = dashscope.Generation.call(
        api_key='your-api-key',
        model="qwen3-max",
        messages=messages,
        result_format='message',
        headers={'X-DashScope-DataInspection': '{"input": "disable", "output": "disable"}'}
    )
    # check if there is valid response content (choices)
    if not hasattr(response, 'output') \
        or not hasattr(response.output, 'choices') \
            or len(response.output.choices) == 0:
        print(response)
        raise Exception("No valid response content")
    return response.output.choices[0].message.content



def call_deepseek(messages)->str:
    response = deepseek_client.chat.completions.create(
        model="deepseek-chat",  # deepseek-chat, deepseek-reasoner
        messages=messages,
        stream=False
    )
    return response.choices[0].message.content


def call_seed(messages)->str:
    completion = seed_client.chat.completions.create(
        model="doubao-seed-1-6-251015",
        messages=messages,
        reasoning_effort="minimal",  # minimal, low, medium, high
        thinking={
            "type": "enabled"
        },
    )
    return completion.choices[0].message.content


def call_mistral(messages)->str:
    response = mistral_client.chat.complete(
        model="mistral-medium-latest",
        messages=messages,
        stream=False
    )
    return response.choices[0].message.content



def llm_judge(call_fn, input, response=None, policy=POLICY_V1):
    prompt = RELABEL_TASK_PROMPT_QUERY_ONLY.format(guide_prompt=RELABEL_GUIDE_PROMPT_v2, policy=policy, input=input)

    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': prompt}
    ]

    # print(response)
    assessment:str = call_fn(messages)

    print(assessment, "\n")
    res = {'prompt': input}
    # parse content: safe or unsafe
    safe_pattern = r"(safe|unsafe)"
    category_pattern = r"<CATEGORY>(.*)</CATEGORY>"
    rationale_pattern = r"<RATIONALE>(.*)</RATIONALE>"
    parts = [p for p in assessment.splitlines() if p != ""]
    assert len(parts) <= 3, f"Invalid assessment: {parts}"
    match = re.search(safe_pattern, parts[0])
    if match:
        res['safety'] = match.group(1)
    
        if res['safety'] == 'unsafe':
            match = re.search(category_pattern, parts[1])
            if match:
                res['category'] = match.group(1).strip()
            match = re.search(rationale_pattern, parts[2])
            if match:
                res['rationale'] = match.group(1).strip()

    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm', type=str, default='deepseek')
    parser.add_argument('--policy', type=str, default='v1')
    args = parser.parse_args()

    if args.llm == 'deepseek':
        deepseek_client = OpenAI(
            api_key='sk-e565ae25b85b4e21a18bcff09e6bf37b',
            base_url="https://api.deepseek.com")
        call_fn = call_deepseek 
    elif args.llm == 'seed':
        from volcenginesdkarkruntime import Ark
        seed_client = Ark(
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key='bf543c79-d27e-4081-b8d5-745461250603',
        )
        call_fn = call_seed 
    elif args.llm == 'qwen':
        call_fn = call_qwen_max
    elif args.llm == 'mistral':
        from mistralai import Mistral
        mistral_client = Mistral(
            api_key='wEHYGjqOA67eXAUdbFVvLQOd0XjPc39l',
        )
        call_fn = call_mistral
    else:
        raise ValueError("Invalid llm")
    
    if args.policy == 'v1':
        policy = POLICY_V1
    elif args.policy == 'v1_brief':
        policy = POLICY_V1_BRIEF
    else:
        raise ValueError("Invalid policy")
    
    # Example to perform relabeling on custom dataset
    dataset = hf_load_omniguard_custom()
    filter_risks = None
    filter_toxic = PROMPT_UNSAFE
    output_file = f'omniguard_custom_{args.llm}_{args.policy}.jsonl'

    is_resume = False
    read_mode = 'w'
    if os.path.exists(output_file):
        read_mode = 'a'
        is_resume = True
        # read the last line of jsonl file
        with open(output_file, "r") as f:
            last_line = f.readlines()[-1]
            last_item = json.loads(last_line)
        resume_prompt = last_item['prompt']
    
    # remove redundant items in dataset with same request_key
    request_keys = set()
    unique_dataset = []
    for i, item in enumerate(dataset):
        if item['request_key'] in request_keys: continue
        request_keys.add(item['request_key'])
        unique_dataset.append(item)

    with open(output_file, read_mode) as f:
        for i, item in enumerate(unique_dataset):
            if filter_risks is not None and item['risk'] not in filter_risks: continue
            if filter_toxic is not None and item['toxicity'] != filter_toxic: continue
            if is_resume:
                if item['request_key'] == resume_prompt: 
                    print(f"Resuming after prompt: {item['request_key']}")
                    is_resume = False
                continue
            
            print(f"Processing {i}/{len(dataset)}: [{item['toxicity']}-{item['risk']}] {item['request_key']}")

            res = llm_judge(call_fn=call_fn, input=item['request_key'], policy=policy)
            if 'safety' in res:
                res['original_risk'] = item['risk']
                f.write(json.dumps(res) + "\n")
    