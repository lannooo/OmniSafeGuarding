from module.mllm import BaseLLM, MiniCPM_o_4_5
from module.prompt import *
from module.util import not_empty, parse_output

def inference_omniguard_sft(llm: BaseLLM, dataset, version:str, report_detail=False, batch_size=4, use_sys_prompt=True):

    results = []
    prompt_token_cfgs = {
        "v2": (EVALUATOR_PROMPT_QUERY_ONLY_v2 + "\n", 32),
        "v3-1": (EVALUATOR_PROMPT_QUERY_ONLY_v3_1, 64),
        "v3-1-zero": (EVALUATOR_PROMPT_QUERY_ONLY_v3_1_Zeroshot, 64),
        "v3-1-balance": (EVALUATOR_PROMPT_QUERY_ONLY_v3_1_balance, 64),
        "v3-2": (EVALUATOR_PROMPT_QUERY_ONLY_v3_2, 512),
        "v3-2r": (EVALUATOR_PROMPT_QUERY_ONLY_v3_2r, 512),
        "v3-3": (EVALUATOR_PROMPT_QUERY_ONLY_v3_3, 512),
        "v3-4": (EVALUATOR_PROMPT_QUERY_ONLY_v3_4, 1024),
        "v3-5": (EVALUATOR_PROMPT_QUERY_ONLY_v3_5, 1024)
    }
    task_prompt, max_token = prompt_token_cfgs.get(version, (None, 1024))
    sys_prompt, user_inst = None, None # by default
    if use_sys_prompt: sys_prompt = task_prompt
    else: user_inst = task_prompt

    formatting = 'simple' if version in ['v1', 'v2'] else 'toml'
    batch_messages = []
    batch_index = []
    for i, item in enumerate(dataset):
        query = item.get('txt', None)
        if query is None: query = ''
        if version == 'v1':
            txt_prompt = EVALUATOR_PROMPT_QUERY_ONLY_v1.format(input=query)
        elif version == 'v2':
            txt_prompt = "Bob: " + query
        elif version.startswith("v3-"):
            txt_prompt = "User' message: " + query

        message = llm.build_messages(txt=txt_prompt, 
                                     img=item.get('img', None),
                                     audio=item.get('audio', None),
                                     video=item.get('video', None),
                                     user_inst=user_inst, sys_prompt=sys_prompt)
        # print(message)
        batch_messages.append(message)
        batch_index.append(i)

        if len(batch_messages) == batch_size or i == len(dataset) - 1:
            try:
                outputs = llm.generate(batch_messages, max_token=max_token)
            except AssertionError as e:
                if isinstance(llm, MiniCPM_o_4_5):
                    # might be error caused by too long audio file, just skip
                    print("[Error] too long audios for minicpm model, skip parsing")
                    outputs = []
                else:
                    raise e

            for j, output in enumerate(outputs):
                idx = batch_index[j]
                # print(output)
                try:
                    info = parse_output(output, formatting=formatting)
                except Exception as e:
                    print("[Error]", e)
                    print(f"[Error] Failed to parse output: {output} for message: {batch_messages[j]}")
                    info = None
                if info is not None:
                    if formatting == 'simple':
                        brief_output = info['pred_toxicity'] + " " + info['pred_risk'] + " " + info['pred_extra']
                    else:
                        brief_output = info['pred_toxicity'] + " " + info['pred_risk'] + " " + info['pred_extra']['reasoning']
                    print(f"{idx+1}/{len(dataset)}:", f"[{brief_output}]", query[:128])

                    info['gt_toxicity'] = item['toxicity']
                    info['gt_risk'] = item['risk']
                    
                    if report_detail:
                        if not_empty(item, 'txt'): info['gt_query'] = item['txt']
                        if not_empty(item, 'img'): info['gt_image'] = item['img']
                        if not_empty(item, 'audio'): info['gt_audio'] = item['audio']
                        if not_empty(item, 'video'): info['gt_video'] = item['video']
                    results.append(info)
            batch_messages.clear()
            batch_index.clear()
    return results
