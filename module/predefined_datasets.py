import sys

from module.configs import DatasetConfigs, Splitable
from module.predefined_modality import (
    TextModality, VisionModality, AudioModality, OmniModality, CustomOmniModality
)


harmful_text_collections_2000 = [
    Splitable(TextModality.forbidden_question_base, train_split=200),       # question style
    Splitable(TextModality.forbidden_question_dan, train_split=400),        # DAN-prompt+question
    Splitable(TextModality.beavertails_30k_test_unsafe, train_split=400),   # realistic queries
    Splitable(TextModality.jbv_redteam_2k, train_split=200),                # question style
    Splitable(TextModality.mm_safetybench_text, train_split=200),           # instruction style
    Splitable(TextModality.aiah_alignment_text, train_split=200),           # question style
    Splitable(TextModality.ajailbench_origin_text, train_split=200),        # jailbreak prompt
    Splitable(TextModality.omni_safetybench_unimodal_t, train_split=200),   # instruction style
]
harmful_text_instructions_collections_1500 = [
    Splitable(TextModality.forbidden_question_base, train_split=200),       # question style
    Splitable(TextModality.beavertails_30k_test_unsafe, train_split=400),   # realistic queries
    Splitable(TextModality.jbv_redteam_2k, train_split=200),                # question style
    Splitable(TextModality.aiah_alignment_text, train_split=200),           # question style
    Splitable(TextModality.mm_safetybench_text, train_split=300),           # instruction style
    Splitable(TextModality.omni_safetybench_unimodal_t, train_split=200),   # instruction style
]
harmful_text_questions_collections_1000 = [
    Splitable(TextModality.forbidden_question_base, train_split=200),       # question style
    Splitable(TextModality.beavertails_30k_test_unsafe, train_split=400),   # realistic queries
    Splitable(TextModality.jbv_redteam_2k, train_split=200),                # question style
    Splitable(TextModality.aiah_alignment_text, train_split=200),           # question style
]
benign_text_collections_2000 = [
    Splitable(TextModality.mtbench, train_split=80),  # realistic queries
    Splitable(TextModality.voicebench_alpacaeval_text, train_split=636), # realistic queries
    Splitable(TextModality.voicebench_ifeval_text, train_split=345), # realistic queries
    Splitable(TextModality.voicebench_openbookqa_text, train_split=455), # multi-choice questions
    Splitable(TextModality.voicebench_wildvoice_text, train_split=1000), # questions style
]
benign_text_instructions_collections_1500 = [
    Splitable(TextModality.mtbench, train_split=80),  # realistic queries
    Splitable(TextModality.voicebench_alpacaeval_text, train_split=600), # realistic queries
    Splitable(TextModality.voicebench_ifeval_text, train_split=345), # realistic queries
    Splitable(TextModality.voicebench_wildvoice_text, train_split=500), # questions style
]
benign_text_questions_collections_1000 = [
    Splitable(TextModality.mtbench, train_split=80),  # realistic queries
    Splitable(TextModality.voicebench_alpacaeval_text, train_split=300), # realistic queries
    Splitable(TextModality.voicebench_ifeval_text, train_split=300), # realistic queries
    Splitable(TextModality.voicebench_wildvoice_text, train_split=400), # questions style
]


benign_vision_collections_1500 = [
    Splitable(VisionModality.okvqa, train_split=900),
    Splitable(VisionModality.mm_vet_v2, train_split=300),
    Splitable(VisionModality.mme, train_split=300),
    Splitable(VisionModality.llava_wild, train_split=60),
]
benign_vision_collections_500 = [
    Splitable(VisionModality.mm_vet_v2, train_split=500),
]
harmful_vision_collections_500 = [
    Splitable(VisionModality.jbv_jailbreak_28k, train_split=300),
    Splitable(VisionModality.mm_safetybench, train_split=200),
]
harmful_vision_collections_750 = [
    Splitable(VisionModality.jbv_jailbreak_28k, train_split=500),
    Splitable(VisionModality.mm_safetybench, train_split=250),
]
harmful_vision_collections_1000 = [
    Splitable(VisionModality.jbv_jailbreak_28k, train_split=600),
    Splitable(VisionModality.mm_safetybench, train_split=400),
]
harmful_vision_questions_collections_1000 = [
    Splitable(VisionModality.vlsafe, train_split=1100),           # VQ question style
]
harmful_vision_collections_1500 = [
    Splitable(VisionModality.jbv_jailbreak_28k, train_split=300), # mixed (query-related, text-transfered, figstep) jailbreaks
    Splitable(VisionModality.mm_safetybench, train_split=200),    # instruction style (query-related jailbreak)
    Splitable(VisionModality.vlsafe, train_split=1000),           # VQ question style
]

harmful_audio_collections_1000 = [
    Splitable(AudioModality.ajailbench_origin_prompt, train_split=500),
    Splitable(AudioModality.ajailbench_origin_prompt_rewrite, train_split=500),
]
harmful_audio_collections_850 = [
    Splitable(AudioModality.ajailbench_origin_prompt_rewrite, train_split=500),
    Splitable(AudioModality.aiah_alignment_prompt_rewrite, train_split=350)
]
benign_audio_collections_1000 = [
    Splitable(AudioModality.voicebench_alpacaeval, train_split=300),
    Splitable(AudioModality.voicebench_commoneval, train_split=300),
    Splitable(AudioModality.voicebench_wildvoice, train_split=400)
]

# ----------------Probing-----------------
probe_text_v2 = DatasetConfigs([
    Splitable(CustomOmniModality.omni_custom_T, train_split=100),
])
probe_text_align_vision_v2 = DatasetConfigs([
    Splitable(CustomOmniModality.omni_custom_T, train_split=100, include_in_test=False),
    # Splitable(VisionModality.omni_custom_I, train_split=0),
    Splitable(CustomOmniModality.omni_custom_TI, train_split=0),
])
probe_text_align_audio_v2 = DatasetConfigs([
    Splitable(CustomOmniModality.omni_custom_T, train_split=100, include_in_test=False),
    # Splitable(AudioModality.omni_custom_A, train_split=0),
    Splitable(CustomOmniModality.omni_custom_TA, train_split=0),
])
probe_text_align_video_v2 = DatasetConfigs([
    Splitable(CustomOmniModality.omni_custom_T, train_split=100, include_in_test=False),
    # Splitable(VisionModality.omni_custom_V, train_split=0),
    Splitable(CustomOmniModality.omni_custom_TV, train_split=0),
])
probe_text_align_vision_audio_v2 = DatasetConfigs([
    Splitable(CustomOmniModality.omni_custom_T, train_split=100, include_in_test=False),
    Splitable(CustomOmniModality.omni_custom_TIA, train_split=0),
])
probe_text_align_video_audio_v2 = DatasetConfigs([
    Splitable(CustomOmniModality.omni_custom_T, train_split=100, include_in_test=False),
    Splitable(CustomOmniModality.omni_custom_TVA, train_split=0),
])
# -------------------------------------
probe_vision_v2 = DatasetConfigs([
    Splitable(CustomOmniModality.omni_custom_I, train_split=100),
])
probe_vision_align_text_v2 = DatasetConfigs([
    Splitable(CustomOmniModality.omni_custom_I, train_split=100, include_in_test=False),
    Splitable(CustomOmniModality.omni_custom_T, train_split=0),
])
probe_vision_align_audio_v2 = DatasetConfigs([
    Splitable(CustomOmniModality.omni_custom_I, train_split=100, include_in_test=False),
    Splitable(CustomOmniModality.omni_custom_A, train_split=0),
])
probe_vision_align_video_v2 = DatasetConfigs([
    Splitable(CustomOmniModality.omni_custom_I, train_split=100, include_in_test=False),
    Splitable(CustomOmniModality.omni_custom_V, train_split=0),
])
# -------------------------------------
probe_audio_v2 = DatasetConfigs([
    Splitable(CustomOmniModality.omni_custom_A, train_split=100),
])
# -------------------------------------
probe_video_v2 = DatasetConfigs([
    Splitable(CustomOmniModality.omni_custom_V, train_split=100),
])
# -------------------------------------
probe_omni_custom_v1 = DatasetConfigs([
    Splitable(CustomOmniModality.omni_custom_T, train_split=100),
    
    Splitable(CustomOmniModality.omni_custom_I, train_split=0),
    Splitable(CustomOmniModality.omni_custom_TI, train_split=0),
    Splitable(CustomOmniModality.omni_custom_ST_I, train_split=0),
    Splitable(CustomOmniModality.omni_custom_T_SI, train_split=0),
    
    Splitable(CustomOmniModality.omni_custom_V, train_split=0),
    Splitable(CustomOmniModality.omni_custom_TV, train_split=0),
    Splitable(CustomOmniModality.omni_custom_ST_V, train_split=0),
    Splitable(CustomOmniModality.omni_custom_T_SV, train_split=0),
    
    Splitable(CustomOmniModality.omni_custom_A, train_split=0),
    Splitable(CustomOmniModality.omni_custom_TA, train_split=0),
    Splitable(CustomOmniModality.omni_custom_ST_A, train_split=0),
    Splitable(CustomOmniModality.omni_custom_T_SA, train_split=0),

    Splitable(CustomOmniModality.omni_custom_IA, train_split=0),
    Splitable(CustomOmniModality.omni_custom_SI_A, train_split=0),
    Splitable(CustomOmniModality.omni_custom_I_SA, train_split=0),
    
    Splitable(CustomOmniModality.omni_custom_VA, train_split=0),
    Splitable(CustomOmniModality.omni_custom_SV_A, train_split=0),
    Splitable(CustomOmniModality.omni_custom_V_SA, train_split=0),

    Splitable(CustomOmniModality.omni_custom_TIA, train_split=0),
    Splitable(CustomOmniModality.omni_custom_ST_I_A, train_split=0),
    Splitable(CustomOmniModality.omni_custom_ST_I_SA, train_split=0),
    
    Splitable(CustomOmniModality.omni_custom_TVA, train_split=0),
    Splitable(CustomOmniModality.omni_custom_ST_V_A, train_split=0),
    Splitable(CustomOmniModality.omni_custom_ST_V_SA, train_split=0),
])
# -------------------------------------

# ----------------Training----------------
train_text_v1 = DatasetConfigs([
    *harmful_text_questions_collections_1000,
    *benign_text_questions_collections_1000,
])
train_text_v2 = DatasetConfigs([
    *harmful_text_instructions_collections_1500,
    *benign_text_instructions_collections_1500,
])
train_text_v3 = DatasetConfigs([
    *harmful_text_collections_2000,
    *benign_text_collections_2000,
])
train_vision_v1 = DatasetConfigs([
    *harmful_vision_questions_collections_1000,
    *benign_vision_collections_1500,
])
train_vision_v3 = DatasetConfigs([
    *harmful_vision_collections_1500,
    *benign_vision_collections_1500,
])

train_audio_v1 = DatasetConfigs([
    *harmful_audio_collections_850,
    *benign_audio_collections_1000,
])

train_multimodal_v3 = DatasetConfigs([
    *benign_text_collections_2000,
    *harmful_text_collections_2000,
    
    *benign_vision_collections_1500,
    *harmful_vision_collections_1000,
    
    *benign_audio_collections_1000,
    *harmful_audio_collections_1000,
])

# ----------------Evaluation By Functions ----------------
eval_fast_validate = DatasetConfigs([
    Splitable(TextModality.mtbench),
    Splitable(TextModality.simplesafety),
    Splitable(VisionModality.mm_vet_v2),
    Splitable(VisionModality.figstep),
    Splitable(AudioModality.aiah_alignment),
])

MASK_CONFIG_TEXT = {'replace_word': 'mask', 'mask_txt': True}
MASK_CONFIG_IMAGE = {'replace_word': 'mask', 'mask_img': True}
MASK_CONFIG_AUDIO = {'replace_word': 'mask', 'mask_audio': True}
MASK_CONFIG_VIDEO = {'replace_word': 'mask', 'mask_video': True}
MASK_CONFIG_ALL = {'replace_word': 'mask', 'mask_txt': True, 'mask_img': True, 'mask_audio': True, 'mask_video': True}

eval_mask_all_exp = DatasetConfigs([
    Splitable(TextModality.toxicchat_test, mask_config=MASK_CONFIG_ALL),
    Splitable(VisionModality.vlsbench, mask_config=MASK_CONFIG_ALL),
    Splitable(VisionModality.mm_safetybench, mask_config=MASK_CONFIG_ALL),
    Splitable(AudioModality.aiah, mask_config=MASK_CONFIG_ALL),
    Splitable(AudioModality.ajailbench, mask_config=MASK_CONFIG_ALL),
    Splitable(VisionModality.video_safetybench_ben, mask_config=MASK_CONFIG_ALL),
    Splitable(VisionModality.safewatch_real, mask_config=MASK_CONFIG_ALL),

    # Splitable(VisionModality.vlguard, mask_config=MASK_CONFIG_ALL),
    # Splitable(VisionModality.siuo, mask_config=MASK_CONFIG_ALL),
    # Splitable(VisionModality.rtvlm, mask_config=MASK_CONFIG_ALL),
    # Splitable(VisionModality.jbv_jailbreak_mini, mask_config=MASK_CONFIG_ALL),
    # Splitable(VisionModality.figstep, mask_config=MASK_CONFIG_ALL),
    # Splitable(VisionModality.mml_hades, mask_config=MASK_CONFIG_ALL),
])

eval_general = DatasetConfigs([
    Splitable(TextModality.truthfulQA),             # Text
    Splitable(VisionModality.mme),                  # Image
    Splitable(AudioModality.voicebench_alpacaeval), # Audio
    Splitable(VisionModality.mmbench_video),        # Video
])

eval_basic_safety = DatasetConfigs([
    # Splitable(TextModality.jbv_redteam_2k),
    Splitable(TextModality.beavertails_30k_test),
    Splitable(TextModality.aegis2_test),
    Splitable(TextModality.openai_moderation),
    Splitable(TextModality.wildguardtest),
    Splitable(TextModality.toxicchat_test),

    Splitable(VisionModality.rtvlm),
    Splitable(VisionModality.vlsbench),
    Splitable(VisionModality.vlguard),
    Splitable(VisionModality.siuo),

    Splitable(AudioModality.safebench_ta),
    Splitable(AudioModality.aiah),

    Splitable(VisionModality.safewatch_real)
])

eval_ext_jailbreaks = DatasetConfigs([
    Splitable(TextModality.forbidden_question_dan),
    Splitable(TextModality.harmbench_contextual),
    Splitable(TextModality.jailbreakbench),
    Splitable(TextModality.cipherchat),

    Splitable(VisionModality.mm_safetybench),
    # Splitable(VisionModality.jbv_jailbreak_mini),
    Splitable(VisionModality.figstep),
    # Splitable(VisionModality.mml_hades),

    Splitable(AudioModality.omni_safetybench_dual_ta),
    # Splitable(AudioModality.aiah_spelling),
    Splitable(AudioModality.ajailbench),

    Splitable(VisionModality.omni_safetybench_dual_tv),
    Splitable(VisionModality.video_safetybench_ben)
])

eval_omni_exp = DatasetConfigs([
    Splitable(TextModality.omni_safetybench_unimodal_t),
    Splitable(AudioModality.omni_safetybench_unimodal_a),
    Splitable(VisionModality.omni_safetybench_unimodal_i),
    Splitable(VisionModality.omni_safetybench_unimodal_v),
    Splitable(TextModality.safebench_t),

    Splitable(AudioModality.omni_safetybench_dual_ta),
    Splitable(VisionModality.omni_safetybench_dual_ti),
    Splitable(VisionModality.omni_safetybench_dual_tv),
    Splitable(VisionModality.safebench_ti),
    Splitable(AudioModality.safebench_ta),

    Splitable(OmniModality.omni_safetybench_omni_tia),
    Splitable(OmniModality.omni_safetybench_omni_tva),
    Splitable(OmniModality.safebench_tia),
])

eval_false_reject = DatasetConfigs([
    Splitable(TextModality.xstest),
    Splitable(VisionModality.false_reject_mme),
    Splitable(AudioModality.false_reject_alpacaeval),
    Splitable(VisionModality.false_reject_mmbench_video)
])

eval_custom_modality = DatasetConfigs([
    Splitable(CustomOmniModality.omni_custom_T),
    Splitable(CustomOmniModality.omni_custom_T_SI),
    Splitable(CustomOmniModality.omni_custom_T_SA),
    Splitable(CustomOmniModality.omni_custom_T_SV),
    Splitable(CustomOmniModality.omni_custom_T_SI_SA),
    Splitable(CustomOmniModality.omni_custom_T_SV_SA),
    Splitable(CustomOmniModality.omni_custom_T_SI_SV),
    Splitable(CustomOmniModality.omni_custom_T_SI_SA_SV),

    Splitable(CustomOmniModality.omni_custom_I),
    Splitable(CustomOmniModality.omni_custom_ST_I),
    Splitable(CustomOmniModality.omni_custom_I_SV),
    Splitable(CustomOmniModality.omni_custom_I_SA),
    Splitable(CustomOmniModality.omni_custom_ST_I_SA),
    Splitable(CustomOmniModality.omni_custom_ST_I_SV),
    Splitable(CustomOmniModality.omni_custom_I_SA_SV),
    Splitable(CustomOmniModality.omni_custom_ST_I_SA_SV),

    Splitable(CustomOmniModality.omni_custom_V),
    Splitable(CustomOmniModality.omni_custom_ST_V),
    Splitable(CustomOmniModality.omni_custom_V_SA),
    Splitable(CustomOmniModality.omni_custom_SI_V),
    Splitable(CustomOmniModality.omni_custom_ST_V_SA),
    Splitable(CustomOmniModality.omni_custom_SI_SA_V),
    Splitable(CustomOmniModality.omni_custom_ST_SI_V),
    Splitable(CustomOmniModality.omni_custom_ST_SI_SA_V),
    
    Splitable(CustomOmniModality.omni_custom_A),
    Splitable(CustomOmniModality.omni_custom_ST_A),
    Splitable(CustomOmniModality.omni_custom_SI_A),
    Splitable(CustomOmniModality.omni_custom_SV_A),
    Splitable(CustomOmniModality.omni_custom_ST_SI_A),
    Splitable(CustomOmniModality.omni_custom_ST_SV_A),
    Splitable(CustomOmniModality.omni_custom_SI_A_SV),
    Splitable(CustomOmniModality.omni_custom_ST_SI_A_SV),

    # Splitable(CustomOmniModality.omni_custom_TI),
    # Splitable(CustomOmniModality.omni_custom_TV),
    # Splitable(CustomOmniModality.omni_custom_TA),
    # Splitable(CustomOmniModality.omni_custom_IA),
    # Splitable(CustomOmniModality.omni_custom_VA),

    # Splitable(CustomOmniModality.omni_custom_TIA),
    # Splitable(CustomOmniModality.omni_custom_ST_I_A),
    # Splitable(CustomOmniModality.omni_custom_T_SI_A),
    # Splitable(CustomOmniModality.omni_custom_T_I_SA),

    # Splitable(CustomOmniModality.omni_custom_TVA),
    # Splitable(CustomOmniModality.omni_custom_ST_V_A),
    # Splitable(CustomOmniModality.omni_custom_T_SV_A),
    # Splitable(CustomOmniModality.omni_custom_T_V_SA),

    # Splitable(CustomOmniModality.omni_custom_TIV),
    # Splitable(CustomOmniModality.omni_custom_ST_I_V),
    # Splitable(CustomOmniModality.omni_custom_T_SI_V),
    # Splitable(CustomOmniModality.omni_custom_T_I_SV),

    # Splitable(CustomOmniModality.omni_custom_IAV),
    # Splitable(CustomOmniModality.omni_custom_SI_A_V),
    # Splitable(CustomOmniModality.omni_custom_I_SA_V),
    # Splitable(CustomOmniModality.omni_custom_I_A_SV),

    # Splitable(CustomOmniModality.omni_custom_all),
    # Splitable(CustomOmniModality.omni_custom_ST_I_A_V),
    # Splitable(CustomOmniModality.omni_custom_T_SI_A_V),
    # Splitable(CustomOmniModality.omni_custom_T_I_SA_V),
    # Splitable(CustomOmniModality.omni_custom_T_I_A_SV),
])

# ----------------Evaluation By Modality----------------
eval_text_v1 = DatasetConfigs([
    Splitable(TextModality.truthfulQA),
    Splitable(TextModality.voicebench_bbh_text),
    Splitable(TextModality.advbench),
    Splitable(TextModality.jbv_redteam_2k),
    Splitable(TextModality.omni_safetybench_unimodal_t),
    Splitable(TextModality.safebench_t),

    Splitable(TextModality.beavertails_30k_test),
    Splitable(TextModality.forbidden_question_dan),
    Splitable(TextModality.simplesafety),
    Splitable(TextModality.harmbench_standard),
    Splitable(TextModality.harmbench_contextual),
    Splitable(TextModality.xstest),
    Splitable(TextModality.saferlhf_test),
    Splitable(TextModality.openai_moderation),
    Splitable(TextModality.wildguardtest),
    Splitable(TextModality.aegis2_test),
    Splitable(TextModality.toxicchat_test),
])

eval_vision_v1 = DatasetConfigs([
    Splitable(VisionModality.mme),
    Splitable(VisionModality.llava_wild),
    Splitable(VisionModality.okvqa),

    Splitable(VisionModality.mm_safetybench),
    Splitable(VisionModality.jbv_jailbreak_28k),
    Splitable(VisionModality.omni_safetybench_dual_ti),
    Splitable(VisionModality.safebench_ti),

    Splitable(VisionModality.vlsafe),
    Splitable(VisionModality.figstep),
    Splitable(VisionModality.vlsbench),
    Splitable(VisionModality.rtvlm),
    Splitable(VisionModality.hades),
    Splitable(VisionModality.mml_hades_wr),
    Splitable(VisionModality.mml_hades_rotate),
    Splitable(VisionModality.mml_hades_mirror),
    Splitable(VisionModality.mml_hades_base64),
    Splitable(VisionModality.vlguard),
    Splitable(VisionModality.siuo),
])

eval_audio_v1 = DatasetConfigs([
    Splitable(AudioModality.voicebench_alpacaeval),
    Splitable(AudioModality.voicebench_advbench),
    Splitable(AudioModality.omni_safetybench_dual_ta),
    Splitable(AudioModality.safebench_ta),
    Splitable(AudioModality.aiah_alignment),
    Splitable(AudioModality.aiah_nonspeech),
    Splitable(AudioModality.aiah_spelling),
    Splitable(AudioModality.ajailbench_origin),
    Splitable(AudioModality.ajailbench_apt),
])


eval_video_v1 = DatasetConfigs([
    Splitable(VisionModality.video_safetybench),
    Splitable(VisionModality.video_safetybench_ben),
    Splitable(VisionModality.omni_safetybench_dual_tv),
    Splitable(VisionModality.safewatch),
])


eval_omni_v1 = DatasetConfigs([
    Splitable(OmniModality.omni_safetybench_omni_tia),
    Splitable(OmniModality.omni_safetybench_omni_tva),
    Splitable(OmniModality.safebench_tia),
])


def load_dataset_configs(stage:str, collection:str='text') -> DatasetConfigs:
    assert stage in ['train', 'eval', 'probe'], f"Unknown stage: {stage}, must be train or eval"
    # base_name = os.path.basename(model_path).lower()
    def get_field(field_name: str):
        current_module = sys.modules[__name__]
        return getattr(current_module, field_name, None)
    
    if stage == 'eval': 
        if collection == 'all': return eval_text_v1 + eval_vision_v1 \
            + eval_audio_v1 + eval_omni_v1 + eval_video_v1
        elif collection == 'text': return eval_text_v1
        elif collection == 'vision': return eval_vision_v1
        elif collection == 'audio': return eval_audio_v1
        elif collection == 'video': return eval_video_v1
        elif collection == 'omni': return eval_omni_v1
        else:
            # none of above is matched, than it is a specified id
            collection_id = f"{stage}_{collection}"
            # get attribute by name from this file
            obj = get_field(collection_id)
            if obj is None:
                raise ValueError(f"Unknown predefined datasets: {stage}.{collection}")
            return obj
    else:
        # none of above is matched, than it is a specified id
        collection_id = f"{stage}_{collection}"
        # get attribute by name from this file
        obj = get_field(collection_id)
        if obj is None:
            raise ValueError(f"Unknown predefined datasets: {stage}.{collection}")
        return obj