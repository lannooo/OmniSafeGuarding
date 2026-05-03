import os

from module.policy import PolicyHub
from module.configs import ConfigItem, relabel_from
from module.augmentation import *
import module.load_datasets as LoadFn


LABEL_DIR = os.path.join(os.getenv("EXTERNAL_DIR", "./data/external"), "labels")

class TextModality:
    mtbench = ConfigItem("MT-Bench", LoadFn.mtbench)
    truthfulQA = ConfigItem("TruthfulQA", LoadFn.truthful)
    advbench = ConfigItem("AdvBench", LoadFn.advbench, risk_mapper=relabel_from(f"{LABEL_DIR}/advbench_*.jsonl"))
    aegis2_test = ConfigItem("Aegis2-Test", LoadFn.aegis2_test, policy=PolicyHub.aegis2, risk_mapper=relabel_from(f"{LABEL_DIR}/aegis2_test_*.jsonl"))
    beavertails_30k_test = ConfigItem("BeaverTails-Test", LoadFn.beavertails_30k_test, policy=PolicyHub.beavertails, risk_mapper=relabel_from(f"{LABEL_DIR}/beavertails_30k_test_*.jsonl"))
    beavertails_30k_test_unsafe = ConfigItem("BeaverTails-Test-unsafe", LoadFn.beavertails_30k_test, policy=PolicyHub.beavertails_unsafe, risk_mapper=relabel_from(f"{LABEL_DIR}/beavertails_30k_test_*.jsonl"))
    cipherchat = ConfigItem("CipherChat", LoadFn.cipher_chat, risk_mapper=relabel_from(f"{LABEL_DIR}/advbench_*.jsonl"))
    forbidden_question_dan = ConfigItem("ForbiddenQuestions", LoadFn.forbidden_questions_dan, policy=PolicyHub.forbidden_question)
    forbidden_question_base = ConfigItem("ForbiddenQuestions_base", LoadFn.forbidden_questions_base, policy=PolicyHub.forbidden_question)
    harmbench_standard = ConfigItem("Harmbench-standard", LoadFn.harmbench_standard, policy=PolicyHub.harmbench, risk_mapper=relabel_from(f'{LABEL_DIR}/harmbench_*.jsonl'))
    harmbench_contextual = ConfigItem("Harmbench-contextual", LoadFn.harmbench_contextual, policy=PolicyHub.harmbench, risk_mapper=relabel_from(f'{LABEL_DIR}/harmbench_*.jsonl'))
    jailbreakbench = ConfigItem("JailbreakBench", LoadFn.jailbreakbench, policy=PolicyHub.jbb)
    openai_moderation = ConfigItem("OpenAI-Mod", LoadFn.openai_moderation, risk_mapper=relabel_from(f"{LABEL_DIR}/openaimod_*.jsonl"))
    saferlhf_test = ConfigItem("SafeRLHF-Test", LoadFn.saferlhf_test, policy=PolicyHub.saferlhf, risk_mapper=relabel_from(f"{LABEL_DIR}/SafeRLHF_test_*.jsonl"))
    simplesafety = ConfigItem("SimpleSafetyTests", LoadFn.simplesafetytests, policy=PolicyHub.simplesafetytests, risk_mapper=relabel_from(f'{LABEL_DIR}/simplesafety_*.jsonl'))
    toxicchat_test = ConfigItem("ToxicChat-Test", LoadFn.toxicchat_test, policy=PolicyHub.toxicchat, risk_mapper=relabel_from(f"{LABEL_DIR}/toxicchat_test_*.jsonl"))  # pyright: ignore[reportAttributeAccessIssue]
    wildguardtest = ConfigItem("Wildguard-Test", LoadFn.wildguardtest, policy=PolicyHub.wildguard, risk_mapper=relabel_from(f'{LABEL_DIR}/wildguardtest_*.jsonl'))
    xstest = ConfigItem("XSTest", LoadFn.xstest, policy=PolicyHub.xstest, risk_mapper=relabel_from(f'{LABEL_DIR}/xstest_*.jsonl'))
    
    # from VoiceBench
    voicebench_advbench_text = ConfigItem("VoiceBench-advbench-text", LoadFn.voicebench_advbench_text, risk_mapper=relabel_from(f"{LABEL_DIR}/advbench_*.jsonl"))
    voicebench_alpacaeval_text = ConfigItem("VoiceBench-alpacaeval-text", LoadFn.voicebench_alpacaeval_text)
    voicebench_bbh_text = ConfigItem("AudioBench-bbh-text", LoadFn.voicebench_bbh_text)
    voicebench_commoneval_text = ConfigItem("VoiceBench-commoneval-text", LoadFn.voicebench_commoneval_text)
    voicebench_ifeval_text = ConfigItem("VoiceBench-ifeval-text", LoadFn.voicebench_ifeval_text)
    voicebench_openbookqa_text = ConfigItem("VoiceBench-openbookqa-text", LoadFn.voicebench_openbookqa_text)
    voicebench_wildvoice_text = ConfigItem("VoiceBench-wildvoice-text", LoadFn.voicebench_wildvoice_text)
    
    # from JailbreakV-28K
    jbv_redteam_2k = ConfigItem("Redteam2K", LoadFn.jbv_redteam_2k, policy=PolicyHub.jailbreakv_28k, risk_mapper=relabel_from(f"{LABEL_DIR}/jailbreakv28k_*.jsonl"))
    # from MM-SafetyBench: Text-only
    mm_safetybench_text = ConfigItem("MMSafetybench-text", LoadFn.mm_safetybench_text, policy=PolicyHub.mm_safetybench)
    # from AIAH: plain text
    aiah_alignment_text = ConfigItem("AIAH-alignment-text", LoadFn.aiah_alignment_text, policy=PolicyHub.aiah)
    # from AudioJailbreak origin text (harmful)
    ajailbench_origin_text = ConfigItem("AJailbench-Origin-text", LoadFn.ajailbench_origin_text, policy=PolicyHub.ajailbench)
    # from safebench 
    safebench_t = ConfigItem("SafeBench-T", LoadFn.safebench_t, policy=PolicyHub.safebench, risk_mapper=relabel_from(f'{LABEL_DIR}/safebench_*.jsonl'))
    # from Omni-SafetyBench
    omni_safetybench_unimodal_t = ConfigItem("OmniSafetybench-unimodal-T", LoadFn.omni_safetybench_unimodal_t, policy=PolicyHub.omni_safetybench)
   

class VisionModality:

    false_reject_mmbench_video = ConfigItem("False-Reject-MMBench-video", LoadFn.mmbench_video, callback=lambda ds: \
                                     replace_text_prompt(ds, [phrase.value for phrase in VideoKeyphrase]))
    false_reject_mme = ConfigItem("False-Reject-MME", LoadFn.mme, callback=lambda ds: \
                                     replace_text_prompt(ds, [phrase.value for phrase in VisionKeyphrase]))

    mme = ConfigItem("MME", LoadFn.mme)
    okvqa = ConfigItem("OKVQA", LoadFn.okvqa)
    llava_wild = ConfigItem("llava-bench-wild", LoadFn.llava_bench_wild)
    mm_vet_v1 = ConfigItem("MM-Vet-v1", LoadFn.mm_vet_v1)
    mm_vet_v2 = ConfigItem("MM-Vet-v2", LoadFn.mm_vet_v2)
    vlsafe = ConfigItem("VLSafe", LoadFn.vlsafe, policy=PolicyHub.vlsafe, risk_mapper=relabel_from(f'{LABEL_DIR}/VLSafe_*.jsonl'))
    vlsbench = ConfigItem("VLSBench", LoadFn.vlsbench, policy=PolicyHub.vlsbench)
    vlguard = ConfigItem("VLGuard", LoadFn.vlguard, policy=PolicyHub.vlguard, risk_mapper=relabel_from(f'{LABEL_DIR}/vlguard_*.jsonl'))
    siuo = ConfigItem("SIUO", LoadFn.siuo, policy=PolicyHub.siuo)
    holisafe = ConfigItem("HoliSafe", LoadFn.holisafe, policy=PolicyHub.holisafe)
    holisafe_safe_image = ConfigItem("HoliSafe-safe-image", LoadFn.holisafe_safe_image, policy=PolicyHub.holisafe)
    holisafe_unsafe_image = ConfigItem("HoliSafe-unsafe-image", LoadFn.holisafe_unsafe_image, policy=PolicyHub.holisafe)
    jbv_jailbreak_28k = ConfigItem("JBV28K_JB", LoadFn.jbv_jailbreak_28k, policy=PolicyHub.jailbreakv_28k, risk_mapper=relabel_from(f"{LABEL_DIR}/jailbreakv28k_*.jsonl"))
    jbv_jailbreak_mini = ConfigItem("JBV28K_JB_mini", LoadFn.jbv_jailbreak_mini, policy=PolicyHub.jailbreakv_28k, risk_mapper=relabel_from(f"{LABEL_DIR}/jailbreakv28k_*.jsonl"))
    # jbv_jailbreak_direct_mini = ConfigItem("JBV28K_JB_direct_mini", LoadFn.jbv_jailbreak_direct_mini, policy=PolicyHub.jailbreakv_28k, risk_mapper=relabel_from("./data/JailBreakV-28k/labels/jailbreakv28k_*.jsonl"))
    figstep = ConfigItem("FigStep", LoadFn.figstep, policy=PolicyHub.figstep)
    hades = ConfigItem("Hades", LoadFn.hades, policy=PolicyHub.hades, risk_mapper=relabel_from(f'{LABEL_DIR}/hades_*.jsonl'))
    rtvlm = ConfigItem("RTVLM", LoadFn.rtvlm, policy=PolicyHub.rtvlm, risk_mapper=relabel_from(f'{LABEL_DIR}/rtvlm_*.jsonl'))
    mm_safetybench = ConfigItem("MMSafetybench", LoadFn.mm_safetybench, policy=PolicyHub.mm_safetybench)
    mm_safetybench_typo = ConfigItem("MMSafetybench-TYPO", LoadFn.mm_safetybench_TYPO, policy=PolicyHub.mm_safetybench)
    mm_safetybench_sd = ConfigItem("MMSafetybench-SD", LoadFn.mm_safetybench_SD, policy=PolicyHub.mm_safetybench)
    mm_safetybench_sdtypo = ConfigItem("MMSafetybench-SDTYPO", LoadFn.mm_safetybench_SDTYPO, policy=PolicyHub.mm_safetybench)
    mml_hades = ConfigItem("MML-Hades", [LoadFn.mml_hades_wr, LoadFn.mml_hades_mirror, LoadFn.mml_hades_rotate, LoadFn.mml_hades_base64], policy=PolicyHub.hades, risk_mapper=relabel_from(f'{LABEL_DIR}/hades_*.jsonl'))
    mml_hades_wr = ConfigItem("MML-Hades-wr", LoadFn.mml_hades_wr, policy=PolicyHub.hades, risk_mapper=relabel_from(f'{LABEL_DIR}/hades_*.jsonl'))
    mml_hades_mirror = ConfigItem("MML-Hades-mirror", LoadFn.mml_hades_mirror, policy=PolicyHub.hades, risk_mapper=relabel_from(f'{LABEL_DIR}/hades_*.jsonl'))
    mml_hades_rotate = ConfigItem("MML-Hades-rotate", LoadFn.mml_hades_rotate, policy=PolicyHub.hades, risk_mapper=relabel_from(f'{LABEL_DIR}/hades_*.jsonl'))
    mml_hades_base64 = ConfigItem("MML-Hades-base64", LoadFn.mml_hades_base64, policy=PolicyHub.hades, risk_mapper=relabel_from(f'{LABEL_DIR}/hades_*.jsonl'))
    mml_mmsafety = ConfigItem("MML-MMSafety", [LoadFn.mml_mmsafety_wr, LoadFn.mml_mmsafety_mirror, LoadFn.mml_mmsafety_rotate, LoadFn.mml_mmsafety_base64], policy=PolicyHub.mm_safetybench)
    mml_mmsafety_wr = ConfigItem("MML-MMSafety-wr", LoadFn.mml_mmsafety_wr, policy=PolicyHub.mm_safetybench)
    mml_mmsafety_mirror = ConfigItem("MML-MMSafety-mirror", LoadFn.mml_mmsafety_mirror, policy=PolicyHub.mm_safetybench)
    mml_mmsafety_rotate = ConfigItem("MML-MMSafety-rotate", LoadFn.mml_mmsafety_rotate, policy=PolicyHub.mm_safetybench)
    mml_mmsafety_base64 = ConfigItem("MML-MMSafety-base64", LoadFn.mml_mmsafety_base64, policy=PolicyHub.mm_safetybench)
    mml_figstep = ConfigItem("MML-FigStep", [LoadFn.mml_figstep_wr, LoadFn.mml_figstep_mirror, LoadFn.mml_figstep_rotate, LoadFn.mml_figstep_base64], policy=PolicyHub.figstep)
    mml_figstep_wr = ConfigItem("MML-FigStep-wr", LoadFn.mml_figstep_wr, policy=PolicyHub.figstep)
    mml_figstep_mirror = ConfigItem("MML-FigStep-mirror", LoadFn.mml_figstep_mirror, policy=PolicyHub.figstep)
    mml_figstep_rotate = ConfigItem("MML-FigStep-rotate", LoadFn.mml_figstep_rotate, policy=PolicyHub.figstep)
    mml_figstep_base64 = ConfigItem("MML-FigStep-base64", LoadFn.mml_figstep_base64, policy=PolicyHub.figstep)
    omni_safetybench_unimodal_i = ConfigItem("OmniSafetybench-unimodal-V", LoadFn.omni_safetybench_unimodal_i, policy=PolicyHub.omni_safetybench)
    omni_safetybench_unimodal_v = ConfigItem("OmniSafetybench-unimodal-Vi", LoadFn.omni_safetybench_unimodal_v, policy=PolicyHub.omni_safetybench)
    omni_safetybench_dual_ti = ConfigItem("OmniSafetybench-dual-TV", LoadFn.omni_safetybench_dual_ti, policy=PolicyHub.omni_safetybench)
    omni_safetybench_dual_tv = ConfigItem("OmniSafetybench-dual-TVi", LoadFn.omni_safetybench_dual_tv, policy=PolicyHub.omni_safetybench)
    safebench_ti = ConfigItem("SafeBench-TV", LoadFn.safebench_ti, policy=PolicyHub.safebench, risk_mapper=relabel_from(f'{LABEL_DIR}/safebench_*.jsonl'))
    mmbench_video_only = ConfigItem("MMBench-video-only", LoadFn.mmbench_video_only)
    mmbench_video = ConfigItem("MMBench-video", LoadFn.mmbench_video)
    video_safetybench_all = ConfigItem("VideoSafetybench-all", [LoadFn.video_safetybench_harmful, LoadFn.video_safetybench_ben],policy=PolicyHub.video_safetybench, risk_mapper=relabel_from(f"{LABEL_DIR}/video_safetybench_*.jsonl"))
    video_safetybench = ConfigItem("VideoSafetybench", LoadFn.video_safetybench_harmful, policy=PolicyHub.video_safetybench, risk_mapper=relabel_from(f"{LABEL_DIR}/video_safetybench_*.jsonl"))
    video_safetybench_ben = ConfigItem("VideoSafetybench-ben", LoadFn.video_safetybench_ben, policy=PolicyHub.video_safetybench, risk_mapper=relabel_from(f"{LABEL_DIR}/video_safetybench_*.jsonl"))
    safewatch_real = ConfigItem("SafeWatch-real", LoadFn.safewatch_real, policy=PolicyHub.safewatch)
    safewatch_genai = ConfigItem("SafeWatch-genai", LoadFn.safewatch_genai, policy=PolicyHub.safewatch)
    safewatch = ConfigItem("SafeWatch", [LoadFn.safewatch_real, LoadFn.safewatch_genai], policy=PolicyHub.safewatch)
 

class AudioModality:
    false_reject_alpacaeval = ConfigItem("FalseReject-alpacaeval", LoadFn.voicebench_alpacaeval, callback=lambda ds: \
                                         replace_text_prompt(ds, [phrase.value for phrase in AudioKeyphrase]))

    voicebench_alpacaeval = ConfigItem("VoiceBench-alpacaeval", LoadFn.voicebench_alpacaeval)
    voicebench_alpacaeval_audio = ConfigItem("VoiceBench-alpacaeval-audio", LoadFn.voicebench_alpacaeval_audio)
    voicebench_bbh = ConfigItem("VoiceBench-bbh", LoadFn.voicebench_bbh)
    voicebench_commoneval = ConfigItem("VoiceBench-commoneval", LoadFn.voicebench_commoneval)
    voicebench_ifeval = ConfigItem("VoiceBench-ifeval", LoadFn.voicebench_ifeval)
    voicebench_openbookqa = ConfigItem("VoiceBench-openbookqa", LoadFn.voicebench_openbookqa)
    voicebench_wildvoice = ConfigItem("VoiceBench-wildvoice", LoadFn.voicebench_wildvoice)
    voicebench_advbench = ConfigItem("VoiceBench-advbench", LoadFn.voicebench_advbench, risk_mapper=relabel_from(f"{LABEL_DIR}/advbench_*.jsonl"))
    voicebench_advbench_audio = ConfigItem("VoiceBench-advbench-audio", LoadFn.voicebench_advbench_audio, risk_mapper=relabel_from(f"{LABEL_DIR}/advbench_*.jsonl"))
    
    aiah = ConfigItem("AIAH", [LoadFn.aiah_alignment, LoadFn.aiah_spelling, LoadFn.aiah_nonspeech_empty, LoadFn.aiah_nonspeech_origin, LoadFn.aiah_nonspeech_standard],
                      policy=PolicyHub.aiah, risk_mapper=relabel_from(f'{LABEL_DIR}/aiah_spelling_*.jsonl'))
    aiah_alignment = ConfigItem("AIAH-alignment", LoadFn.aiah_alignment, policy=PolicyHub.aiah)
    aiah_nonspeech = ConfigItem("AIAH-nonspeech", [LoadFn.aiah_nonspeech_empty, LoadFn.aiah_nonspeech_origin, LoadFn.aiah_nonspeech_standard], policy=PolicyHub.aiah)
    aiah_spelling = ConfigItem("AIAH-spelling", LoadFn.aiah_spelling, risk_mapper=relabel_from(f'{LABEL_DIR}/aiah_spelling_*.jsonl'))
    aiah_alignment_prompt_rewrite = ConfigItem("AIAH-alignment-prompt-rewrite", LoadFn.aiah_alignment, policy=PolicyHub.aiah,
                                               callback=lambda ds: replace_text_prompt(ds, [AudioKeyphrase.Audio_Prompt_v1.value]))

    ajailbench = ConfigItem("AJailbench", [LoadFn.ajailbench_origin_prompt, LoadFn.ajailbench_apt_prompt], policy=PolicyHub.ajailbench)
    ajailbench_origin = ConfigItem("AJailbench-Origin", LoadFn.ajailbench_origin, policy=PolicyHub.ajailbench)
    ajailbench_origin_prompt = ConfigItem("AJailbench-Origin-prompt", LoadFn.ajailbench_origin_prompt, policy=PolicyHub.ajailbench)
    ajailbench_origin_prompt_rewrite = ConfigItem("AJailbench-Origin-prompt-rewrite", LoadFn.ajailbench_origin_prompt, policy=PolicyHub.ajailbench,
                                                  callback=lambda ds: replace_text_prompt(ds, [AudioKeyphrase.Audio_Prompt_v1.value]))
    ajailbench_apt = ConfigItem("AJailbench-APT", LoadFn.ajailbench_apt, policy=PolicyHub.ajailbench)
    ajailbench_apt_prompt = ConfigItem("AJailbench-APT-prompt", LoadFn.ajailbench_apt_prompt, policy=PolicyHub.ajailbench)
    ajailbench_apt_prompt_rewrite = ConfigItem("AJailbench-APT-prompt-rewrite", LoadFn.ajailbench_apt_prompt, policy=PolicyHub.ajailbench,
                                                  callback=lambda ds: replace_text_prompt(ds, [AudioKeyphrase.Audio_Prompt_v1.value]))
    omni_safetybench_unimodal_a = ConfigItem("OmniSafetybench-unimodal-A", LoadFn.omni_safetybench_unimodal_a, policy=PolicyHub.omni_safetybench)
    omni_safetybench_dual_ta = ConfigItem("OmniSafetybench-dual-TA", LoadFn.omni_safetybench_dual_ta, policy=PolicyHub.omni_safetybench)
    safebench_ta = ConfigItem("SafeBench-TA", LoadFn.safebench_ta, policy=PolicyHub.safebench, risk_mapper=relabel_from(f'{LABEL_DIR}/safebench_*.jsonl'))

    
class OmniModality:
    omni_safetybench_omni_tia = ConfigItem("OmniSafetybench-omni-TVA", LoadFn.omni_safetybench_omni_tia, policy=PolicyHub.omni_safetybench)
    omni_safetybench_omni_tva = ConfigItem("OmniSafetybench-omni-TViA", LoadFn.omni_safetybench_omni_tva, policy=PolicyHub.omni_safetybench)
    safebench_tia = ConfigItem("SafeBench-TVA", LoadFn.safebench_tia, policy=PolicyHub.safebench, risk_mapper=relabel_from(f'{LABEL_DIR}/safebench_*.jsonl'))
    

class CustomOmniModality:
    omni_custom_T = ConfigItem("Omni-Custom-T", LoadFn.omni_custom_T, risk_mapper=relabel_from(f'{LABEL_DIR}/omniguard_custom_*.jsonl'))
    omni_custom_I = ConfigItem("Omni-Custom-I", LoadFn.omni_custom_I, risk_mapper=relabel_from(f'{LABEL_DIR}/omniguard_custom_*.jsonl'))
    omni_custom_V = ConfigItem("Omni-Custom-V", LoadFn.omni_custom_V, risk_mapper=relabel_from(f'{LABEL_DIR}/omniguard_custom_*.jsonl'))
    omni_custom_A = ConfigItem("Omni-Custom-A", LoadFn.omni_custom_A, risk_mapper=relabel_from(f'{LABEL_DIR}/omniguard_custom_*.jsonl'))

    omni_custom_TI = ConfigItem("Omni-Custom-TI", LoadFn.omni_custom_T_I, risk_mapper=relabel_from(f'{LABEL_DIR}/omniguard_custom_*.jsonl'))
    omni_custom_ST_I = ConfigItem("Omni-Custom-ST-I", LoadFn.omni_custom_ST_I, risk_mapper=relabel_from(f'{LABEL_DIR}/omniguard_custom_*.jsonl'))
    omni_custom_T_SI = ConfigItem("Omni-Custom-T-SI", LoadFn.omni_custom_T_SI, risk_mapper=relabel_from(f'{LABEL_DIR}/omniguard_custom_*.jsonl'))

    omni_custom_TV = ConfigItem("Omni-Custom-TV", LoadFn.omni_custom_T_V, risk_mapper=relabel_from(f'{LABEL_DIR}/omniguard_custom_*.jsonl'))
    omni_custom_ST_V = ConfigItem("Omni-Custom-ST-V", LoadFn.omni_custom_ST_V, risk_mapper=relabel_from(f'{LABEL_DIR}/omniguard_custom_*.jsonl'))
    omni_custom_T_SV = ConfigItem("Omni-Custom-T-SV", LoadFn.omni_custom_T_SV, risk_mapper=relabel_from(f'{LABEL_DIR}/omniguard_custom_*.jsonl'))
    
    omni_custom_TA = ConfigItem("Omni-Custom-TA", LoadFn.omni_custom_T_A, risk_mapper=relabel_from(f'{LABEL_DIR}/omniguard_custom_*.jsonl'))
    omni_custom_ST_A = ConfigItem("Omni-Custom-ST-A", LoadFn.omni_custom_ST_A, risk_mapper=relabel_from(f'{LABEL_DIR}/omniguard_custom_*.jsonl'))
    omni_custom_T_SA = ConfigItem("Omni-Custom-T-SA", LoadFn.omni_custom_T_SA, risk_mapper=relabel_from(f'{LABEL_DIR}/omniguard_custom_*.jsonl'))

    omni_custom_IV = ConfigItem("Omni-Custom-IV", LoadFn.omni_custom_I_V, risk_mapper=relabel_from(f'{LABEL_DIR}/omniguard_custom_*.jsonl'))
    omni_custom_SI_V = ConfigItem("Omni-Custom-SI-V", LoadFn.omni_custom_SI_V, risk_mapper=relabel_from(f'{LABEL_DIR}/omniguard_custom_*.jsonl'))
    omni_custom_I_SV = ConfigItem("Omni-Custom-I-SV", LoadFn.omni_custom_I_SV, risk_mapper=relabel_from(f'{LABEL_DIR}/omniguard_custom_*.jsonl'))
    
    omni_custom_IA = ConfigItem("Omni-Custom-IA", LoadFn.omni_custom_I_A, risk_mapper=relabel_from(f'{LABEL_DIR}/omniguard_custom_*.jsonl'))
    omni_custom_SI_A = ConfigItem("Omni-Custom-SI-A", LoadFn.omni_custom_SI_A, risk_mapper=relabel_from(f'{LABEL_DIR}/omniguard_custom_*.jsonl'))
    omni_custom_I_SA = ConfigItem("Omni-Custom-I-SA", LoadFn.omni_custom_I_SA, risk_mapper=relabel_from(f'{LABEL_DIR}/omniguard_custom_*.jsonl'))
    
    omni_custom_VA = ConfigItem("Omni-Custom-VA", LoadFn.omni_custom_V_A, risk_mapper=relabel_from(f'{LABEL_DIR}/omniguard_custom_*.jsonl'))
    omni_custom_SV_A = ConfigItem("Omni-Custom-SV-A", LoadFn.omni_custom_SV_A, risk_mapper=relabel_from(f'{LABEL_DIR}/omniguard_custom_*.jsonl'))
    omni_custom_V_SA = ConfigItem("Omni-Custom-V-SA", LoadFn.omni_custom_V_SA, risk_mapper=relabel_from(f'{LABEL_DIR}/omniguard_custom_*.jsonl'))

    omni_custom_TIA = ConfigItem("Omni-Custom-TIA", LoadFn.omni_custom_T_I_A, risk_mapper=relabel_from(f'{LABEL_DIR}/omniguard_custom_*.jsonl'))
    omni_custom_T_SI_SA = ConfigItem("Omni-Custom-T-SI-SA", LoadFn.omni_custom_T_SI_SA, risk_mapper=relabel_from(f'{LABEL_DIR}/omniguard_custom_*.jsonl'))
    omni_custom_ST_I_SA = ConfigItem("Omni-Custom-ST-I-SA", LoadFn.omni_custom_ST_I_SA, risk_mapper=relabel_from(f'{LABEL_DIR}/omniguard_custom_*.jsonl'))
    omni_custom_ST_SI_A = ConfigItem("Omni-Custom-ST-SI-A", LoadFn.omni_custom_ST_SI_A, risk_mapper=relabel_from(f'{LABEL_DIR}/omniguard_custom_*.jsonl'))
    omni_custom_ST_I_A = ConfigItem("Omni-Custom-ST-I-A", LoadFn.omni_custom_ST_I_A, risk_mapper=relabel_from(f'{LABEL_DIR}/omniguard_custom_*.jsonl'))
    omni_custom_T_SI_A = ConfigItem("Omni-Custom-T-SI-A", LoadFn.omni_custom_T_SI_A, risk_mapper=relabel_from(f'{LABEL_DIR}/omniguard_custom_*.jsonl'))
    omni_custom_T_I_SA = ConfigItem("Omni-Custom-T-I-SA", LoadFn.omni_custom_T_I_SA, risk_mapper=relabel_from(f'{LABEL_DIR}/omniguard_custom_*.jsonl'))

    omni_custom_TVA = ConfigItem("Omni-Custom-TVA", LoadFn.omni_custom_T_V_A, risk_mapper=relabel_from(f'{LABEL_DIR}/omniguard_custom_*.jsonl'))
    omni_custom_T_SV_SA = ConfigItem("Omni-Custom-T-SV-SA", LoadFn.omni_custom_T_SV_SA, risk_mapper=relabel_from(f'{LABEL_DIR}/omniguard_custom_*.jsonl'))
    omni_custom_ST_V_SA = ConfigItem("Omni-Custom-ST-V-SA", LoadFn.omni_custom_ST_V_SA, risk_mapper=relabel_from(f'{LABEL_DIR}/omniguard_custom_*.jsonl'))
    omni_custom_ST_SV_A = ConfigItem("Omni-Custom-ST-SV-A", LoadFn.omni_custom_ST_SV_A, risk_mapper=relabel_from(f'{LABEL_DIR}/omniguard_custom_*.jsonl'))
    omni_custom_ST_V_A = ConfigItem("Omni-Custom-ST-V-A", LoadFn.omni_custom_ST_V_A, risk_mapper=relabel_from(f'{LABEL_DIR}/omniguard_custom_*.jsonl'))
    omni_custom_T_SV_A = ConfigItem("Omni-Custom-T-SV-A", LoadFn.omni_custom_T_SV_A, risk_mapper=relabel_from(f'{LABEL_DIR}/omniguard_custom_*.jsonl'))
    omni_custom_T_V_SA = ConfigItem("Omni-Custom-T-V-SA", LoadFn.omni_custom_T_V_SA, risk_mapper=relabel_from(f'{LABEL_DIR}/omniguard_custom_*.jsonl'))

    omni_custom_TIV = ConfigItem("Omni-Custom-TIV", LoadFn.omni_custom_T_I_V, risk_mapper=relabel_from(f'{LABEL_DIR}/omniguard_custom_*.jsonl'))
    omni_custom_T_SI_SV = ConfigItem("Omni-Custom-T-SI-SV", LoadFn.omni_custom_T_SI_SV, risk_mapper=relabel_from(f'{LABEL_DIR}/omniguard_custom_*.jsonl'))
    omni_custom_ST_I_SV = ConfigItem("Omni-Custom-ST-I-SV", LoadFn.omni_custom_ST_I_SV, risk_mapper=relabel_from(f'{LABEL_DIR}/omniguard_custom_*.jsonl'))
    omni_custom_ST_SI_V = ConfigItem("Omni-Custom-ST-SI-V", LoadFn.omni_custom_ST_SI_V, risk_mapper=relabel_from(f'{LABEL_DIR}/omniguard_custom_*.jsonl'))
    omni_custom_ST_I_V = ConfigItem("Omni-Custom-ST-I-V", LoadFn.omni_custom_ST_I_V, risk_mapper=relabel_from(f'{LABEL_DIR}/omniguard_custom_*.jsonl'))
    omni_custom_T_SI_V = ConfigItem("Omni-Custom-T-SI-V", LoadFn.omni_custom_T_SI_V, risk_mapper=relabel_from(f'{LABEL_DIR}/omniguard_custom_*.jsonl'))
    omni_custom_T_I_SV = ConfigItem("Omni-Custom-T-I-SV", LoadFn.omni_custom_T_I_SV, risk_mapper=relabel_from(f'{LABEL_DIR}/omniguard_custom_*.jsonl'))

    omni_custom_IAV = ConfigItem("Omni-Custom-IAV", LoadFn.omni_custom_I_A_V, risk_mapper=relabel_from(f'{LABEL_DIR}/omniguard_custom_*.jsonl'))
    omni_custom_I_SA_SV = ConfigItem("Omni-Custom-I-SA-SV", LoadFn.omni_custom_I_SA_SV, risk_mapper=relabel_from(f'{LABEL_DIR}/omniguard_custom_*.jsonl'))
    omni_custom_SI_A_SV = ConfigItem("Omni-Custom-SI-A-SV", LoadFn.omni_custom_SI_A_SV, risk_mapper=relabel_from(f'{LABEL_DIR}/omniguard_custom_*.jsonl'))
    omni_custom_SI_SA_V = ConfigItem("Omni-Custom-SI-SA-V", LoadFn.omni_custom_SI_SA_V, risk_mapper=relabel_from(f'{LABEL_DIR}/omniguard_custom_*.jsonl'))
    omni_custom_SI_A_V = ConfigItem("Omni-Custom-SI-A-V", LoadFn.omni_custom_SI_A_V, risk_mapper=relabel_from(f'{LABEL_DIR}/omniguard_custom_*.jsonl'))
    omni_custom_I_SA_V = ConfigItem("Omni-Custom-I-SA-V", LoadFn.omni_custom_I_SA_V, risk_mapper=relabel_from(f'{LABEL_DIR}/omniguard_custom_*.jsonl'))
    omni_custom_I_A_SV = ConfigItem("Omni-Custom-I-A-SV", LoadFn.omni_custom_I_A_SV, risk_mapper=relabel_from(f'{LABEL_DIR}/omniguard_custom_*.jsonl'))
    
    omni_custom_all = ConfigItem("Omni-Custom", LoadFn.omni_custom, risk_mapper=relabel_from(f'{LABEL_DIR}/omniguard_custom_*.jsonl'))
    omni_custom_T_SI_SA_SV = ConfigItem("Omni-Custom-T-SI-SA-SV", LoadFn.omni_custom_T_SI_SA_SV, risk_mapper=relabel_from(f'{LABEL_DIR}/omniguard_custom_*.jsonl'))
    omni_custom_ST_I_SA_SV = ConfigItem("Omni-Custom-ST-I-SA-SV", LoadFn.omni_custom_ST_I_SA_SV, risk_mapper=relabel_from(f'{LABEL_DIR}/omniguard_custom_*.jsonl'))
    omni_custom_ST_SI_A_SV = ConfigItem("Omni-Custom-ST-SI-A-SV", LoadFn.omni_custom_ST_SI_A_SV, risk_mapper=relabel_from(f'{LABEL_DIR}/omniguard_custom_*.jsonl'))
    omni_custom_ST_SI_SA_V = ConfigItem("Omni-Custom-ST-SI-SA-V", LoadFn.omni_custom_ST_SI_SA_V, risk_mapper=relabel_from(f'{LABEL_DIR}/omniguard_custom_*.jsonl'))
    omni_custom_ST_I_A_V = ConfigItem("Omni-Custom-ST-I-A-V", LoadFn.omni_custom_ST_I_A_V, risk_mapper=relabel_from(f'{LABEL_DIR}/omniguard_custom_*.jsonl'))
    omni_custom_T_SI_A_V = ConfigItem("Omni-Custom-T-SI-A-V", LoadFn.omni_custom_T_SI_A_V, risk_mapper=relabel_from(f'{LABEL_DIR}/omniguard_custom_*.jsonl'))
    omni_custom_T_I_SA_V = ConfigItem("Omni-Custom-T-I-SA-V", LoadFn.omni_custom_T_I_SA_V, risk_mapper=relabel_from(f'{LABEL_DIR}/omniguard_custom_*.jsonl'))
    omni_custom_T_I_A_SV = ConfigItem("Omni-Custom-T-I-A-SV", LoadFn.omni_custom_T_I_A_SV, risk_mapper=relabel_from(f'{LABEL_DIR}/omniguard_custom_*.jsonl'))


def load_modality_dataset(modality, dataset):
    if modality.lower() == 'text':
        return getattr(TextModality, dataset)
    elif modality.lower() == 'vision':
        return getattr(VisionModality, dataset)
    elif modality.lower() == 'audio':
        return getattr(AudioModality, dataset)
    elif modality.lower() == 'omni':
        return getattr(OmniModality, dataset)
    else:
        raise ValueError(f"Unknown modality: {modality}")