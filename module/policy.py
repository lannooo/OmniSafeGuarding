PROMPT_UNSAFE = 1
PROMPT_SAFE   = 0

class Const:
    BENIGN = 'BNI'
    UNKNOWN = 'UNK'
    IA = 'IA'
    PH = 'PH'
    EH = 'EH'
    FA = 'FA'
    HS = 'HS'
    SC = 'SC'
    PV = 'PV'
    CS = 'CS'
    UA = 'UA'
    ST = 'ST'
    CTRL_RELABEL = '<RELABEL>'   # A special type, means it needs relabel mapping from extra label information to relabel risk category
    CTRL_DROP = '<DROP>'         # A special type, means it needs to be removed

UNIFIED_POLICY_TOXINOMY = {
    Const.IA: "Illegal Activity and Crimes",
    Const.PH: "Physical Harm",
    Const.EH: "Economical Harm",
    Const.FA: "Fraud and Deceptive Activity",
    Const.HS: "Hateful and Toxic Speech",
    Const.SC: "Sexual Content",
    Const.PV: "Privacy Violation",
    Const.CS: "Cyber Security",
    Const.UA: "Unlicensed Consultation and Advice",
    Const.ST: "Controversial Sensitive Topics"
}

UNIFIED_POLICY_SUB_CATEGORY = {
    Const.IA: [],
    Const.PH: [],
    Const.EH: [],
    Const.FA: [],
    Const.HS: [],
    Const.SC: [],
    Const.PV: [],
    Const.CS: [],
    Const.UA: [],
    Const.ST: []
}

class DatasetKeys:
    ForbiddenQuestions = "forbidden_questions"
    JailbreakBench = "jailbreakbench"
    SaladBench = "saladbench"
    HarmBench = "harmbench"
    XSTest = "xstest"
    SimpleSafetyTests = "simplesafety"
    Beavertails = "beavertails"
    SafeRLHF = "saferlhf"
    SafeSora = "safesora"
    Wildguard = "wildguard"
    ToxicChat = "toxicchat"
    Aegis2 = "aegis2"

    Figstep = "figstep"
    MMSafetyBench = "mm-safetybench"
    JailbreakV = "jailbreakv-28k"
    Hades = "hades"
    RTVLM = "rtvlm"
    VLGuard = "vlguard"
    VLSBench = "vlsbench"
    SIUO = 'siuo'
    HoliSafe = "holisafe"
    
    AIAH = "aiah"
    AJailBench = "ajailbench"
    SafeWatch = "safewatch"
    VideoSafetybench = "video-safetybench"
    OmniSafetyBench = "omni-safetybench"
    SafeBench = "safebench"

_POLICY_MAPPING = {
    DatasetKeys.MMSafetyBench: {
        'Illegal_Activitiy': Const.IA, 
        'Physical_Harm': Const.PH, 
        'EconomicHarm': Const.EH, 
        'Fraud': Const.FA, 
        'HateSpeech': Const.HS, 
        'Sex': Const.SC,
        'Malware_Generation': Const.CS, 
        'Privacy_Violence': Const.PV, 
        'Gov_Decision': Const.UA, 
        'Financial_Advice': Const.UA, 
        'Health_Consultation': Const.UA, 
        'Legal_Opinion': Const.UA, 
        'Political_Lobbying': Const.ST, 
    },

     DatasetKeys.Figstep: {
        'Illegal Activity': Const.IA,
        'Physical Harm': Const.PH,
        'Fraud': Const.FA,
        'Hate Speech': Const.HS,
        'Adult Content': Const.SC,
        'Privacy Violation': Const.PV,
        'Malware Generation': Const.CS,
        'Health Consultation': Const.UA,
        'Financial Advice': Const.UA,
        'Legal Opinion': Const.UA,
    },
    
    DatasetKeys.JailbreakV: {
        'Illegal Activity': Const.IA,
        'Physical Harm': Const.PH,
        'Economic Harm': Const.EH,
        'Animal Abuse': Const.PH,
        'Fraud': Const.FA,
        'Bias': Const.HS,
        'Hate Speech': Const.HS,
        'Malware': Const.CS,
        'Privacy Violation': Const.PV,
        'Child Abuse': Const.CTRL_RELABEL,
        'Child Abuse Content': Const.CTRL_RELABEL, # need relabel (IA, SC, etc)
        'Violence': Const.CTRL_RELABEL, # relabel this category (IA, PH, HS, etc)
        'Government Decision': Const.UA,
        'Health Consultation': Const.UA,
        'Tailored Unlicensed Advice': Const.UA,
        'Political Sensitivity': Const.ST,
        'Unethical Behavior': Const.CTRL_RELABEL, # relabel this category (FA, etc)
    },

    DatasetKeys.Hades: {
        'Animal': Const.PH,
        'Financial': Const.CTRL_RELABEL,
        'Privacy': Const.CTRL_RELABEL,
        'Self-Harm': Const.PH,
        'Violence': Const.CTRL_RELABEL,
    },

    DatasetKeys.VLSBench: {
        'Abuse and Physical Altercations': Const.PH,
        'Adult Content': Const.SC,
        'Cultural Xenophobia': Const.HS,
        'Cybercrime': Const.CS,
        'Drug-Related Offenses': Const.IA,
        'Erotic Visual Content': Const.SC,
        'Gender and Sexual Orientation Discrimination': Const.HS,
        'Human Trafficking and Exploitation': Const.IA,
        'Identity Theft and Impersonation': Const.FA,
        'Physical Self-Injury': Const.PH,
        'Property Crimes': Const.IA,
        'Psychological and Disordered Harm': Const.PH,
        'Public Violence and Rioting': Const.IA,
        'Racial and Ethnic Discrimination': Const.HS,
        'Religious Intolerance': Const.HS,
        'Risky or Dangerous Behaviors': Const.PH,
        'Substance Abuse and Poisoning': Const.PH,
        'Unauthorized Data Collection': Const.PV,
        'Weapon-Related Violence': Const.IA,
    },

    DatasetKeys.RTVLM: {
        'HateSpeech': Const.HS,
        'Illegal_Activitiy': Const.CTRL_RELABEL,
        'jailbreak': Const.CTRL_RELABEL,
        'politics': Const.CTRL_DROP,
        'politics_Political_Lobbying': Const.ST,
        'racial': Const.CTRL_DROP,
        'racial_Gov_Decision': Const.UA,
        'racial_Political_Lobbying': Const.ST,
    },

    DatasetKeys.SIUO: {
        'controversial topics, politics': Const.ST,
        'dangerous behavior': Const.PH,
        'discrimination & stereotyping': Const.HS,
        'illegal activities & crime': Const.IA,
        'information misinterpretation': Const.CTRL_DROP,
        'morality': Const.ST,
        'privacy violation': Const.PV,
        'religion beliefs': Const.ST,
        'self-harm': Const.PH,
    },

    DatasetKeys.HoliSafe: {
        # 'BNI': 682,
        'animal_abuse': Const.PH,
        'drug_related_hazards': Const.IA,
        'disability_discrimination': Const.HS,
        'gender': Const.HS,
        'harassment': Const.HS, # TODO relabel
        'facial_data_exposure': Const.PV,
        'identity_data_exposure': Const.PV,
        'financial_advice': Const.UA,
        'medical_advice': Const.UA,
        'obscene_gestures': Const.HS,
        'physical_altercation': Const.PH,
        'physical_self_injury': Const.PH,
        'property_crime': Const.IA,
        'race': Const.HS,
        'religion': Const.ST,
        'sexual_content': Const.SC,
        'suicide': Const.PH,
        'terrorism': Const.IA,
        'weapon_related_violence': Const.IA,
    },

    DatasetKeys.VLGuard: {
        'disinformation': Const.FA,
        'other': Const.CTRL_RELABEL, # TODO Const.CTRL_RELABEL,
        'personal data': Const.PV,
        'political': Const.ST,
        'professional advice': Const.UA,
        'race': Const.HS,
        'sex': Const.HS,
        'sexually explicit': Const.SC,
        'violence': Const.IA,  # TODO Const.CTRL_RELABEL
    }, 

    DatasetKeys.VideoSafetybench: {
        'Animal_Abuse': Const.PH,
        'Mass_Violence': Const.IA,
        'Child_Abuse': Const.IA,
        'Weapons_Crimes': Const.IA,
        'Cyber_Crimes': Const.CS,
        'Personal_Crimes': Const.IA,
        'Drug_Crimes': Const.IA,
        'Property_Crimes': Const.IA,
        'Financial_Crimes': Const.CTRL_RELABEL,
        'Sex_Trafficking': Const.IA,
        'Sexual_Harassment': Const.IA,
        'Sexual_Assault': Const.IA,
        'Prostitution': Const.IA,
        'Sexual_Abuse_of_Children': Const.IA,
        'Inappropriate_Relationships_with_Children': Const.CTRL_RELABEL,
        'Child_Sexual_Abuse_Material': Const.SC,
        'Public_Figure_Defamation': Const.FA,
        'Event_Defamation': Const.FA,
        'Business_or_Organizational_Defamation': Const.FA,
        'Health_Consultation': Const.UA,
        'Legal_Opinion': Const.UA,
        'Financial_Advice': Const.UA,
        'Identity_Theft_and_Impersonation': Const.FA,
        'Non-Consensual_Intrusion': Const.PV,
        'Unauthorized_Data_Intrusion': Const.PV,
        'Trademark_Infringement': Const.CTRL_DROP,
        'Copyright_Infringement': Const.CTRL_DROP,
        'Patent_Infringement': Const.CTRL_DROP,
        'High-yield_Explosive_Weapons': Const.PH,
        'Chemical_Weapons': Const.PH,
        'Biological_Weapons': Const.PH,
        'Nuclear_Weapons': Const.CTRL_RELABEL,
        'Radiological_Weapons': Const.CTRL_RELABEL,
        'Religious_Intolerance': Const.HS,
        'Disability_Discrimination': Const.HS,
        'Sexual_Orientation_Discrimination': Const.HS,
        'Gender_Discrimination': Const.HS,
        'Racial_and_Ethnic_Discrimination': Const.HS,
        'Disordered_Eating': Const.PH,
        'Self-injury': Const.PH,
        'Suicide': Const.PH,
        'Sexual_Acts': Const.SC,
        'Sexualized_Body': Const.SC,
        'Erotic_Chats': Const.SC,
        'Election_Materials': Const.CTRL_RELABEL,
        'Political_Campaign': Const.ST,
        'Voter_Participation': Const.CTRL_RELABEL,
        'Election_Systems': Const.CS
    },

    
    DatasetKeys.AIAH: {
        'Adult_Content': Const.SC,
        'Fraud': Const.FA,
        'Hate_Speech': Const.HS,
        'Illegal_Activity': Const.IA,
        'Malware_Generation': Const.CS,
        'Physical_Harm': Const.PH,
        'Privacy_Violation': Const.PV,
    },
    
    DatasetKeys.AJailBench: {
        'Disinformation': Const.FA,
        'Economic harm': Const.EH,
        'Expert advice': Const.UA,
        'Fraud/Deception': Const.FA,
        'Government decision-making': Const.UA,
        'Harassment/Discrimination': Const.HS,
        'Malware/Hacking': Const.CS,
        'Physical harm': Const.PH,
        'Privacy': Const.PV,
        'Sexual/Adult content': Const.SC,
        'Adult Content': Const.SC,
        'Social stereotypes and unfair discrimination': Const.HS,
        'Toxic language (hate speech)': Const.HS,
        'Mental Health or Overreliance Crisis': Const.UA,
        'Treat Chatbot as a Human': Const.ST,
        'Risks from leaking or inferring sensitive information (organization/gov)': Const.PV,
        'Nudging or advising users to perform unethical or unsafe actions': Const.ST,
        'Reducing the cost of disinformation campaigns': Const.FA,
        'Causing material harm by disseminating misinformation e.g. in medicine or law': Const.UA,
        'Disseminating false or misleading information': Const.FA,
        'Malwarke/Hacking': Const.CS,
        'Illegal Activity': Const.CS  # only one sample related to hacking into acccount is marked as this
    },

    DatasetKeys.OmniSafetyBench: {
        'EconomicHarm': Const.EH,
        'Fraud': Const.FA,
        'HateSpeech': Const.HS,
        'Illegal_Activity': Const.IA,
        'Malware_Generation': Const.CS,
        'Physical_Harm': Const.PH,
        'Privacy_Violence': Const.PV,
        'Sex': Const.SC,
    },

    DatasetKeys.SafeBench: {
        '1Pornographic Content': Const.SC,
        '2Hate Speech': Const.HS,
        '3Fraudulent Activities': Const.FA,
        '4Medical Advice': Const.UA,
        '5Legal Advice': Const.CTRL_RELABEL,
        '6Investment Advice': Const.CTRL_RELABEL,
        '7Personal Privacy': Const.CTRL_RELABEL,
        '8Physical harm': Const.PH,
        '9Illegal Activities': Const.CTRL_RELABEL,
        '10Political Opinions': Const.ST,
        '11Religious Opinions': Const.ST,
        '12Bias': Const.HS,
        '13Penetration Testing': Const.CS,
        '14Vulnerability Exploitation': Const.CS,
        '15Intellectual Property Infringement': Const.CTRL_DROP,
        '16Child Abuse': Const.IA,
        '17Malicious Code Generation': Const.CS,
        '18Trade Confidentiality': Const.CTRL_DROP,
        '19Hardware Security': Const.CS,
        '20Bioethics': Const.CTRL_RELABEL,
        '21Social Ethics': Const.CTRL_RELABEL,
        '22Psychological Harm': Const.PH,
        '23Defamatory Speech': Const.FA,
    },

    DatasetKeys.JailbreakBench: {
        'Disinformation': Const.FA,
        'Economic harm': Const.EH,
        'Expert advice': Const.UA,
        'Fraud/Deception': Const.FA,
        'Government decision-making': Const.ST,
        'Harassment/Discrimination': Const.HS,
        'Malware/Hacking': Const.CS,
        'Physical harm': Const.PH,
        'Privacy': Const.PV,
        'Sexual/Adult content': Const.SC,
    },

    DatasetKeys.ForbiddenQuestions: {
        'Illegal Activity': Const.IA, 
        'Hate Speech': Const.HS, 
        'Malware': Const.CS, 
        'Physical Harm': Const.PH,
        'Economic Harm': Const.EH, 
        'Fraud': Const.FA, 
        'Pornography': Const.SC, 
        'Political Lobbying': Const.ST,
        'Privacy Violence': Const.PV, 
        'Legal Opinion': Const.UA, 
        'Financial Advice': Const.UA,
        'Health Consultation': Const.UA, 
        'Gov Decision': Const.UA
    }, 
    DatasetKeys.SafeWatch: {
        'abuse': Const.IA,
        'acting': Const.CTRL_DROP, # Const.FA maybe
        'aigc content': Const.FA,
        'animal_abuse': Const.PH,
        'arson and explosion': Const.PH,
        'assault': Const.IA,
        'benign': Const.BENIGN,
        'campus_bully': Const.HS,
        'child_abuse': Const.IA,
        'drugs': Const.IA,
        'evident': Const.SC,
        'extremely disturbing content': Const.PH,
        'fighting': Const.IA,
        'hentai': Const.SC,
        'implication': Const.SC,
        'incitement to mental depression': Const.HS,
        'incitement to violence': Const.HS,
        'misinformation': Const.FA,
        'out-of-date': Const.CTRL_DROP, # FA maybe
        'robbery and burglary': Const.IA,
        'sexual bullying': Const.SC,
        'sexual violence': Const.SC,
        'shooting': Const.IA,
        'shoplifting and stealing': Const.IA,
        'subtle': Const.SC,
        'suicide and self-harm': Const.PH,
        'vandalism': Const.ST,
        'war and military actions': Const.PH,
    },
    DatasetKeys.SafeSora: {
        'BNI': Const.CTRL_RELABEL,
        'Unknown': Const.CTRL_DROP,
        'animal_abuse': Const.PH,
        'child_abuse': Const.IA,
        'contraband': Const.CTRL_RELABEL,
        'controversial': Const.ST,
        'crime': Const.IA,
        'hate': Const.CTRL_RELABEL,
        'other_discrimination': Const.HS,
        'other_harmful': Const.CTRL_RELABEL,
        'porn': Const.SC,
        'racism': Const.HS,
        'terrorism': Const.IA,
        'violence': Const.CTRL_RELABEL,
    },

    DatasetKeys.SafeRLHF: {
        'benign': Const.CTRL_RELABEL, # 6263,
        'Animal Abuse': Const.PH,
        'Copyright Issues': Const.CTRL_DROP,
        'Cybercrime': Const.CS,
        'Discriminatory Behavior': Const.HS,
        'Disrupting Public Order': Const.CTRL_RELABEL, # 48,
        'Drugs': Const.IA,
        'Economic Crime': Const.CTRL_RELABEL, # 732,
        'Endangering National Security': Const.CTRL_RELABEL, # 463,
        'Endangering Public Health': Const.CTRL_RELABEL, # 102,
        'Environmental Damage': Const.PH,
        'Human Trafficking': Const.IA,
        'Insulting Behavior': Const.CTRL_RELABEL, # 598,
        'Mental Manipulation': Const.CTRL_RELABEL, # 628,
        'Physical Harm': Const.CTRL_RELABEL, # 137,
        'Privacy Violation': Const.CTRL_RELABEL, # 879,
        'Psychological Harm': Const.PH,
        'Sexual Content': Const.SC,
        'Violence': Const.CTRL_RELABEL, # 609,
        'White-Collar Crime': Const.IA,
    },

    DatasetKeys.Beavertails: {
        'benign': Const.CTRL_RELABEL,
        'animal_abuse': Const.PH,
        'child_abuse': Const.IA,
        'controversial_topics,politics': Const.ST,
        'discrimination,stereotype,injustice': Const.HS,
        'drug_abuse,weapons,banned_substance': Const.IA,
        'financial_crime,property_crime,theft': Const.IA,
        'hate_speech,offensive_language': Const.HS,
        'misinformation_regarding_ethics,laws_and_safety': Const.CTRL_RELABEL, 
        'non_violent_unethical_behavior': Const.CTRL_RELABEL, 
        'privacy_violation': Const.PV,
        'self_harm': Const.PH,
        'sexually_explicit,adult_content': Const.SC,
        'terrorism,organized_crime': Const.IA,
        'violence,aiding_and_abetting,incitement': Const.CTRL_RELABEL 
    },

    DatasetKeys.SimpleSafetyTests: {
        "Child Abuse": Const.CTRL_RELABEL,
        "Illegal and Highly Regulated items": Const.CTRL_RELABEL,
        'Physical Harm and Violence': Const.CTRL_RELABEL,
        'Scams and Fraud': Const.CTRL_RELABEL,
        'Suicide, Self-Harm, and Eating Disorders': Const.PH
    },

    DatasetKeys.XSTest: {
        'benign': Const.BENIGN,
        'contrast_definitions': Const.CTRL_RELABEL,
        'contrast_discr': Const.CTRL_RELABEL,
        'contrast_figurative_language': Const.CTRL_RELABEL,
        'contrast_historical_events': Const.CTRL_RELABEL,
        'contrast_homonyms': Const.CTRL_RELABEL,
        'contrast_privacy': Const.CTRL_RELABEL,
        'contrast_safe_contexts': Const.CTRL_RELABEL,
        'contrast_safe_targets': Const.CTRL_RELABEL
    },

    DatasetKeys.SaladBench: {
        'O1: Hate Speech': Const.HS,
        'O2: Harass, Threaten, or Bully An Individual': Const.CTRL_RELABEL,
        "O3: Enjoying someone else's Pain": Const.CTRL_RELABEL, # TODO relabel
        'O4: Terrorism': Const.IA,
        'O5: Violent Content': Const.CTRL_RELABEL, # TODO relabel
        'O6: Child Abuse': Const.IA,
        'O7: Insult': Const.HS,
        'O8: Racial and Ethnic Stereotyping': Const.HS,
        'O9: Age Stereotyping': Const.HS,
        'O10: Disability Stereotyping': Const.HS,
        'O11: Gender Stereotyping': Const.HS,
        'O12: Religious Stereotyping': Const.HS,
        'O13: Location Stereotyping': Const.HS,
        'O14: Erotic Chat': Const.SC,
        'O15: Sexual Activity': Const.SC,
        'O16: Pornography': Const.SC,
        'O17: Law Advice': Const.CTRL_RELABEL, # TODO relabel
        'O18: Common Sense': Const.CTRL_DROP,  # removed
        'O19: Illegal Drugs and Regulated/Controlled Substances': Const.IA,
        'O20: Medical Advice': Const.CTRL_RELABEL, # TODO relabel
        'O21: Historical Revisionism': Const.FA,
        'O22: Science Denialism': Const.FA,
        'O23: Radicalization': Const.CTRL_DROP, # TODO removed or relabel
        'O24: Multi-level Marketing': Const.EH,
        'O25: Paypal Lending': Const.CTRL_RELABEL, # TODO relabel
        'O26: Gambling': Const.EH,
        'O27: Labor Issues': Const.ST,
        'O28: Trade Affairs': Const.CTRL_RELABEL,  # TODO relabel
        'O29: Fake News': Const.FA,
        'O30: Social Media Rumors': Const.FA,
        'O31: Government Secrets': Const.PV,
        'O32: Financial Data Leaks': Const.CTRL_RELABEL,  # TODO relabel
        # below not checked
        'O33: Inferring Personal Information': Const.PV,
        'O34: Leaking Private Information': Const.PV,
        'O35: Weapon Generation and Transportation': Const.PH,
        'O36: Management or Operation of Critical Infrastructure': Const.PH,
        'O37: Malware Generation': Const.CS,
        'O38: Cyber Attack': Const.CS,
        'O39: Biological and Chemical Harms': Const.PH,
        'O40: Group Defamation': Const.HS,
        'O41: Defamation Someone': Const.HS,
        'O42: Scams': Const.FA,
        'O43: Financial Fraud': Const.FA,
        'O44: Academic Dishonesty': Const.FA,
        'O45: Fake Review Generation': Const.FA,
        'O46: Fake Online Content': Const.FA,
        'O47: Disinformation': Const.FA,
        'O48: Spam': Const.FA,
        'O49: Pseudo-pharmaceuticals': Const.FA,
        'O50: Manipulation of Public Opinion': Const.ST,
        'O51: Campaign Materials': Const.ST,
        'O52: Illegitimate Surveillance': Const.PV,
        'O53: Financial Crimes': Const.IA,
        'O54: Drug-related Crimes': Const.IA,
        'O55: Sexual Offenses': Const.IA,
        'O56: Violent Crimes': Const.IA,
        'O57: Theft': Const.IA,
        'O58: Illegal Law Advice': Const.UA,
        'O59: Environmental Crimes': Const.ST,
        'O60: Traffic and Driving Offenses': Const.CTRL_DROP, # to be removed
        'O62: Self-Harm': Const.PH,
        'O63: Psychological Manipulation': Const.PH,
        'O64: Mimicking Writing Style': Const.CTRL_DROP,  # to be removed
        'O65: Plagiarism': Const.FA,
        'O66: Copyright Infringement': Const.CTRL_DROP, # to be removed
        'O67: Forge Identity': Const.FA,
    },
    DatasetKeys.HarmBench: {
        'chemical_biological': Const.PH,
        'cybercrime_intrusion': Const.CS,
        'harassment_bullying': Const.HS,
        'harmful': Const.CTRL_RELABEL,
        'illegal': Const.IA,
        'misinformation_disinformation': Const.FA,
    },
    DatasetKeys.Wildguard: {
        'benign': Const.BENIGN,
        'causing_material_harm_by_disseminating_misinformation': Const.CTRL_RELABEL,
        'copyright_violations': Const.CTRL_DROP, # to be removed
        'cyberattack': Const.CS,
        'defamation_encouraging_unethical_or_unsafe_actions': Const.CTRL_RELABEL,
        'disseminating_false_or_misleading_information_encouraging_disinformation_campaigns': Const.FA,
        'fraud_assisting_illegal_activities': Const.CTRL_RELABEL,
        'mental_health_over-reliance_crisis': Const.CTRL_RELABEL,
        'others': Const.CTRL_RELABEL,
        'private_information_individual': Const.PV,
        'sensitive_information_organization_government': Const.PV,
        'sexual_content': Const.CTRL_RELABEL,
        'social_stereotypes_and_unfair_discrimination': Const.HS,
        'toxic_language_hate_speech': Const.HS,
        'violence_and_physical_harm': Const.CTRL_RELABEL
    },
    DatasetKeys.ToxicChat: {
        'benign': Const.CTRL_RELABEL,
        'harassment': Const.CTRL_RELABEL,
        'hate': Const.CTRL_RELABEL,
        'self-harm': Const.CTRL_RELABEL,
        'self-harm/instructions': Const.CTRL_RELABEL,
        'sexual': Const.CTRL_RELABEL,
        'sexual/minors': Const.CTRL_RELABEL,
        'violence': Const.CTRL_RELABEL,
        'violence/graphic': Const.CTRL_RELABEL,
    },

    DatasetKeys.Aegis2: {
        'benign': Const.CTRL_RELABEL, # 905
        'Controlled/Regulated Substances': Const.IA,
        'Copyright/Trademark/Plagiarism': Const.CTRL_DROP,
        'Criminal Planning/Confessions': Const.CTRL_RELABEL, #317,
        'Fraud/Deception': Const.CTRL_RELABEL, #7,
        'Guns and Illegal Weapons': Const.IA,
        'Harassment': Const.CTRL_RELABEL, #69,
        'Hate/Identity Hate': Const.HS,
        'Illegal Activity': Const.CTRL_RELABEL, #3,
        'Immoral/Unethical': Const.CTRL_RELABEL, #3,
        'Malware': Const.CS,
        'Needs Caution': Const.CTRL_RELABEL, #19,
        'Other': Const.CTRL_RELABEL, #2,
        'PII/Privacy': Const.CTRL_RELABEL, #76,
        'Political/Misinformation/Conspiracy': Const.CTRL_RELABEL, #12,
        'Profanity': Const.HS,
        'Sexual': Const.SC,
        'Sexual (minor)': Const.CTRL_RELABEL, #8,
        'Suicide and Self Harm': Const.PH,
        'Threat': Const.CTRL_RELABEL, #7,
        'Unauthorized Advice': Const.CTRL_RELABEL, #22,
        'Violence': Const.CTRL_RELABEL, #135,
    }

}

class AgreementPolicy:
    # 三分之一
    NO_AGREEMENT=0.0
    QUARTER_AGREEMENT=0.25
    HALF_AGREEMENT=0.5
    FULL_AGREEMENT=1.0

class RelabelPolicy:
    KEEP_ALL=0
    DROP_TOXIC_TO_BENIGN=1
    DROP_BENIGN_TO_TOXIC=2
    DROP_ALL_CHANGE=3

class Policy:
    def __init__(self, mapping_name:str=None) -> None:
        self.categories = list(UNIFIED_POLICY_TOXINOMY.keys())
        self.category_index = {c: i for i, c in enumerate(self.categories)}
        self.inverse_category_index = {i: c for c, i in self.category_index.items()}
        self.relabel_drop_policy = RelabelPolicy.KEEP_ALL
        self.relabel_agreement_policy = AgreementPolicy.NO_AGREEMENT
        if mapping_name is not None and mapping_name in _POLICY_MAPPING:
            self.mapping = _POLICY_MAPPING[mapping_name]
        else:
            self.mapping = {}
        self.allow_risks(None)

    def drop(self, relabel_drop_policy):
        self.relabel_drop_policy = relabel_drop_policy
        return self
    
    def agreement(self, relabel_agreement_policy):
        self.relabel_agreement_policy = relabel_agreement_policy
        return self
    
    def allow_risks(self, allowed_risks):
        if allowed_risks is None:
            self.allowed_risks = self.categories + [Const.UNKNOWN, Const.BENIGN]
        elif isinstance(allowed_risks, list):
            self.allowed_risks = allowed_risks
        elif isinstance(allowed_risks, str):
            ori_risks = self.categories + [Const.UNKNOWN, Const.BENIGN]
            if allowed_risks.startswith('~'):
                # remove all the parsed items
                removed_items = allowed_risks[1:].split(',')
                for item in removed_items:
                    ori_risks.remove(item)
            elif allowed_risks.startswith('+'):
                added_items = allowed_risks[1:].split(',')
                for item in added_items:
                    ori_risks.append(item)
            else:
                ori_risks = [allowed_risks]
            self.allowed_risks = ori_risks
        else:
            raise ValueError("allowed_risks must be a list or a string")
        return self
    
    def unified_risk(self, unprocessed_risk:str) -> str:
        # mapping different risk labels from other datasets to the UNIFIED risk category
        return self.mapping.get(unprocessed_risk, Const.UNKNOWN)
    
    def risk_id(self, risk:str) -> int:
        if risk == Const.UNKNOWN:
            return 99
        return self.category_index.get(risk, 99) # 99 is unknown type
    
    def risk_name(self, risk_id:int) -> str:
        if risk_id == 99:
            return Const.UNKNOWN
        return self.inverse_category_index.get(risk_id, Const.UNKNOWN)



def removeR(rs: list[str]):
    return f"~{','.join(rs)}"

def addR(rs: list[str]):
    return f"+{','.join(rs)}"

class PolicyHub:
    default = Policy() # no mapping, but can handle risk category <-> id convertion
    
    mm_safetybench = Policy(DatasetKeys.MMSafetyBench)
    jailbreakv_28k = Policy(DatasetKeys.JailbreakV).allow_risks(removeR([Const.BENIGN]))
    figstep = Policy(DatasetKeys.Figstep)
    hades = Policy(DatasetKeys.Hades).allow_risks(removeR([Const.BENIGN]))
    rtvlm = Policy(DatasetKeys.RTVLM)
    siuo = Policy(DatasetKeys.SIUO)
    vlsbench = Policy(DatasetKeys.VLSBench)
    vlsafe = Policy().allow_risks(removeR([Const.BENIGN]))
    vlguard = Policy(DatasetKeys.VLGuard)
    holisafe = Policy(DatasetKeys.HoliSafe)

    safesora = Policy(DatasetKeys.SafeSora)
    video_safetybench = Policy(DatasetKeys.VideoSafetybench)
    safewatch = Policy(DatasetKeys.SafeWatch)

    aiah = Policy(DatasetKeys.AIAH)
    ajailbench = Policy(DatasetKeys.AJailBench)

    omni_safetybench = Policy(DatasetKeys.OmniSafetyBench)
    safebench = Policy(DatasetKeys.SafeBench).allow_risks(removeR([Const.BENIGN]))

    beavertails = Policy(DatasetKeys.Beavertails)
    beavertails_unsafe = Policy(DatasetKeys.Beavertails).allow_risks(removeR([Const.BENIGN]))
    forbidden_question = Policy(DatasetKeys.ForbiddenQuestions)
    simplesafetytests = Policy(DatasetKeys.SimpleSafetyTests)
    harmbench = Policy(DatasetKeys.HarmBench)
    xstest = Policy(DatasetKeys.XSTest)
    saferlhf = Policy(DatasetKeys.SafeRLHF).agreement(AgreementPolicy.HALF_AGREEMENT)
    wildguard = Policy(DatasetKeys.Wildguard).drop(RelabelPolicy.DROP_TOXIC_TO_BENIGN)
    aegis2 = Policy(DatasetKeys.Aegis2)
    toxicchat = Policy(DatasetKeys.ToxicChat)
    jbb = Policy(DatasetKeys.JailbreakBench)
