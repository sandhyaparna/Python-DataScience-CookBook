# Package is stored in C- WIndows - System32 - Valx-master
# C:\Windows\System32\Valx-master

import sys 
!{sys.executable} -m pip install nltk
# Trail ID: NCT00297583
# NCT00211536
* ClinicalNote = "- Her vital signs: heart rate of 66, blood pressure 120/63, respiratory rate 14, 100% on 5 liters nasal cannula O2 saturation# -  Significant endogenous insulin secretion indicated by fasting C-peptide# -  The value of HbA1c is <= 10.0 %# - glucose test is 99.8"
ClinicalNote
* ClinicalNote_preprocessing = preprocessing(ClinicalNote)
ClinicalNote_preprocessing
* ClinicalNote_split_text_inclusion_exclusion = split_text_inclusion_exclusion(ClinicalNote_preprocessing)
ClinicalNote_split_text_inclusion_exclusion
* ClinicalNote_extract_candidates_numeric = extract_candidates_numeric(ClinicalNote_preprocessing)
ClinicalNote_extract_candidates_numeric
* name_list=""
ClinicalNote_extract_candidates_name = extract_candidates_name(ClinicalNote_extract_candidates_numeric[0],ClinicalNote_extract_candidates_numeric[1],name_list)
ClinicalNote_extract_candidates_name
* ClinicalNote_formalize_expressions = formalize_expressions(ClinicalNote_extract_candidates_name[1][1])
ClinicalNote_formalize_expressions
• feature_dict_dk = {}
fea_dict_umls = {}
ClinicalNote_identify_variable = identify_variable(ClinicalNote_preprocessing, feature_dict_dk, fea_dict_umls)
ClinicalNote_identify_variable






