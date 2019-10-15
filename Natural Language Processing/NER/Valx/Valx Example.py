# Package is stored in C- WIndows - System32 - Valx-master
# C:\Windows\System32\Valx-master

import sys 
!{sys.executable} -m pip install nltk

### After making changes to the codes - To import all the functions 
from W_utility.log import * 
from W_utility.file import * 

from NLP.porter2 import *
from NLP.word import *
from NLP.sentence import *
from NLP.sentence_keywords import * 

import Valx_core
from Valx_core import *

Valx_core.init_features ()
name_list=""
name_list="heart rate | temperature | blood pressure"

fea_dict_dk = ufile.read_csv_as_dict ('data/variable_features_dk.csv', 0, 1, True)
fea_dict_umls = ufile.read_csv_as_dict ('data/variable_features_umls.csv', 0, 1, True)

add_stopwords =["finding","of"]

### Valx package as it is
ClinicalNote_preprocessing = Valx_core.preprocessing(ClinicalNote)
ClinicalNote_split_text_inclusion_exclusion = Valx_core.split_text_inclusion_exclusion(ClinicalNote_preprocessing)
ClinicalNote_extract_candidates_numeric = Valx_core.extract_candidates_numeric(ClinicalNote_preprocessing)
ClinicalNote_formalize_expressions = Valx_core.formalize_expressions(ClinicalNote_extract_candidates_numeric[1][0])
ClinicalNote_identify_variable = Valx_core.identify_variable(ClinicalNote_formalize_expressions, fea_dict_dk, fea_dict_umls)
ClinicalNote_associate_variable_values = Valx_core.associate_variable_values(ClinicalNote_identify_variable[0])
ClinicalNote_associate_variable_values

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
• fea_dict_dk = ufile.read_csv_as_dict ('data\variable_features_dk.csv', 0, 1, True)
fea_dict_umls = ufile.read_csv_as_dict ('data\variable_features_umls.csv', 0, 1, True)
ClinicalNote_identify_variable = identify_variable(ClinicalNote_preprocessing, fea_dict_dk, fea_dict_umls)
ClinicalNote_identify_variable
* ClinicalNote_associate_variable_values = Valx_core.associate_variable_values(ClinicalNote_identify_variable[0])
ClinicalNote_associate_variable_values


##### Running the whole thing as a function ######
# Create a function for NumericAttributeExtraction
from W_utility.log import * 
from W_utility.file import * 

from NLP.porter2 import *
from NLP.word import *
from NLP.sentence import *
from NLP.sentence_keywords import * 
import Valx_core
from Valx_core import *
Valx_core.init_features ()

fea_dict_dk = ufile.read_csv_as_dict ('data/variable_features_dk.csv', 0, 1, True)
fea_dict_umls = ufile.read_csv_as_dict ('data/variable_features_umls.csv', 0, 1, True)
   
def NumericAttributeExtraction(ClinicalNote):
    ClinicalNote_preprocessing = Valx_core.preprocessing(ClinicalNote)
    ClinicalNote_split_text_inclusion_exclusion = Valx_core.split_text_inclusion_exclusion(ClinicalNote_preprocessing)
    ClinicalNote_extract_candidates_numeric = Valx_core.extract_candidates_numeric(ClinicalNote_preprocessing)
    ClinicalNote_formalize_expressions = Valx_core.formalize_expressions(ClinicalNote_extract_candidates_numeric[1][0])
    ClinicalNote_identify_variable = Valx_core.identify_variable(ClinicalNote_formalize_expressions, fea_dict_dk, fea_dict_umls)
    ClinicalNote_associate_variable_values = Valx_core.associate_variable_values(ClinicalNote_identify_variable[0])
    return(ClinicalNote_associate_variable_values)
## End

# Function by adding name_list
def NumericAttributeExtraction(ClinicalNote):
    ClinicalNote_preprocessing = Valx_core.preprocessing(ClinicalNote)
    ClinicalNote_split_text_inclusion_exclusion = Valx_core.split_text_inclusion_exclusion(ClinicalNote_preprocessing)
    
    name_list="heart rate | temperature | blood pressure | wbc"
    sections_num = Valx_core.extract_candidates_numeric(ClinicalNote_preprocessing)[0]
    candidates_num = Valx_core.extract_candidates_numeric(ClinicalNote_preprocessing)[1]
    ClinicalNote_extract_candidates_name = Valx_core.extract_candidates_name(sections_num, candidates_num, name_list)
    
    ClinicalNote_formalize_expressions = Valx_core.formalize_expressions(ClinicalNote_extract_candidates_name[1][0])
    ClinicalNote_identify_variable = Valx_core.identify_variable(ClinicalNote_formalize_expressions, fea_dict_dk, fea_dict_umls)
    ClinicalNote_associate_variable_values = Valx_core.associate_variable_values(ClinicalNote_identify_variable[0])
    return(ClinicalNote_associate_variable_values)
# End
  
# Function for running all sentences within a text 
def NumericAttributeExtraction(ClinicalNote):
    try:
        ClinicalNote_preprocessing = Valx_core.preprocessing(ClinicalNote)
        ClinicalNote_split_text_inclusion_exclusion = Valx_core.split_text_inclusion_exclusion(ClinicalNote_preprocessing)
        ClinicalNote_extract_candidates_numeric = Valx_core.extract_candidates_numeric(ClinicalNote_preprocessing)

        ClinicalNote_Numerical_Attributes = []
        for i in range(len(ClinicalNote_extract_candidates_numeric[1])):
            ClinicalNote_formalize_expressions = Valx_core.formalize_expressions(ClinicalNote_extract_candidates_numeric[1][i])
            ClinicalNote_identify_variable = Valx_core.identify_variable(ClinicalNote_formalize_expressions, fea_dict_dk, fea_dict_umls)
            ClinicalNote_associate_variable_values = Valx_core.associate_variable_values(ClinicalNote_identify_variable[0])
            ClinicalNote_Numerical_Attributes = ClinicalNote_Numerical_Attributes + list(ClinicalNote_associate_variable_values)[1]
        return(ClinicalNote_Numerical_Attributes)
    except Exception:
        pass


  
  
  
  
  
  
