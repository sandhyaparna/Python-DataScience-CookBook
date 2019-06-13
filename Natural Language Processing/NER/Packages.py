#### Install spaCy ####
## Anaconda Prompt
Pip install spacy
python -m spacy download en

## Jupyter Notebook
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()



