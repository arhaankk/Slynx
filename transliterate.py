from indic_transliteration.sanscript import transliterate
from indic_transliteration import sanscript
import re

def clean_halant(text):
    return re.sub(r'्(?=\s|$)', '', text)

# Better ITRANS-like input with explicit vowels:
text = "kyuu bhaai? mujhe paani nahi chahiye. maine ek baar bola."
output = transliterate(text, sanscript.ITRANS, sanscript.DEVANAGARI)
cleaned_output = clean_halant(output)
print(cleaned_output)
