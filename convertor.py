from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

def transliterate_text(text, source_lang, target_script):
    return transliterate(text, source_lang, target_script)


text = "veere, ki hal hai"
source_language = sanscript.HK  
target_script = sanscript.GURMUKHI  


gurmukhi_text = transliterate_text(text, source_language, target_script)
print(gurmukhi_text)
