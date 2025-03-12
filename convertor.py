from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

class TransliterateText:
    def __init__(self, source_lang, target_script):
        self.source_lang = source_lang
        self.target_script = target_script

    def transliterate_text(self, text: str) -> str:
        return transliterate(text, self.source_lang, self.target_script)

if __name__ == "__main__":
    source_language = sanscript.HK  
    target_script = sanscript.GURMUKHI  
    transliterator = TransliterateText(source_language, target_script)
    text = "veere, ki hal hai"
    gurmukhi_text = transliterator.transliterate_text(text)
    print(gurmukhi_text)

