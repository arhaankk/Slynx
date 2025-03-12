from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

class TransliterateText:
    def transliterate_text(self, text: str, source_lang, target_script) -> str:
        return transliterate(text, source_lang, target_script)

if __name__ == "__main__": 
    transliterator = TransliterateText()
    text = "veere, ki hal hai"
    gurmukhi_text = transliterator.transliterate_text(text, sanscript.HK, sanscript.GURMUKHI)
    print(gurmukhi_text)

