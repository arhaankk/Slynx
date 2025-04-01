from speech import SpeechGenerator
from convertor import TransliterateText
from classify import LanguageDetector

import os
from indic_transliteration import sanscript

class App:
    def __init__(self):
        self.lang_detector = LanguageDetector()
        self.transliterator = TransliterateText()
        self.speech_generator = SpeechGenerator()
    
    def launch(self, text: str):
        description = "Rohit is an 18 year old boy that likes to speak in informal hindi." 
        detected_lang = self.lang_detector.detect_lang(text)
        transliterated_text = self.transliterator.transliterate_text(text, sanscript.HK, sanscript.GURMUKHI)
        self.speech_generator.generate_speech(transliterated_text, description)

if __name__ == "__main__":
    app = App()
    app.launch("aap kaise ho")