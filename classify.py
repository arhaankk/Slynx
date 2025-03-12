import os
from dotenv import load_dotenv
import google.generativeai as genai

class LanguageDetector:
    def __init__(self):
        load_dotenv()
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    def detect_lang(self, text: str) -> str:
        prompt = (
            """ 
            You are a model for detecting languages. For the given text, you will output just one word - the language it is written in. 
            Try your very best to get it as correct as possible. If you don't know the answer, reply "False"
            """
            + text
        )
        
        response = self.model.generate_content(prompt)
        return response.text


if __name__ == "__main__":
    detector = LanguageDetector()
    language = detector.detect_lang("aaj kal tere mere")
    print(language)
