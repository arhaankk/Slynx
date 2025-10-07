from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from speech import SpeechGenerator
from transliterator import HindiTransliterator, TeluguTransliterator
from classifier import LanguageClassifier, languages, file_paths

app = FastAPI()

origins = [
    "*",
]

app.add_middleware(  
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    text: str

@app.get("/", status_code=200)
async def root():
    return{"message": "Welcome to Slynx"}

@app.post("/generate-audio", status_code=200)
async def generate_audio(request: TextRequest):
    input_text = request.text
    description = "Rohit's voice is clear and friendly with a moderate pace."
    output_filename = "output.wav"

    transliterator = HindiTransliterator()
    transliterator.run_pipeline()
    transliterated_text = transliterator.transliterate(input_text)

    classifier = LanguageClassifier(file_paths, languages)
    classifier.load_data()
    x = classifier.prepare_tokenizer()
    y = classifier.encode_labels()

    classifier.load_or_train_model(x, y)

    classification_result = classifier.predict_language(transliterated_text)

    combined_description = f"{description} Classified as: {classification_result}"
    tts_generator = SpeechGenerator()
    tts_generator.generate_speech(
        prompt=transliterated_text,
        description=combined_description,
        output_filename=output_filename
    )

    return FileResponse(
        path=output_filename,
        media_type="audio/wav",
        filename="audio_output.wav"
    )