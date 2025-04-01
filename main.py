import os
import re
import io
import uuid
import torch
import soundfile as sf
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences




app = FastAPI()

# Directory to store temporary audio files
TEMP_AUDIO_DIR = "temp_audio"
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

# ----- Placeholder functions for language detection and translation -----
def detect_language(roman_text: str) -> str:
    model = load_model('language_classifier.keras')
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    max_seq_length = 100  # ensure this matches your training setting
    cleaned_text = re.sub(r'[^a-z0-9\s]+', '', roman_text.lower())
    seq = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(seq, maxlen=max_seq_length, padding='post')
    pred = model.predict(padded)
    label = label_encoder.inverse_transform([np.argmax(pred)])
    return label[0]
    
    
    # return "te" if "te" in roman_text.lower() else "ml"

def translate_text(roman_text: str, language: str) -> str:
    """
    Dummy translation from romanized text to native text.
    Replace this with your actual translation model or logic.
    """
    if language == "te":
        
        return "త ల్లని పిలీీ"  # native Telugu text example
    elif language == "ml":
        return "പൊന്നാനി"       # native Malayalam text example
    else:
        return roman_text

# ----- Load TTS Model and Tokenizers -----
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# tts_model = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts").to(device)
# tts_tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
# description_tokenizer = AutoTokenizer.from_pretrained(tts_model.config.text_encoder._name_or_path)

def generate_speech(prompt: str, description: str) -> bytes:
    """
    Generate speech audio (WAV) from a native text prompt and a description.
    Returns the WAV audio as bytes.
    """
    # Tokenize the description and prompt
    desc_inputs = description_tokenizer(description, return_tensors="pt").to(device)
    prompt_inputs = tts_tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate speech using the TTS model
    generation = tts_model.generate(
        input_ids=desc_inputs.input_ids,
        attention_mask=desc_inputs.attention_mask,
        prompt_input_ids=prompt_inputs.input_ids,
        prompt_attention_mask=prompt_inputs.attention_mask
    )
    
    # Convert the output tensor to a numpy array and write as WAV into a BytesIO buffer
    audio_arr = generation.cpu().numpy().squeeze()
    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, audio_arr, tts_model.config.sampling_rate, format="WAV")
    wav_buffer.seek(0)
    return wav_buffer.read()

# ----- Pydantic model for the processing request -----
class ProcessRequest(BaseModel):
    roman_text: str  # The romanized text from the website

# Default description to use for TTS (since the user doesn't supply one)
DEFAULT_DESCRIPTION = "A clear, natural voice speaking at a moderate pace."

# ----- Endpoint 1: Process input and generate WAV file -----
@app.post("/process")
def process_text(request: ProcessRequest):
    try:
        # 1. Detect language from the romanized text.
        language = detect_language(request.roman_text)
        
        # 2. Translate the romanized text to native text.
        native_text = translate_text(request.roman_text, language)
        
        # 3. Generate speech using the native text and a default description.
        wav_bytes = generate_speech(native_text, DEFAULT_DESCRIPTION)
        
        # 4. Save the WAV bytes to a file with a unique ID.
        file_id = f"{uuid.uuid4()}.wav"
        file_path = os.path.join(TEMP_AUDIO_DIR, file_id)
        with open(file_path, "wb") as f:
            f.write(wav_bytes)
        
        # Return the file ID and metadata so the website can fetch the audio.
        return {"file_id": file_id, "detected_language": language, "native_text": native_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----- Endpoint 2: Retrieve the generated WAV file -----
@app.get("/get_audio/{file_id}")
def get_audio(file_id: str):
    file_path = os.path.join(TEMP_AUDIO_DIR, file_id)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type="audio/wav", filename=file_id)

# To run the API:
# uvicorn your_filename:app --host 0.0.0.0 --port 8000
print(detect_language("Sukhamaano?"))