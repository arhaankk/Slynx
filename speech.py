import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf

def generate_speech(prompt, description, output_filename="output.wav"):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts").to(device)
    tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
    description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)

    description_input_ids = description_tokenizer(description, return_tensors="pt").to(device)
    prompt_input_ids = tokenizer(prompt, return_tensors="pt").to(device)

    generation = model.generate(
        input_ids=description_input_ids.input_ids, 
        attention_mask=description_input_ids.attention_mask,
        prompt_input_ids=prompt_input_ids.input_ids, 
        prompt_attention_mask=prompt_input_ids.attention_mask)

    audio_arr = generation.cpu().numpy().squeeze()
    sf.write(output_filename, audio_arr, model.config.sampling_rate)


prompt = "త ల్లని పిలీీ"
description = "Kiran's voice is clear and friendly with a moderate pace."
generate_speech(prompt, description)
# sf.write("/Users/apoorvadevarakonda/Documents/output.wav", audio_arr, model.config.sampling_rate)

