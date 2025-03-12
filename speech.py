import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf

class SpeechGenerator:
    def __init__(self, model_name="ai4bharat/indic-parler-tts", device=None):
        self.device = device if device else ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.description_tokenizer = AutoTokenizer.from_pretrained(self.model.config.text_encoder._name_or_path)

    def generate_speech(self, prompt, description, output_filename="output.wav"):
        description_input_ids = self.description_tokenizer(description, return_tensors="pt").to(self.device)
        prompt_input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        generation = self.model.generate(
            input_ids=description_input_ids.input_ids, 
            attention_mask=description_input_ids.attention_mask,
            prompt_input_ids=prompt_input_ids.input_ids, 
            prompt_attention_mask=prompt_input_ids.attention_mask
        )

        audio_arr = generation.cpu().numpy().squeeze()
        sf.write(output_filename, audio_arr, self.model.config.sampling_rate)

if __name__ == "__main__":
    prompt = "यार, तू क्या टाइम पास कर रहा है? थोड़ा काम भी कर ले, नहीं तो फिर समझ ले, मजे नहीं आएंगे!"
    description = "Rohit is an 18 year old boy that likes to speak in informal hindi."
    speech_generator = SpeechGenerator()
    speech_generator.generate_speech(prompt, description)
