import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf

class SpeechGenerator:
    def __init__(self, model_name="ai4bharat/indic-parler-tts"):
        # Select device: use GPU if available, otherwise CPU
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # Load the model and move it to the appropriate device
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(model_name).to(self.device)
        
        # Load tokenizers for both the prompt and description
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.description_tokenizer = AutoTokenizer.from_pretrained(
            self.model.config.text_encoder._name_or_path
        )
        
        # Retrieve the model's sampling rate for audio generation
        self.sampling_rate = self.model.config.sampling_rate

    def generate_speech(self, prompt, description, output_filename="audio/output.wav"):
        # Tokenize the description and prompt, then move tensors to the device
        description_inputs = self.description_tokenizer(description, return_tensors="pt").to(self.device)
        prompt_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate speech using the model
        generation = self.model.generate(
            input_ids=description_inputs.input_ids, 
            attention_mask=description_inputs.attention_mask,
            prompt_input_ids=prompt_inputs.input_ids, 
            prompt_attention_mask=prompt_inputs.attention_mask
        )

        # Convert the generated tensor to a numpy array and remove extra dimensions
        audio_arr = generation.cpu().numpy().squeeze()

        # Write the audio to a WAV file using the model's sampling rate
        sf.write(output_filename, audio_arr, self.sampling_rate)
        print(f"Audio saved to {output_filename}")

if __name__ == "__main__":

    prompt = "త ల్లని పిలీీ"
    description = "Kiran's voice is clear and friendly with a moderate pace."
    
    tts_generator = SpeechGenerator()
    tts_generator.generate_speech(prompt, description)
