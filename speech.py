import os
from TTS.api import TTS

# Ensure proper weight loading
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'

# Load the XTTSv2 model
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")

tts.tts_to_file(
    text="ఈరోజు నేను పార్కులో నడిచాను, తరువాత ఐస్ క్రీం తినడానికి బయటకు వెళ్ళాను.",
    file_path="output.wav",
    speaker="Kumar Dahl",
    language="te",
    split_sentences=True)
