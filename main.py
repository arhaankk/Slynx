import os
from speech import SpeechGenerator
from transliterator import HindiTransliterator, TeluguTransliterator
from classifier import LanguageClassifier, languages, file_paths

from utils import load_existing_model

def main():
    input_text = "tu bahut acha admi hai yaar!"  
    description = "Rohit's voice is clear and friendly with a moderate pace."
    output_filename = "output.wav"

   

    # Step 1: Transliteration.
    transliterator = HindiTransliterator()
    transliterator.run_pipeline()
    transliterated_text = transliterator.transliterate(input_text)
    print("Transliterated text:", transliterated_text)

    # Step 2: Classification.
    classifier = LanguageClassifier(file_paths, languages)
    classifier.model = load_existing_model(classifier.model_path)
    classifier.load_data()
    x = classifier.prepare_tokenizer()
    y = classifier.encode_labels()

    if os.path.exists(classifier.model_path):
        print("Loading existing model from", classifier.model_path)
        classifier.model = load_existing_model(classifier.model_path)
    else:
        print("Model file not found. Building and training a new model.")
        classifier.build_model()
        classifier.train_model(x, y)
    classification_result = classifier.predict_language(transliterated_text)
    print("Classification result:", classification_result)

    # Step 3: Speech synthesis.
    # We combine the original description with the classification result.
    combined_description = f"{description} Classified as: {classification_result}"
    tts_generator = SpeechGenerator()
    tts_generator.generate_speech(
        prompt=transliterated_text,
        description=combined_description,
        output_filename=output_filename
    )
    print(f"Audio generated and saved as {output_filename}")

if __name__ == "__main__":
    main()
