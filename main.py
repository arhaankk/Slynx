import os
from speech import SpeechGenerator
from transliterator import HindiTransliterator, TeluguTransliterator
from classifier import LanguageClassifier, languages, file_paths



def main():

    voice_list = {
        "bn": "Arjun",
        "hi": "Rohit",
        "ml": "Harish",
        "te": "Prakash",
        "mr": "Sanjay"
    }

    transliterator_list = {
        "hi": HindiTransliterator(),
        "te": TeluguTransliterator()
    }

    input_text = "Indha naala weather romba nalla irukku."  
    
    output_filename = "output.wav"

    # Step 1: Classification
    try:
        classifier = LanguageClassifier(file_paths, languages)
        classifier.load_data()
        x = classifier.prepare_tokenizer()
        y = classifier.encode_labels()

        classifier.load_or_train_model(x, y)

        classification_result = classifier.predict_language(input_text)
        print("Classification result:", classification_result)
    except:
        print("Language not supported yet")
        


    # Step 2: Transliteration.
    try:
        transliterator = transliterator_list.get(classification_result)
        transliterator.run_pipeline()
        transliterated_text = transliterator.transliterate(input_text)
        print("Transliterated text:", transliterated_text)
    except:
        print("No available transliterator. Please check classification.")

        

    
    # Step 3: Speech synthesis.
    try:
        speaker = voice_list.get(classification_result)
        description = f"{speaker}'s voice is clear and friendly with a moderate pace."
        print(f"SPEAKER IS {speaker}")
        combined_description = f"{description} Classified as: {classification_result}"
        tts_generator = SpeechGenerator()
        tts_generator.generate_speech(
            prompt=transliterated_text,
            description=combined_description,
            output_filename=output_filename
        )
        print(f"Audio generated and saved as {output_filename}")
    except:
        print("No available TTS. Please check classification.")

if __name__ == "__main__":
    main()
