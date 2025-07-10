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

    input_text = "mera naam Rahul hai aur mujhe khaana acha lagta hai"  
    
    output_filename = "output.wav"

    # Step 1: Transliteration.
    transliterator = HindiTransliterator()
    transliterator.run_pipeline()
    transliterated_text = transliterator.transliterate(input_text)
    print("Transliterated text:", transliterated_text)

    # Step 2: Classification.
    classifier = LanguageClassifier(file_paths, languages)
    classifier.load_data()
    x = classifier.prepare_tokenizer()
    y = classifier.encode_labels()


    classifier.load_or_train_model(x, y)

    classification_result = classifier.predict_language(input_text)
    print("Classification result:", classification_result)


    speaker = voice_list.get(classification_result)
    
    description = f"{speaker}'s voice is clear and friendly with a moderate pace."
    print(f"SPEAKER IS {speaker}")
    # Step 3: Speech synthesis.
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
