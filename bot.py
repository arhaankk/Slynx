from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
from speech import SpeechGenerator
from transliterator import HindiTransliterator
from classifier import LanguageClassifier, languages, file_paths

import requests

tts_generator = SpeechGenerator()
classifier = LanguageClassifier(file_paths, languages)
transliterator = HindiTransliterator()

classifier.load_data()
x = classifier.prepare_tokenizer()
y = classifier.encode_labels()
classifier.load_or_train_model(x,y)

voice_list = {
        "bn": "Arjun",
        "hi": "Rohit",
        "ml": "Harish",
        "te": "Prakash",
        "mr": "Sanjay"
    }

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Welcome to *Slynx* on Telegram!\nSend me a message and I'll read it for you.", parse_mode="Markdown")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    input_text = update.message.text
    print(f"User Input: {input_text}")

    transliterator.run_pipeline()
    transliterated_text = transliterator.transliterate(input_text)
    print(f"Transliterated Text: {transliterated_text}")

    lang = classifier.predict_language(transliterated_text)
    print(f"Predicted Language: {lang}")

    speaker = voice_list.get(lang)
    description = f"{speaker}'s voice is clear and friendly with a moderate pace."


    url = "http://localhost:8000/generate-audio/"
    payload = {
        "text": transliterated_text,
        "description": description,
        "voice": speaker
    }

    response = requests.post(url, json=payload)
    response.raise_for_status()

    output_filename = "output.wav"
    with open(output_filename, "wb") as f:
        f.write(response.content)

    await update.message.reply_voice(voice=open(output_filename, 'rb'))


if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    load_dotenv()

    TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    if not TELEGRAM_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN not set in environment variables")
    
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Telegram Bot is running...")
    app.run_polling()


