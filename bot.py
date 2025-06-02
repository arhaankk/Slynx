from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters

from speech import SpeechGenerator
from transliterator import HindiTransliterator
from classifier import LanguageClassifier, languages, file_paths

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
    