import re
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
# --- Load and label each TSV file ---
def load_and_label(file_path, language):
    # Read file with on_bad_lines='skip' to handle malformed lines
    df = pd.read_csv(file_path, delimiter="\t", header=None, names=['native', 'roman'], 
                     encoding='utf-8', dtype=str, on_bad_lines='skip')
    df['language'] = language
    return df

# Load Hindi, Marathi, and Telugu data
df_hi = load_and_label("datahi.tsv", "hi")
df_mr = load_and_label("datamr.tsv", "mr")
df_te = load_and_label("datate.tsv", "te")  # Assuming a TSV for Telugu
df_ml = load_and_label("dataml.tsv", "ml")
df_bn = load_and_label("databn.tsv", "bn")
# --- Clean the romanized text ---
def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    # Remove unwanted characters but keep letters, numbers, and spaces.
    text = re.sub(r'[^a-z0-9\s]+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df_hi['text'] = df_hi['roman'].apply(clean_text)
df_mr['text'] = df_mr['roman'].apply(clean_text)
df_te['text'] = df_te['roman'].apply(clean_text)
df_ml['text'] = df_ml['roman'].apply(clean_text)
df_bn['text'] = df_bn['roman'].apply(clean_text)
# --- Combine the datasets ---
data = pd.concat([df_hi[['text', 'language']], 
                  df_mr[['text', 'language']], 
                  df_te[['text', 'language']],
                  df_ml[['text', 'language']],
                  df_bn[['text', 'language']]],
                 ignore_index=True)

# print("Combined data sample:")
# print(data.head())

# --- Tokenize at character level ---
tokenizer = Tokenizer(char_level=False)
tokenizer.fit_on_texts(data['text'])

sequences = tokenizer.texts_to_sequences(data['text'])
max_seq_length = max(len(seq) for seq in sequences)
X = pad_sequences(sequences, maxlen=max_seq_length, padding='post')

# --- Encode language labels ---
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['language'])
print("Languages found:", label_encoder.classes_)

# --- Split the data ---
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# --- Define the classifier model ---
embedding_dim = 32
hidden_units = 64
vocab_size = len(tokenizer.word_index) + 1
num_classes = len(label_encoder.classes_)

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_shape=(max_seq_length,)),
    LSTM(hidden_units),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --- Set up callbacks and train the classifier ---
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ModelCheckpoint('language_classifier.h5', monitor='val_loss', save_best_only=True)
]

# model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=64, callbacks=callbacks)
model = load_model('language_classifier.h5')

# --- Define a function to predict language from a new romanized input ---
def predict_language(text, model, tokenizer, label_encoder, max_seq_length):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]+', '', text)
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_seq_length, padding='post')
    pred = model.predict(padded)
    label = label_encoder.inverse_transform([np.argmax(pred)])
    return label[0]

# --- Test the classifier with a sample input ---
test_text = "enda pere raju"
predicted_language = predict_language(test_text, model, tokenizer, label_encoder, max_seq_length)
print("Predicted language:", predicted_language)
