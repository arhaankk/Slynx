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

# --- Combine the two TSV files ---
def load_and_label(file_path, language, roman_col=1):
    # Read file with on_bad_lines='skip' to handle malformed lines
    df = pd.read_csv(file_path, delimiter="\t", header=None, names=['native', 'roman'], 
                     encoding='utf-8', dtype=str, on_bad_lines='skip')
    df['language'] = language
    return df

# Load Hindi and Malayalam data
df_hi = load_and_label("datahi.tsv", "hi")
df_ml = load_and_label("dataml.tsv", "ml")

# Use the romanized text for classification, and clean it
def clean_text(text):
    # Convert non-string values to an empty string or string representation
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    return re.sub('[^a-z ]+', '', text)


df_hi['text'] = df_hi['roman'].apply(clean_text)
df_ml['text'] = df_ml['roman'].apply(clean_text)

# Combine the datasets
data = pd.concat([df_hi[['text', 'language']], df_ml[['text', 'language']]], ignore_index=True)

# --- Proceed with the classifier as before ---

# 1. Preprocess text: already lowercased and cleaned
# 2. Tokenize at character level
tokenizer = Tokenizer(char_level=True, filters='')
tokenizer.fit_on_texts(data['text'])
sequences = tokenizer.texts_to_sequences(data['text'])
max_seq_length = max(len(seq) for seq in sequences)
X = pad_sequences(sequences, maxlen=max_seq_length, padding='post')

# 3. Encode language labels ("hi" and "ml")
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['language'])

# 4. Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# 5. Define the classifier model
embedding_dim = 32
hidden_units = 64
vocab_size = len(tokenizer.word_index) + 1

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_seq_length),
    LSTM(hidden_units),
    Dropout(0.5),
    Dense(2, activation='softmax')  # Two languages: Hindi and Malayalam
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 6. Set up callbacks and train the classifier
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ModelCheckpoint('language_classifier.h5', monitor='val_loss', save_best_only=True)
]

# model.fit(X_train, y_train, validation_data=(X_val, y_val),
#           epochs=10, batch_size=64, callbacks=callbacks)
model = load_model('language_classifier.h5')

# 7. Define a function to predict language from a new romanized input
def predict_language(text, model, tokenizer, label_encoder, max_seq_length):
    text = text.lower()
    text = re.sub('[^a-z ]+', '', text)
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_seq_length, padding='post')
    pred = model.predict(padded)
    label = label_encoder.inverse_transform([np.argmax(pred)])
    return label[0]

# 8. Test the classifier with a sample input
test_text = "ente peru raju aanu"
predicted_language = predict_language(test_text, model, tokenizer, label_encoder, max_seq_length)
print("Predicted language:", predicted_language)
