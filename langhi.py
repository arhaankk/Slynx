import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Embedding, LSTM, Bidirectional, Dense, RepeatVector, TimeDistributed, Concatenate, Attention
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from datasets import load_dataset

# 1. Define source tokens (romanized text)
source_tokens = list('abcdefghijklmnopqrstuvwxyz ')
source_tokenizer = Tokenizer(char_level=True, filters='')
source_tokenizer.fit_on_texts(source_tokens)

# 2. Define Hindi tokens (Devanagari characters)
hindi_tokens = [
    # Independent vowels
    'अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ऋ', 'ए', 'ऐ', 'ओ', 'औ',
    # Consonants
    'क', 'ख', 'ग', 'घ', 'ङ', 'च', 'छ', 'ज', 'झ', 'ञ',
    'ट', 'ठ', 'ड', 'ढ', 'ण', 'त', 'थ', 'द', 'ध', 'न',
    'प', 'फ', 'ब', 'भ', 'म', 'य', 'र', 'ल', 'व', 'श', 'ष', 'स', 'ह',
    # Matras / vowel modifiers and other signs
    'ा', 'ि', 'ी', 'ु', 'ू', 'े', 'ै', 'ो', 'ौ', 'ं', 'ः', '्', ' '
]
target_tokenizer = Tokenizer(char_level=True, filters='')
target_tokenizer.fit_on_texts(hindi_tokens)

# 3. Load Hindi dataset from Hugging Face (romanized Hindi to native Hindi)
# (Make sure you have internet access and the dataset is available.)
from datasets import load_dataset

dataset = load_dataset(
    "csv", 
    data_files="data/datahi.tsv", 
    delimiter="\t", 
    column_names=["hi", "en"]
)

data = dataset['train']

# 4. Remove empty rows and ensure strings in both 'hi' and 'en' fields
def filter_empty_rows(example):
    return (example['hi'] is not None and example['hi'].strip() != '') and \
           (example['en'] is not None and example['en'].strip() != '')


data = data.filter(filter_empty_rows)

def ensure_strings(example):
    example['hi'] = str(example['hi']) if example['hi'] is not None else ''
    example['en'] = str(example['en']) if example['en'] is not None else ''
    return example

data = data.map(ensure_strings)

# 5. Data cleaning: lowercase and keep only defined tokens
def clean_text(example):
    example['en'] = example['en'].lower()
    cleaned_en = ''.join([char if char in source_tokens else ' ' for char in example['en']]).strip()
    cleaned_en = ' '.join(cleaned_en.split())
    example['en'] = cleaned_en

    cleaned_hi = ''.join([char if char in hindi_tokens else ' ' for char in example['hi']]).strip()
    cleaned_hi = ' '.join(cleaned_hi.split())
    example['hi'] = cleaned_hi
    return example

data = data.map(clean_text)

# 6. Determine maximum sequence length
def calc_lengths(example):
    return {'hi_length': len(example['hi']), 'en_length': len(example['en'])}

lengths = data.map(calc_lengths)
max_hi_length = max(lengths['hi_length'])
max_en_length = max(lengths['en_length'])
max_seq_length = max(max_hi_length, max_en_length)
print("Max sequence length:", max_seq_length)

# 7. Prepare input and target sequences
source_sequences = source_tokenizer.texts_to_sequences(data['en'])
X = pad_sequences(source_sequences, maxlen=max_seq_length, padding='post')

target_sequences = target_tokenizer.texts_to_sequences(data['hi'])
y = pad_sequences(target_sequences, maxlen=max_seq_length, padding='post')

# 8. Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# 9. Define the transliteration model (encoder-decoder with attention)
embedding_dim = 64
hidden_units = 128

# Encoder
encoder_input = Input(shape=(max_seq_length,), name='encoder_input')
encoder_embedding = Embedding(input_dim=len(source_tokenizer.word_index) + 1,
                              output_dim=embedding_dim,
                              name='encoder_embedding')(encoder_input)
encoder_lstm = Bidirectional(LSTM(hidden_units, return_sequences=True, name='encoder_lstm'))(encoder_embedding)
encoder_dense = Dense(hidden_units, activation='relu', name='encoder_dense')(encoder_lstm)

# Decoder: using the last time step from encoder_dense
decoder_input = RepeatVector(max_seq_length, name='decoder_repeat')(encoder_dense[:, -1, :])
decoder_lstm = LSTM(hidden_units, return_sequences=True, name='decoder_lstm')(decoder_input)

# Attention layer
attn_output = Attention(name='attention')([decoder_lstm, encoder_dense])
decoder_concat = Concatenate(name='concat')([decoder_lstm, attn_output])
decoder_output = TimeDistributed(Dense(len(target_tokenizer.word_index) + 1, activation='softmax'),
                                   name='time_distributed')(decoder_concat)

model = Model(encoder_input, decoder_output)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# # 10. Set up callbacks and train the model
# callbacks = [
#     EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
#     ModelCheckpoint('models/hindi_transliteration.h5', monitor='val_loss', save_best_only=True),
#     ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
# ]

# history = model.fit(X_train, np.expand_dims(y_train, -1),
#                     validation_data=(X_val, np.expand_dims(y_val, -1)),
#                     epochs=10,
#                     batch_size=64,
#                     callbacks=callbacks)

# 11. Save the trained model
# model.save('models/hindi_transliteration_model.keras')
model = load_model('models/hindi_transliteration_model.keras')

# 12. Define a function to transliterate input while preserving non-token characters
def transliterate_with_non_tokens(input_text, model, source_tokenizer, target_tokenizer, max_seq_length):
    # Split text into alphabetic tokens and non-alphabetic characters
    tokens_and_non_tokens = re.findall(r"([a-zA-Z]+)|([^a-zA-Z]+)", input_text)
    transliterated_text = ""
    for token, non_token in tokens_and_non_tokens:
        if token:
            seq = source_tokenizer.texts_to_sequences([token])[0]
            padded = pad_sequences([seq], maxlen=max_seq_length, padding='post')
            pred = model.predict(padded)
            pred_indices = np.argmax(pred, axis=-1)[0]
            word = ''.join([target_tokenizer.index_word.get(idx, '') for idx in pred_indices if idx != 0])
            transliterated_text += word
        elif non_token:
            transliterated_text += non_token
    return transliterated_text

# 13. Test with an example romanized Hindi input
test_text = "aur bhai kaisa hain?"
result = transliterate_with_non_tokens(test_text, model, source_tokenizer, target_tokenizer, max_seq_length)
print("Predicted Transliteration:", result)
