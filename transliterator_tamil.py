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

# 1. Define source tokens (romanized text)
source_tokens = list('abcdefghijklmnopqrstuvwxyz ')
source_tokenizer = Tokenizer(char_level=True, filters='')
source_tokenizer.fit_on_texts(source_tokens)

# 2. Define Telugu tokens (Devanagari characters)
telugu_tokens = [
    'అ', 'ఆ', 'ఇ', 'ఈ', 'ఉ', 'ఊ', 'ఋ', 'ఎ', 'ఏ', 'ఐ', 'ఒ', 'ఓ', 'ఔ',
    'క', 'ఖ', 'గ', 'ఘ', 'ఙ',
    'చ', 'ఛ', 'జ', 'ఝ', 'ఞ',
    'ట', 'ఠ', 'డ', 'ఢ', 'ణ',
    'త', 'థ', 'ద', 'ధ', 'న',
    'ప', 'ఫ', 'బ', 'భ', 'మ',
    'య', 'ర', 'ల', 'వ',
    'శ', 'ష', 'స', 'హ',
    # Matras
    'ా', 'ి', 'ీ', 'ు', 'ూ', 'ృ', 'ే', 'ై', 'ొ', 'ో', 'ౌ', 'ం', 'ః', '్', ' '
]
target_tokenizer = Tokenizer(char_level=True, filters='')
target_tokenizer.fit_on_texts(telugu_tokens)

# 3. Load the TSV file using pandas
data = pd.read_csv("data/datate.tsv", delimiter="\t", header=None, names=["te", "en"], 
                   encoding="utf-8", dtype=str, on_bad_lines='skip')

# 4. Remove empty rows and ensure strings in both 'te' and 'en' fields
data = data.dropna(subset=['te', 'en'])
data = data[(data['te'].str.strip() != '') & (data['en'].str.strip() != '')]

# 5. Data cleaning: lowercase and keep only defined tokens
def clean_text(text, valid_tokens):
    text = text.lower()
    cleaned = ''.join([char if char in valid_tokens else ' ' for char in text]).strip()
    cleaned = ' '.join(cleaned.split())
    return cleaned

data['en'] = data['en'].apply(lambda x: clean_text(x, source_tokens))
data['te'] = data['te'].apply(lambda x: clean_text(x, telugu_tokens))

# 6. Determine maximum sequence length
max_en_length = data['en'].apply(len).max()
max_te_length = data['te'].apply(len).max()
max_seq_length = max(max_en_length, max_te_length)
print("Max sequence length:", max_seq_length)

# 7. Prepare input and target sequences
source_sequences = source_tokenizer.texts_to_sequences(data['en'])
X = pad_sequences(source_sequences, maxlen=max_seq_length, padding='post')

target_sequences = target_tokenizer.texts_to_sequences(data['te'])
y = pad_sequences(target_sequences, maxlen=max_seq_length, padding='post')

# 8. Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# 9. Define the transliteration model (encoder-decoder with attention)
embedding_dim = 64
hidden_units = 128

encoder_input = Input(shape=(max_seq_length,), name='encoder_input')
encoder_embedding = Embedding(input_dim=len(source_tokenizer.word_index) + 1,
                              output_dim=embedding_dim,
                              name='encoder_embedding')(encoder_input)
encoder_lstm = Bidirectional(LSTM(hidden_units, return_sequences=True, name='encoder_lstm'))(encoder_embedding)
encoder_dense = Dense(hidden_units, activation='relu', name='encoder_dense')(encoder_lstm)

decoder_input = RepeatVector(int(max_seq_length), name='decoder_repeat')(encoder_dense[:, -1, :])
decoder_lstm = LSTM(hidden_units, return_sequences=True, name='decoder_lstm')(decoder_input)

attn_output = Attention(name='attention')([decoder_lstm, encoder_dense])
decoder_concat = Concatenate(name='concat')([decoder_lstm, attn_output])
decoder_output = TimeDistributed(Dense(len(target_tokenizer.word_index) + 1, activation='softmax'),
                                   name='time_distributed')(decoder_concat)

model = Model(encoder_input, decoder_output)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 10. Set up callbacks and train the model
# callbacks = [
#     EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
#     ModelCheckpoint('models/telugu_transliteration.h5', monitor='val_loss', save_best_only=True),
#     ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
# ]

# history = model.fit(X_train, np.expand_dims(y_train, -1),
#                     validation_data=(X_val, np.expand_dims(y_val, -1)),
#                     epochs=10,
#                     batch_size=64,
#                     callbacks=callbacks)

# 11. Save the trained model
#model.save('models/telugu_transliteration_model.keras')
model = load_model('models/telugu_transliteration_model.keras')

# 12. Define a function to transliterate input while preserving non-token characters
def transliterate_with_non_tokens(input_text, model, source_tokenizer, target_tokenizer, max_seq_length):
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

# 13. Test with an example romanized input
test_text = "tellani pilly"
result = transliterate_with_non_tokens(test_text, model, source_tokenizer, target_tokenizer, max_seq_length)
print("Predicted Transliteration:", result)
