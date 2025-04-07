import os
import re
import numpy as np
import pandas as pd
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from keras._tf_keras.keras.models import Model, load_model
from keras._tf_keras.keras.layers import Input, Embedding, LSTM, Bidirectional, Dense, RepeatVector, TimeDistributed, Concatenate, Attention, Lambda
from keras._tf_keras.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from datasets import Dataset
from utils import clean_text, load_existing_model

class BaseTransliterator:
    def __init__(self, data_file, model_path, source_tokens, target_tokens):
        """
        Generic transliterator class.
        
        Args:
            data_file (str): Path to the TSV file with target data.
            model_path (str): Path where the model is saved/loaded.
            source_tokens (list): List of valid source (romanized) tokens.
            target_tokens (list): List of valid target (native) tokens.
        """
        self.data_file = data_file
        self.model_path = model_path
        self.source_tokens = source_tokens
        self.target_tokens = target_tokens

        # Initialize tokenizers
        self.source_tokenizer = Tokenizer(char_level=True, filters='')
        self.source_tokenizer.fit_on_texts(source_tokens)
        self.target_tokenizer = Tokenizer(char_level=True, filters='')
        self.target_tokenizer.fit_on_texts(target_tokens)

        self.data = None
        self.max_seq_length = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        self.model = None

    def load_dataset(self):
        """
        Loads the dataset using pandas and converts it to a Hugging Face Dataset.
        Assumes a TSV file with two columns: target (native) and source (romanized).
        Skips malformed lines.
        """
        df = pd.read_csv(
            self.data_file,
            delimiter="\t",
            header=None,
            names=["target", "source"],
            on_bad_lines="skip",
            engine="python"
        )
        self.data = Dataset.from_pandas(df)
        return self.data

    def preprocess_data(self):
        """
        Runs the full preprocessing pipeline:
          1. Loads the dataset.
          2. Filters out empty rows.
          3. Ensures fields are strings.
          4. Cleans both source and target text using language-specific allowed tokens.
        """
        self.load_dataset()
        self.data = self.data.filter(lambda ex: ex['target'] and ex['source'] and ex['target'].strip() and ex['source'].strip())
        self.data = self.data.map(lambda ex: {"target": str(ex["target"]), "source": str(ex["source"])})
        self.data = self.data.map(lambda ex: {
            "target": clean_text(ex["target"], allowed_chars=self.target_tokens, lower=False),
            "source": clean_text(ex["source"], allowed_chars=self.source_tokens, lower=True)
        })
        return self.data

    def determine_max_seq_length(self):
        """
        Determines the maximum sequence length among source and target texts.
        """
        def calc_lengths(example):
            return {'target_length': len(example['target']), 'source_length': len(example['source'])}
        lengths = self.data.map(calc_lengths)
        max_target = max(lengths['target_length'])
        max_source = max(lengths['source_length'])
        self.max_seq_length = max(max_target, max_source)
        print("Max sequence length:", self.max_seq_length)
        return self.max_seq_length

    def prepare_sequences(self):
        """
        Converts texts to sequences using tokenizers and pads them.
        """
        source_sequences = self.source_tokenizer.texts_to_sequences(self.data['source'])
        self.X = pad_sequences(source_sequences, maxlen=self.max_seq_length, padding='post')

        target_sequences = self.target_tokenizer.texts_to_sequences(self.data['target'])
        self.y = pad_sequences(target_sequences, maxlen=self.max_seq_length, padding='post')
        return self.X, self.y

    def train_val_split(self, test_size=0.1, random_state=42):
        """
        Splits the dataset into training and validation sets.
        """
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        return self.X_train, self.X_val, self.y_train, self.y_val

    def build_model(self, embedding_dim=64, hidden_units=128):
        """
        Builds an encoder-decoder model with attention.
        """
        # Encoder.
        encoder_input = Input(shape=(self.max_seq_length,), name='encoder_input')
        encoder_embedding = Embedding(input_dim=len(self.source_tokenizer.word_index)+1,
                                      output_dim=embedding_dim,
                                      name='encoder_embedding')(encoder_input)
        encoder_lstm = Bidirectional(LSTM(hidden_units, return_sequences=True, name='encoder_lstm'))(encoder_embedding)
        encoder_dense = Dense(hidden_units, activation='relu', name='encoder_dense')(encoder_lstm)
        
        # Extract the last time step using a Lambda layer.
        get_last_step = Lambda(lambda x: x[:, -1, :], name='get_last_step')
        last_encoder_output = get_last_step(encoder_dense)
        
        # Decoder.
        decoder_input = RepeatVector(self.max_seq_length, name='decoder_repeat')(last_encoder_output)
        decoder_lstm = LSTM(hidden_units, return_sequences=True, name='decoder_lstm')(decoder_input)
        
        # Attention mechanism.
        attn_output = Attention(name='attention')([decoder_lstm, encoder_dense])
        decoder_concat = Concatenate(name='concat')([decoder_lstm, attn_output])
        decoder_output = TimeDistributed(
            Dense(len(self.target_tokenizer.word_index)+1, activation='softmax'),
            name='time_distributed'
        )(decoder_concat)
        
        self.model = Model(encoder_input, decoder_output)
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model.summary()
        return self.model

    def train_model(self, epochs=10, batch_size=64):
        """
        Trains the model using early stopping, model checkpointing, and learning rate reduction.
        """
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ModelCheckpoint(self.model_path, monitor='val_loss', save_best_only=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
        ]
        self.model.fit(
            self.X_train, np.expand_dims(self.y_train, -1),
            validation_data=(self.X_val, np.expand_dims(self.y_val, -1)),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        return self.model

    def load_existing_model(self):
        """
        Loads an existing model using the shared function.
        """
        self.model = load_existing_model(self.model_path)
        return self.model

    def transliterate(self, input_text):
        """
        Transliterates an input string while preserving non-token characters.
        """
        tokens_and_non_tokens = re.findall(r"([a-zA-Z]+)|([^a-zA-Z]+)", input_text)
        transliterated_text = ""
        for token, non_token in tokens_and_non_tokens:
            if token:
                seq = self.source_tokenizer.texts_to_sequences([token])[0]
                padded = pad_sequences([seq], maxlen=self.max_seq_length, padding='post')
                pred = self.model.predict(padded)
                pred_indices = np.argmax(pred, axis=-1)[0]
                word = ''.join([self.target_tokenizer.index_word.get(idx, '')
                                for idx in pred_indices if idx != 0])
                transliterated_text += word
            elif non_token:
                transliterated_text += non_token
        return transliterated_text

    def run_pipeline(self, train_new_model=False, epochs=10, batch_size=64):
        """
        Executes the full pipeline: preprocess, prepare sequences, split data, and build/train (or load) the model.
        """
        self.preprocess_data()
        self.determine_max_seq_length()
        self.prepare_sequences()
        self.train_val_split()
        if train_new_model or not os.path.exists(self.model_path):
            self.build_model()
            self.train_model(epochs=epochs, batch_size=batch_size)
        else:
            self.load_existing_model()

# Language-specific subclasses:

class HindiTransliterator(BaseTransliterator):
    def __init__(self, data_file="data/datahi.tsv", model_path="models/hindi_transliteration_model.keras"):
        source_tokens = list('abcdefghijklmnopqrstuvwxyz ')
        hindi_tokens = [
            'अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ऋ', 'ए', 'ऐ', 'ओ', 'औ',
            'क', 'ख', 'ग', 'घ', 'ङ', 'च', 'छ', 'ज', 'झ', 'ञ',
            'ट', 'ठ', 'ड', 'ढ', 'ण', 'त', 'थ', 'द', 'ध', 'न',
            'प', 'फ', 'ब', 'भ', 'म', 'य', 'र', 'ल', 'व', 'श', 'ष', 'स', 'ह',
            'ा', 'ि', 'ी', 'ु', 'ू', 'े', 'ै', 'ो', 'ौ', 'ं', 'ः', '्', ' '
        ]
        super().__init__(data_file, model_path, source_tokens, hindi_tokens)

class TeluguTransliterator(BaseTransliterator):
    def __init__(self, data_file="data/datate.tsv", model_path="models/telugu_transliteration_model.keras"):
        source_tokens = list('abcdefghijklmnopqrstuvwxyz ')
        telugu_tokens = [
            'అ', 'ఆ', 'ఇ', 'ఈ', 'ఉ', 'ఊ', 'ఋ', 'ఎ', 'ఏ', 'ఐ', 'ఒ', 'ఓ', 'ఔ',
            'క', 'ఖ', 'గ', 'ఘ', 'ఙ',
            'చ', 'ఛ', 'జ', 'ఝ', 'ఞ',
            'ట', 'ఠ', 'డ', 'ఢ', 'ణ',
            'త', 'థ', 'ద', 'ధ', 'న',
            'ప', 'ఫ', 'బ', 'భ', 'మ',
            'య', 'ర', 'ల', 'వ',
            'శ', 'ష', 'స', 'హ',
            'ా', 'ి', 'ీ', 'ు', 'ూ', 'ృ', 'ే', 'ై', 'ొ', 'ో', 'ౌ', 'ం', 'ః', '్', ' '
        ]
        super().__init__(data_file, model_path, source_tokens, telugu_tokens)

# Example usage:
if __name__ == "__main__":
    # For Hindi
    hindi_transliterator = HindiTransliterator()
    hindi_transliterator.run_pipeline(train_new_model=False, epochs=10, batch_size=64)
    test_text_hindi = "aur bhai kaisa hain?"
    print("Hindi Predicted Transliteration:", hindi_transliterator.transliterate(test_text_hindi))
    
    # For Telugu
    telugu_transliterator = TeluguTransliterator()
    telugu_transliterator.run_pipeline(train_new_model=False, epochs=10, batch_size=64)
    test_text_telugu = "tellani pilly"
    print("Telugu Predicted Transliteration:", telugu_transliterator.transliterate(test_text_telugu))