import os
import numpy as np
import pandas as pd
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Embedding, LSTM, Dense, Dropout
from keras._tf_keras.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras._tf_keras.keras.layers import Bidirectional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from utils import clean_text, load_dataset, compute_class_weights, load_existing_model

languages = ["te", "ml", "hi", "bn", "mr"]
file_paths = [
    "data/datate.tsv",
    "data/dataml.tsv",
    "data/datahi.tsv",
    "data/databn.tsv",
    "data/datamr.tsv"
]

class LanguageClassifier:
    def __init__(self, file_paths, languages, model_path='models/language_classifier.h5'):
        """
        Initializes the classifier with file paths and corresponding language labels.
        """
        self.file_paths = file_paths
        self.languages = languages
        self.model_path = model_path
        self.data = None
        self.tokenizer = None
        self.label_encoder = None
        self.max_seq_length = None
        self.model = None

    def load_and_label(self, file_path, language):
        """
        Loads a TSV file, labels it with the given language, and returns a DataFrame.
        Uses the common load_tsv function.
        """
        df = load_dataset(file_path, delimiter="\t", header=None, names=['native', 'roman'])
        df['language'] = language
        return df

    def load_data(self):
        """
        Loads data from all provided file paths, cleans the text using the shared clean_text
        function, and concatenates the datasets into one DataFrame.
        """
        dataframes = [self.load_and_label(fp, lang) for fp, lang in zip(self.file_paths, self.languages)]
        for df in dataframes:
            df['text'] = df['roman'].apply(lambda x: clean_text(x))
        self.data = pd.concat([df[['text', 'language']] for df in dataframes], ignore_index=True)
        print("Data distribution:")
        print(self.data['language'].value_counts())

    def prepare_tokenizer(self):
        """
        Fits a tokenizer on the text data and pads sequences to the maximum sequence length.
        Returns the padded sequences.
        """
        self.tokenizer = Tokenizer(char_level=False)
        self.tokenizer.fit_on_texts(self.data['text'])
        sequences = self.tokenizer.texts_to_sequences(self.data['text'])
        self.max_seq_length = max(len(seq) for seq in sequences)
        x = pad_sequences(sequences, maxlen=self.max_seq_length, padding='post')
        return x

    def encode_labels(self):
        """
        Encodes the language labels using a LabelEncoder and returns the encoded labels.
        """
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(self.data['language'])
        print("Languages found:", self.label_encoder.classes_)
        return y

    def build_model(self, embedding_dim=32, hidden_units=64):
        vocab_size = len(self.tokenizer.word_index) + 1
        num_classes = len(self.label_encoder.classes_)

        self.model = Sequential([
            Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=self.max_seq_length),
            Bidirectional(LSTM(hidden_units, return_sequences=True)),
            Dropout(0.5),
            Bidirectional(LSTM(hidden_units)),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])

        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model.build(input_shape=(None, self.max_seq_length))
        self.model.summary()

        return self.model

    def train_model(self, x, y, batch_size=64, epochs=10):
        """
        Splits the data, computes class weights using the shared compute_class_weights function,
        and trains the model with early stopping and model checkpoint callbacks.
        """
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
        weights = compute_class_weights(y)
        print("Class weights:", weights)

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
            ModelCheckpoint(self.model_path, monitor='val_loss', save_best_only=True)
        ]

        self.model.fit(x_train, y_train, validation_data=(x_val, y_val), 
                    epochs=epochs, batch_size=batch_size, callbacks=callbacks)

        # 🔽 Save model after training
        self.model.save(self.model_path)
        print(f"Trained model saved at {self.model_path}")

        return self.model


    def predict_language(self, text):
        """
        Cleans the input text, tokenizes it, pads the sequence, and predicts the language.
        """
        cleaned_text = clean_text(text)
        seq = self.tokenizer.texts_to_sequences([cleaned_text])
        padded = pad_sequences(seq, maxlen=self.max_seq_length, padding='post')
        pred = self.model.predict(padded)
        label = self.label_encoder.inverse_transform([np.argmax(pred)])
        return label[0]
    
    def load_or_train_model(self, x, y):
        """
        Loads an existing model if available, else builds and trains a new one.
        """
        if os.path.exists(self.model_path):
            print("Loading existing model from", self.model_path)
            self.model = load_existing_model(self.model_path)
        else:
            print("Model file not found. Building and training a new model.")
            self.build_model()
            self.train_model(x, y)

if __name__ == "__main__":
    classifier = LanguageClassifier(file_paths, languages)
    classifier.load_data()
    x = classifier.prepare_tokenizer()
    y = classifier.encode_labels()
    classifier.load_or_train_model(x, y)

    text = "Tu kasa aahes?"
    predicted_language = classifier.predict_language(text)
    print("Predicted language:", predicted_language)

