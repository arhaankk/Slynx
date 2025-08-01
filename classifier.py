import os
import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification, create_optimizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from utils import clean_text, load_dataset, compute_class_weights

languages = ["te", "hi", "bn"]
file_paths = [
    "data/datate.tsv",
    "data/datahi.tsv",
    "data/databn.tsv",
]

class LanguageClassifier:
    def __init__(self, file_paths, languages, model_path='models/language_classifier'):
        self.file_paths = file_paths
        self.languages = languages
        self.model_path = model_path
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.label_encoder = LabelEncoder()
        self.model = None
        self.data = None

    def load_data(self):
        dfs = []
        for path, lang in zip(self.file_paths, self.languages):
            df = load_dataset(path, delimiter="\t", header=None, names=['native', 'roman'])
            df['text'] = df['roman'].apply(clean_text)
            df['language'] = lang
            dfs.append(df[['text', 'language']])
        self.data = pd.concat(dfs, ignore_index=True)
        print("Data distribution:\n", self.data['language'].value_counts())

    def encode_labels(self):
        y = self.label_encoder.fit_transform(self.data['language'])
        print("Languages found:", self.label_encoder.classes_)
        return y

    def tokenize(self, texts):
        return self.tokenizer(
            texts.tolist(),
            padding=True,
            truncation=True,
            return_tensors='tf'
        )

    def build_model(self, num_classes):
        self.model = TFBertForSequenceClassification.from_pretrained(
            'bert-base-multilingual-cased',
            num_labels=num_classes
        )
        return self.model

    def train_model(self, x_texts, y, batch_size=32, epochs=3):
        x_train, x_val, y_train, y_val = train_test_split(x_texts, y, test_size=0.2, random_state=42)
        class_weights = compute_class_weights(y)
        print("Class weights:", class_weights)

        train_encodings = self.tokenize(x_train)
        val_encodings = self.tokenize(x_val)

        y_train = np.array(y_train)
        y_val = np.array(y_val)

        train_dataset = tf.data.Dataset.from_tensor_slices((
            {
                "input_ids": train_encodings["input_ids"],
                "attention_mask": train_encodings["attention_mask"]
            },
            y_train
        )).shuffle(1000).batch(batch_size)

        val_dataset = tf.data.Dataset.from_tensor_slices((
            {
                "input_ids": val_encodings["input_ids"],
                "attention_mask": val_encodings["attention_mask"]
            },
            y_val
        )).batch(batch_size)

        steps_per_epoch = len(x_train) // batch_size
        optimizer, schedule = create_optimizer(
            init_lr=2e-5,
            num_warmup_steps=0,
            num_train_steps=steps_per_epoch * epochs
        )

        self.model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        self.model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)

        self.model.save_pretrained(self.model_path)
        self.tokenizer.save_pretrained(self.model_path)
        print(f"Model saved to {self.model_path}")

    def predict_language(self, text):
        cleaned = clean_text(text)
        encoding = self.tokenizer([cleaned], padding=True, truncation=True, return_tensors='tf')
        logits = self.model(encoding).logits
        pred = np.argmax(logits.numpy(), axis=1)
        return self.label_encoder.inverse_transform(pred)[0]

    def load_or_train_model(self, x_texts, y):
        if os.path.exists(self.model_path):
            print("Loading model from:", self.model_path)
            self.model = TFBertForSequenceClassification.from_pretrained(self.model_path)
            self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        else:
            print("Training new model...")
            self.build_model(num_classes=len(set(y)))
            self.train_model(x_texts, y)

if __name__ == "__main__":
    classifier = LanguageClassifier(file_paths, languages)
    classifier.load_data()
    x = classifier.data['text']
    y = classifier.encode_labels()
    classifier.load_or_train_model(x, y)

    text = "Tumi kothay? Amar sathe kotha bolte aso."
    predicted_language = classifier.predict_language(text)
    print("Predicted language:", predicted_language)
