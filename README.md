# Slynx


# 🎙️ Slynx – Romanized Text to Native Speech

**Slynx** is an end-to-end speech pipeline that converts **romanized language text** into **native-script speech audio**. It is designed to assist **screen reader users**, individuals with **reading difficulties**, and anyone interacting with languages in romanized form.

---

## 💡 What It Does

1. **Language Detection** – Identifies the correct Indian language from text
2. **Transliteration** – Converts romanized text to native script (e.g., "mera naam Rahul hai" → "मेरा नाम राहुल है")
3. **Speech Synthesis** –generates natural-sounding speech

---

## 🧩 Features

- 📝 Romanized input support
- 🌐 Auto language detection (Hindi, Telugu, etc.)
- 🔊 Native speech output (e.g., WAV file)
- ⚙️ Modular and extendable classes
- ✅ Useful for assistive tools and screen readers

---

## 🔽 Download & Use Our Models

You can **download** or **directly use** our pretrained models for:

- 🔤 **Language Detection** (e.g., Hindi, Telugu, Malayalam, Bengali and Marathi)
- 🔁 **Transliteration** from romanized text to native script (Hindi and Telugu)

---

## 🚀 Getting Started

### Prerequisites
-  Python 3.12


### To Run The Pipeline
```bash
# Step 1: Set up and activate virtual environment
python3.12 -m venv venv
source venv/bin/activate

# Step 3: Install dependencies
 pip install -r requirements.txt
```

### Project Structure
```bash
├── main.py                 # Entry point for the pipeline
├── classifier.py           # Language detection logic
├── transliterator.py       # Base + per-language transliteration classes
├── models/                 # Pretrained models
└── data/                   # TSV input data for training 
```

### 🛠️ Tech Stack

| Category            | Tools/Libraries / Sources                                                        |
|---------------------|----------------------------------------------------------------------------------|
| Programming Language| Python 3.12                                                                      |
| Deep Learning       | TensorFlow, Keras, PyTorch                                                       |
| NLP & ML            | Hugging Face Transformers, scikit-learn                                          |
| Data Handling       | pandas, datasets                                                                 |
| Speech Synthesis    | [Parler-TTS](https://github.com/huggingface/parler-tts)                          |
| Model Types         | Embedding, Encoding-Decoding (Seq2Seq), Bidirectional LSTM, Attention, Dense     |
| Dataset             | [Dakshina Dataset](https://github.com/google-research-datasets/dakshina)         |
| Data Format         | TSV                                                                              |


### Extend Support (To add new languages)

### Documentation

[Slynx-Project Proposal](/Slynx-Project_Proposal%20(1).pdf)

### 📄 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

**Note:** This project is for demonstration and educational purposes only. All rights to dataset and model usage belong to their respective owners.




