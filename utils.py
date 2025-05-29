import os
import re
import numpy as np
import pandas as pd
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from sklearn.utils import class_weight
from keras._tf_keras.keras.models import load_model

def clean_text(text, allowed_chars=None, lower=True):
    """
    Cleans the input text.
    
    Parameters:
      text (str): The text to be cleaned.
      allowed_chars (iterable, optional): If provided, only characters in this set are retained.
          All other characters are replaced with a space. If None, a default pattern (a-z, 0-9, whitespace) is used.
      lower (bool): Whether to convert text to lowercase.
    
    Returns:
      str: The cleaned text.
    """
    if not isinstance(text, str):
        text = str(text)
    if lower:
        text = text.lower()
    if allowed_chars is not None:
        text = ''.join([char if char in allowed_chars else ' ' for char in text])
    else:
        text = re.sub(r'[^a-z0-9\s]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_dataset(file_path, delimiter="\t", header=None, names=None, encoding="utf-8", dtype=str, on_bad_lines='skip'):
    """
    Loads a TSV (or CSV) file into a pandas DataFrame.
    
    Parameters:
      file_path (str): Path to the file.
      delimiter (str): The delimiter used in the file.
      header (int or None): Row to use as header (None means no header).
      names (list, optional): List of column names to use.
      encoding (str): File encoding.
      dtype: Data type to force.
      on_bad_lines: Behavior when encountering malformed lines.
    
    Returns:
      pd.DataFrame: The loaded DataFrame.
    """
    return pd.read_csv(file_path, delimiter=delimiter, header=header, names=names,
                       encoding=encoding, dtype=dtype, on_bad_lines=on_bad_lines)

def create_tokenizer(token_list=None, char_level=False, filters=''):
    """
    Creates and returns a Keras Tokenizer.
    
    Parameters:
      token_list (list, optional): A list of tokens to fit the tokenizer on.
      char_level (bool): Whether to use character-level tokenization.
      filters (str): Characters to filter out.
    
    Returns:
      Tokenizer: A fitted Keras tokenizer.
    """
    tokenizer = Tokenizer(char_level=char_level, filters=filters)
    if token_list is not None:
        tokenizer.fit_on_texts(token_list)
    return tokenizer

def pad_text_sequences(tokenizer, texts, max_seq_length=None, padding='post'):
    """
    Converts texts to sequences using a tokenizer and pads them.
    
    Parameters:
      tokenizer (Tokenizer): A fitted Keras tokenizer.
      texts (list): List of text strings.
      max_seq_length (int, optional): Maximum sequence length. If not provided, 
          the maximum length among the sequences is used.
      padding (str): Padding type ('post' or 'pre').
    
    Returns:
      tuple: A tuple containing the padded sequences (as a NumPy array) and the maximum sequence length used.
    """
    sequences = tokenizer.texts_to_sequences(texts)
    if max_seq_length is None:
        max_seq_length = max(len(seq) for seq in sequences)
    padded = pad_sequences(sequences, maxlen=max_seq_length, padding=padding)
    return padded, max_seq_length

def compute_class_weights(y):
    """
    Computes class weights for imbalanced class distributions.
    
    Parameters:
      y (array-like): Array of class labels.
    
    Returns:
      dict: A mapping from class indices to weights.
    """
    weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
    return dict(enumerate(weights))

def split_tokens_and_non_tokens(input_text):
    """
    Splits the input text into alphabetic tokens and non-alphabetic characters.
    
    Parameters:
      input_text (str): The text to split.
    
    Returns:
      list: A list of tuples, where each tuple contains an alphabetic token or a non-alphabetic substring.
    """
    return re.findall(r"([a-zA-Z]+)|([^a-zA-Z]+)", input_text)

def load_existing_model(model_path):
    """
    Loads an existing Keras model from disk.
    
    Parameters:
      model_path (str): Path to the model file.
    
    Returns:
      Model: The loaded Keras model.
    
    Raises:
      FileNotFoundError: If the model file does not exist.
    """
    if os.path.exists(model_path):
        return load_model(model_path)
    else:
        raise FileNotFoundError(f"Model file {model_path} not found.")
