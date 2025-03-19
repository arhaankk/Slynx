import pandas as pd

def process_language_tsv(file_path):
    # Use on_bad_lines='skip' to skip malformed lines
    df = pd.read_csv(
        file_path,
        delimiter="\t",
        header=None,
        names=['bn', 'en'],
        encoding='utf-8',
        dtype=str,
        on_bad_lines='skip'  # For pandas 1.3+, or use error_bad_lines=False in older versions
    )
    
    native_sentences = []
    roman_sentences = []
    current_native = []
    current_roman = []
    
    for _, row in df.iterrows():
        if row['bn'] == '</s>' and row['en'] == '</s>':
            if current_native and current_roman:
                native_sentences.append(" ".join(current_native))
                roman_sentences.append(" ".join(current_roman))
            current_native = []
            current_roman = []
        else:
            current_native.append(str(row['bn']))
            current_roman.append(str(row['en']))
            
    if current_native and current_roman:
        native_sentences.append(" ".join(current_native))
        roman_sentences.append(" ".join(current_roman))
    
    return pd.DataFrame({'bn': native_sentences, 'en': roman_sentences})

# Example usage:
df_sentences = process_language_tsv("databn.tsv")
print(df_sentences.head())
