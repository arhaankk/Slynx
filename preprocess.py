# import pandas as pd

# def process_hindi_tsv(file_path):
#     # Read TSV file with no header; assumes two columns: native and roman
#     df = pd.read_csv(file_path, delimiter="\t", header=None, names=['hi', 'en'], encoding='utf-8')
    
#     native_sentences = []
#     roman_sentences = []
#     current_native = []
#     current_roman = []
    
#     # Group tokens until a sentence separator is found
#     for _, row in df.iterrows():
#         if row['hi'] == '</s>' and row['en'] == '</s>':
#             if current_native and current_roman:
#                 native_sentences.append(" ".join(current_native))
#                 roman_sentences.append(" ".join(current_roman))
#             current_native = []
#             current_roman = []
#         else:
#             current_native.append(row['hi'])
#             current_roman.append(row['en'])
            
#     # Append any remaining tokens as a sentence
#     if current_native and current_roman:
#         native_sentences.append(" ".join(current_native))
#         roman_sentences.append(" ".join(current_roman))
    
#     return pd.DataFrame({'hi': native_sentences, 'en': roman_sentences})

# # Example usage:
# df_sentences = process_hindi_tsv("datahi.tsv")
# print(df_sentences.head())

import pandas as pd

def process_hindi_tsv(file_path):
    # Use on_bad_lines='skip' to skip malformed lines
    df = pd.read_csv(
        file_path,
        delimiter="\t",
        header=None,
        names=['te', 'en'],
        encoding='utf-8',
        dtype=str,
        on_bad_lines='skip'  # For pandas 1.3+, or use error_bad_lines=False in older versions
    )
    
    native_sentences = []
    roman_sentences = []
    current_native = []
    current_roman = []
    
    for _, row in df.iterrows():
        if row['te'] == '</s>' and row['en'] == '</s>':
            if current_native and current_roman:
                native_sentences.append(" ".join(current_native))
                roman_sentences.append(" ".join(current_roman))
            current_native = []
            current_roman = []
        else:
            current_native.append(str(row['te']))
            current_roman.append(str(row['en']))
            
    if current_native and current_roman:
        native_sentences.append(" ".join(current_native))
        roman_sentences.append(" ".join(current_roman))
    
    return pd.DataFrame({'te': native_sentences, 'en': roman_sentences})

# Example usage:
df_sentences = process_hindi_tsv("datate.tsv")
print(df_sentences.head())
