import pandas as pd

class LanguageProcessor:
    def __init__(self, file_path, native_column='bn', roman_column='en', delimiter="\t", encoding="utf-8"):
        """
        Initializes the processor with the TSV file path and configuration.
        
        Args:
            file_path (str): Path to the TSV file.
            native_column (str): Column name for the native language text.
            roman_column (str): Column name for the romanized text.
            delimiter (str): Delimiter used in the TSV file.
            encoding (str): Encoding of the TSV file.
        """
        self.file_path = file_path
        self.native_column = native_column
        self.roman_column = roman_column
        self.delimiter = delimiter
        self.encoding = encoding
        self.dataframe = None

    def read_tsv(self):
        """
        Reads the TSV file into a DataFrame while skipping malformed lines.
        
        Returns:
            pd.DataFrame: The raw DataFrame from the TSV file.
        """
        self.dataframe = pd.read_csv(
            self.file_path,
            delimiter=self.delimiter,
            header=None,
            names=[self.native_column, self.roman_column],
            encoding=self.encoding,
            dtype=str,
            on_bad_lines='skip'
        )
        return self.dataframe

    def process_rows(self):
        """
        Processes the DataFrame rows by grouping text between sentence delimiters.
        Rows containing '</s>' in both columns indicate sentence boundaries.
        
        Returns:
            pd.DataFrame: A DataFrame with grouped sentences for native and romanized text.
        """
        if self.dataframe is None:
            self.read_tsv()
        
        native_sentences = []
        roman_sentences = []
        current_native = []
        current_roman = []

        for _, row in self.dataframe.iterrows():
            if row[self.native_column] == '</s>' and row[self.roman_column] == '</s>':
                if current_native and current_roman:
                    native_sentences.append(" ".join(current_native))
                    roman_sentences.append(" ".join(current_roman))
                current_native = []
                current_roman = []
            else:
                current_native.append(str(row[self.native_column]))
                current_roman.append(str(row[self.roman_column]))

        # Check for any remaining sentences not terminated by the delimiter
        if current_native and current_roman:
            native_sentences.append(" ".join(current_native))
            roman_sentences.append(" ".join(current_roman))

        return pd.DataFrame({
            self.native_column: native_sentences,
            self.roman_column: roman_sentences
        })

    def process(self):
        """
        The main method to process the TSV file and return the final DataFrame.
        
        Returns:
            pd.DataFrame: Processed DataFrame with grouped sentences.
        """
        self.read_tsv()
        return self.process_rows()

# Example usage:
if __name__ == "__main__":
    processor = LanguageProcessor("databn.tsv")
    df_sentences = processor.process()
    print(df_sentences.head())
