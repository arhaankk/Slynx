import pytest

from classifier import LanguageClassifier, file_paths, languages

@pytest.fixture(scope='module')
def classifier():
    classifier = LanguageClassifier(file_paths, languages)
    classifier.load_data()
    x = classifier.prepare_tokenizer()
    y = classifier.encode_labels()
    classifier.load_or_train_model(x, y)

    return classifier

# hindi script, no code-switching
def test_prediction_hindi(classifier):
    text = "mera naam Rahul hai aur mujhe khaana acha lagta hai"
    prediction_result = classifier.predict_language(text)
    assert prediction_result == "hi", f"Expected 'hi', got {prediction_result}"

