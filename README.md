# NER Chatbot — NLP & Machine Learning

A Streamlit-based educational chatbot for extracting person names, predefined locations, and day/month references from text. It combines an intent classifier trained with TF-IDF and Multinomial Naive Bayes with NLTK POS tagging and rule-based entity extraction.

## Features
- Interactive dark-themed chat interface and clear-chat control
- Person recognition using proper-noun POS tags
- Location matching against a predefined list
- Day and month matching
- Intent-specific responses for queries beginning with a request such as “Find locations in: ...”
- Locally stored trained intent model and TF-IDF vectorizer

## Stack
Python, Streamlit, NLTK, scikit-learn, pickle.

## Setup

```bash
python -m venv .venv
```

Activate on Windows CMD:

```cmd
.venv\Scripts\activate
```

Install dependencies and NLTK resources:

```bash
python -m pip install -r requirements.txt
python -m nltk.downloader punkt punkt_tab stopwords averaged_perceptron_tagger_eng
python -m streamlit run app.py
```

Open the local URL shown in your terminal (typically http://localhost:8501).

## Example

Input: `Ali went to Lahore on Monday`

Expected entity categories: Person (`Ali`), Location (`Lahore`), Date (`Monday`). Results depend on NLTK tagging and exact case-sensitive list matching.

## Files
- `app.py`: Streamlit application (based on the original `new_app.py`)
- `intent_model.pkl`, `vectorizer.pkl`: existing serialized model artifacts
- `notebooks/NLP_Project.ipynb`: training and evaluation notebook

## Model and evaluation notes
The training notebook uses CoNLL-2003 sentences to generate four types of intent requests (person, location, organization, date), and trains a Multinomial Naive Bayes classifier. The project report records **65.12% intent-classification accuracy**. This is an original experimental result, **not a leakage-free benchmark**: the notebook fits TF-IDF before the train/test split. The current app also uses explicit text checks and does not use the predicted intent for every input.

## Limitations
- Organization extraction is not implemented.
- Location detection is limited to a predefined, case-sensitive list; dates cover selected days and months.
- POS-based person extraction may misclassify proper nouns.
- The app is an educational prototype, not a general-purpose pretrained NER model.
- Pickle model files should only be loaded from trusted sources.

## Data
The original project used CoNLL-2003 data. Dataset files are not redistributed in this repository; obtain them separately from an authorized source if retraining.

## Future work
Implement organization detection, improve evaluation with a train-only vectorizer fit, and consider spaCy or transformer-based NER.
