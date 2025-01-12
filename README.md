# Thesis 2024-2025

Accompanying methodology of the paper on _Non-Literal Binary Classification On Cebuano Using Word Sense Disambiguation (WSD)_. The paper is presented to SU-CCS by Canlas, Jalandoni, and Partosa.

# Work Used

- [Cebuano-POS-Tagger](https://github.com/rjrequina/Cebuano-POS-Tagger?tab=readme-ov-file) by Arjemariel Requina. We refactored the code to work for 3.0 versions of Python as well as some syntax and filename changes.
- Stopwords removal from cebstemmer?

# TODO

- [ ] Setup Topic Modelling for Cebuano to work
- [ ] Import CBERT implementation
- [ ] Change CBERT implementation to binary classification
- [ ] Combine all methods into one `.ipynb` file for Methodology
- [ ] Turn install packages into one setup.py

# POS Tagger

## Setup

- Currently working only for [the Google Colab notebook](https://colab.research.google.com/drive/1_hcScQRSiFGuKV3w9ku55G4sd7on07CJ?usp=sharing)
- `cd Cebuano_POS_Tagger`

## Installation

- Skip to test if using the Colab
- `pip install -r requirements.txt`
- `pip install setuptools`
- `python setup.py install` _(This is if requirements.txt is not properly working)_

## Test if POS works

- Main directory for running files is in the current folder for now since packages aren't exposed
- `python test_tagger.py`

# **(EXPERIMENTAL)** Clustering

- `pip install gensim`
- `pip install pandas`
- `pip install scikit-learn`
- `pip install spacy`
- `python -m spacy download en_core_web_sm`

# Clustering normal

- `pip install sentence_transformers`

# CBERT

- `pip install torch`
