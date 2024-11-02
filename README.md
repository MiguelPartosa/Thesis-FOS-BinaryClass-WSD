# Thesis 2024-2025

Accompanying methodology of the paper on _Non-Literal Binary Classification On Cebuano Using Word Sense Disambiguation (WSD)_. The paper is presented to SU-CCS by Canlas, Jalandoni, and Partosa.

# Work Used

-   [Cebuano-POS-Tagger](https://github.com/rjrequina/Cebuano-POS-Tagger?tab=readme-ov-file) by Arjemariel Requina. We refactored the code to work for 3.0 versions of Python as well as some syntax and filename changes.
-

# TODO

-   [ ] Setup Topic Modelling for Cebuano to work
-   [ ] Import CBERT implementation 
-   [ ] Change CBERT implementation to binary classification 
-   [ ] Combine all methods into one `.ipynb` file for Methodology
-   [ ] Turn install packages into one setup.py

# POS Tagger

## Setup

-   `python -m venv .venv `
-   `cd Cebuano_POS_Tagger`

## Installation

-   `pip install -r requirements.txt`
-   `pip install cebpostagger`
-   `pip install setuptools`
-   `python setup.py install` _(This is if requirements.txt is not properly working)_

## Test if POS works

-   `cd ..`
-   `python test_tagger.py`

# Clustering

-   `pip install gensim`
-   `pip install pandas`
-   `pip install scikit-learn`
-   `pip install spacy`
-   `python -m spacy download en_core_web_sm`

# CBERT

-   `pip install torch`
