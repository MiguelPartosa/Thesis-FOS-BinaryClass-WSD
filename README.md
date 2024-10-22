# Thesis 2024-2025

Accompanying methodology of the paper on _Non-Literal Binary Classification On Cebuano Using Word Sense Disambiguation (WSD)_. The paper is presented to SU-CCS by Canlas, Jalandoni, and Partosa.

# Work Used

-   [Cebuano-POS-Tagger](https://github.com/rjrequina/Cebuano-POS-Tagger?tab=readme-ov-file) by Arjemariel Requina. We refactored the code to work for 3.0 versions of Python as well as some syntax and filename changes.
-

# POS Tagger

## Setup

-   `python -m venv .venv `
-   `cd Cebuano_POS_Tagger`

## Installation

-   `pip install -r requirements.txt`
-   `pip install cebpostagger`
-   `python setup.py install`

## Test if POS works

-   `cd ..`
-   `python test_tagger.py`
