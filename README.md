# Thesis 2024-2025

Accompanying methodology of the paper on _Non-Literal Binary Classification On Cebuano Using Word Sense Disambiguation (WSD)_. The paper is presented to SU-CCS by Canlas, Jalandoni, and Partosa.

# Works Used

- We cloned the  [Cebuano-POS-Tagger](https://github.com/rjrequina/Cebuano-POS-Tagger?tab=readme-ov-file) by Arjemariel Requina. We refactored the code to work for 3.0 versions of Python and primarily for our implementation.
- We implemented the following packages of the cloned repository as local modules to fix errors with our google colab implementation and some legacy code.
  - [cebstemmer](https://pypi.org/project/cebstemmer/)
  - [cebpostagger](https://pypi.org/project/cebpostagger/)
  - [cebdict](https://pypi.org/project/cebdict/)


# POS Tagger

## Setup with Colab

- rename `local.env` to `.env` and add the OpenAI key
- Open the file `Thesis_Implementation_1_2.ipynb` on colab
- Run All

# Clustering normal

- `pip install sentence_transformers`

# CBERT

- `pip install torch`
