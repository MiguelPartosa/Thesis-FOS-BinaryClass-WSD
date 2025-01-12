from cebpostagger.tagger import tag_sentence
# import nltk
# nltk.download('punkt_tab')


print(tag_sentence('Ang bata naligo sa sapa.'))
print(tag_sentence('Ganahan kog prutas.'))

# [('Ang', 'DET'), ('bata', 'NOUN'), ('naligo', 'VERB'),
#  ('sa', 'PART'), ('sapa', 'NOUN'), ('.', 'SYM')]
