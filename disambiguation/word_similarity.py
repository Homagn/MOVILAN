import spacy

nlp = spacy.load("en_core_web_md")  # make sure to use larger model!
tokens = nlp("Bed Dresser")

for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))