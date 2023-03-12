import numpy as np
import spacy

nlp = spacy.load("en_core_web_sm")


def Detect_descriptive_words(Sentence):
    # Input: a sentence
    # Output: a list of descriptive words in the sentence, using spacy
    doc = nlp(Sentence)
    descriptive = []
    for token in doc:
        if token.pos_ == 'ADJ' or token.pos_ == 'ADV' or token.pos_ == 'VERB':
            descriptive.append(token.text)
    # return the list of descriptive words and their indices
    # if descriptive is empty, return an empty lists
    if len(descriptive) == 0:
        return [], []
    indices = np.argwhere(np.isin(Sentence.split(' '), descriptive))[0]
    return descriptive, indices


if __name__ == '__main__':
    # get sentence from user
    sentence = "A cat with a hat is lying on a beach chair"
    # sentence = input("Enter a sentence: ")
    # get descriptive words
    descriptive_words, indices_dw = Detect_descriptive_words(sentence)
    # print the sentence and the descriptive words
    print("The sentence is: ", sentence)
    print("The descriptive words are: ", descriptive_words)
    print("The indices of the descriptive words are: ", indices_dw)

