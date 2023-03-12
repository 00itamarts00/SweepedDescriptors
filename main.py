import numpy as np
import spacy
import cv2

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
    # Part 1: Detect descriptive words from a sentence (user input)

    # get sentence from user
    sentence = "A cat with a hat is lying on a beach chair"
    # sentence = input("Enter a sentence: ")
    # get descriptive words
    descriptive_words, indices_dw = Detect_descriptive_words(sentence)
    # print the sentence and the descriptive words
    print("The sentence is: ", sentence)
    print("The descriptive words are: ", descriptive_words)

    # Part 2: Generate images using diffusion models
    images = []
    # TODO: create more images using diffusion models

    # Part 3: create a gif or video from the images
    height, width, layers = images[0].shape
    size = (width, height)
    frame_rate = 2
    out = cv2.VideoWriter(f'scene.avi', cv2.VideoWriter_fourcc(*'DIVX'), frame_rate, size)
    for i in range(len(images)):
        out.write(images[i])
    out.release()

