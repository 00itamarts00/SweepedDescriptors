import cv2
import numpy as np
import spacy
from rich.console import Console
from rich.prompt import Prompt
from torch import nn


class DescriptorDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.console = Console()
        self.nlp = spacy.load("en_core_web_sm")

    def prompt_user(self):
        prompt = f"Enter a sentence to animate the descriptors"
        sentence = Prompt.ask(f":rocket: {prompt}", default=5)
        return sentence
    
    def cprint(self, msg, style=None):
        self.console.print(f'{msg}', style=style)

    def forward(self, sentence=None):
        if sentence is None:
            sentence = self.prompt_user()
        else:
            descriptors = self.get_descriptive_words(sentence=sentence)
        if len(descriptors) ==0:
            self.console.print(f"The sentence is void of descriptive words! input a different sentence", style="bold yellow")
        else:
            self.cprint(msg=f'The descriptors are: {descriptors}', style='green')
        return descriptors

    def get_descriptive_words(self, sentence):
        doc = self.nlp(sentence)
        descriptive = []
        for token in doc:
            if token.pos_ == 'ADJ' or token.pos_ == 'ADV' or token.pos_ == 'VERB':
                descriptive.append(token.text)
        # return the list of descriptive words and their indices
        # if descriptive is empty, return an empty lists
        # indices = np.argwhere(np.isin(sentence.split(' '), descriptive))[0]
        return descriptive


if __name__ == '__main__':
    desc_man = DescriptorDetector()
    descriptors = desc_man.forward()
