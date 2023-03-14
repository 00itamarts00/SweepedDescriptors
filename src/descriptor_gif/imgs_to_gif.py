import cv2
import numpy as np
import spacy
from rich.console import Console
from rich.prompt import Prompt
from torch import nn
from typing import List


class Imgs2Gif(nn.Module):
    def __init__(self) -> None:
        self.console = Console()
        super().__init__()

    def prompt_user(self):
        prompt = f"Enter a sentence to animate the descriptors"
        sentence = Prompt.ask(f":rocket: {prompt}", default=5)
        return sentence
    
    def cprint(self, msg, style=None):
        self.console.print(f'{msg}', style=style)

    def forward(self, imgs:List[np.ndarray], destination=None):
        assert destination is not None, self.cprint("Please insert valid sestination for gif", style="bold red on white")
        assert len(imgs) != 0, self.cprint("Image list is empty!", style="bold red on white")
        height, width, _ = imgs[0].shape
        size = (width, height)
        frame_rate = 1
        out = cv2.VideoWriter(f'{destination}', cv2.VideoWriter_fourcc(*'DIVX'), frame_rate, size)
        for i in range(len(imgs)):
            out.write(imgs[i])
        out.release()
        self.cprint(msg=f'Succesfully saved gif to {destination}', style='green')


if __name__ == '__main__':
    img2gif = Imgs2Gif()
    a = np.random.randint(low=0, high=255, size=(20, 224, 224, 3)).astype(np.uint8)
    img2gif.forward(a, destination='/Users/itamar/Git/SweepedDescriptors/test.avi')
