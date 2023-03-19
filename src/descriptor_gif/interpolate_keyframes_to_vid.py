from typing import List

import cv2
import numpy as np
from rich.console import Console
from rich.prompt import Prompt
from torch import nn


class InterpolateKeyframes(nn.Module):
    def __init__(self, video_frames=300, method='firnet') -> None:
        self.console = Console()
        self.video_frames = video_frames
        self.method = method
        super().__init__()        

    def prompt_user(self):
        prompt = f"Enter a sentence to animate the descriptors"
        sentence = Prompt.ask(f":rocket: {prompt}", default=5)
        return sentence

    def cprint(self, msg, style=None):
        self.console.print(f'{msg}', style=style)

    def forward_firnet(self, keyframes: List[np.ndarray]):
        self.cprint(f'Interpolating the {len(keyframes)} keyframes')
        
        
        


if __name__ == '__main__':
    interpolator = InterpolateKeyframes()
    a = np.random.randint(low=0, high=255, size=(224, 224, 3)).astype(np.uint8)
    b = np.random.randint(low=0, high=255, size=(224, 224, 3)).astype(np.uint8)

    imgs = interpolator.forward_firnet([a, b])
