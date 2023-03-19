
import os

import torch
from descriptor_detection import DescriptorDetector
from imgs_to_mp4 import Imgs2Vid
from interpolate_keyframes_to_vid import InterpolateKeyframes
from prompt_to_image import Prompt2Img
from rich.console import Console
from rich.prompt import Prompt
from torch import nn


class Prompt2VID(nn.Module):
    def __init__(self,
                 num_keyframes=4
                 ) -> None:
        super().__init__()
        self.num_keyframes = num_keyframes
        self.setup()

    def setup(self):
        self.device = torch.device(
            'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.console = Console()

        self.descriptor_detection = DescriptorDetector()
        self.interp_keyframes = InterpolateKeyframes()
        self.prompt2imgs = Prompt2Img()
        self.imgs2vid = Imgs2Vid()

    def prompt_user(self):
        prompt = f"Enter a sentence to animate the descriptors"
        sentence = Prompt.ask(f":rocket: {prompt}", default=5)
        return sentence

    def cprint(self, msg, style=None):
        self.console.print(f'{msg}', style=style)

    def forward(self, prompt: str):
        descriptors = self.descriptor_detection(sentence=prompt)
        self.cprint(f'Using descriptor {descriptors[0]} for Prompt to Video')
        descriptor = descriptors[0]
        keyframes = self.prompt2imgs(
            prompt, descriptor=descriptor, clip_len=self.num_keyframes)
        video = self.interp_keyframes(keyframes)
        self.imgs2vid(video, destination=os.getcwd())


if __name__ == '__main__':
    p2vid = Prompt2VID()
    prompt = "A photo of a fluffy doll"
    imgs = p2vid.forward(prompt)
