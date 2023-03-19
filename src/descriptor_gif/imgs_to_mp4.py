from typing import List

import cv2
import numpy as np
import spacy
from rich.console import Console
from rich.prompt import Prompt
from torch import nn


class Imgs2Vid(nn.Module):
    def __init__(self, fps=30) -> None:
        self.console = Console()
        self.fps = fps
        super().__init__()

    def prompt_user(self):
        prompt = f"Enter a sentence to animate the descriptors"
        sentence = Prompt.ask(f":rocket: {prompt}", default=5)
        return sentence

    def cprint(self, msg, style=None):
        self.console.print(f'{msg}', style=style)
        
    def write_video(self, file_path, frames, fps):
        """
        Writes frames to an mp4 video file
        :param file_path: Path to output video, must end with .mp4
        :param frames: List of PIL.Image objects
        :param fps: Desired frame rate
        """

        w, h, c = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        writer = cv2.VideoWriter(file_path, fourcc, fps, (w, h))

        for frame in frames:
            writer.write(frame)

        writer.release() 

    def forward(self, vid: List[np.ndarray], destination=None):
        self.cprint(msg=f'Converting images to MP4 video')
        assert destination is not None, self.cprint("Please insert valid sestination for gif", style="bold red on white")
        assert len(vid) != 0, self.cprint("Image list is empty!", style="bold red on white")
        self.write_video(file_path=destination, frames=np.array(vid), fps=self.fps)
        
        self.cprint(msg=f'Done Converting images to MP4 video!!')


if __name__ == '__main__':
    img2vid = Imgs2Vid()
    a = np.random.randint(low=0, high=255, size=(
        20, 224, 224, 3)).astype(np.uint8)
    img2vid.forward(a, destination='test.avi')
