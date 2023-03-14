import abc
from typing import List

import numpy as np
import torch
import torch.nn.functional as nnf
from diffusers import DiffusionPipeline
from IPython.display import display
from PIL import Image
from prompt_to_prompt import ptp_utils, seq_aligner
from rich.console import Console
from rich.prompt import Prompt
from torch import nn


class Prompt2Img(nn.Module):
    def __init__(self,
                 model_id="CompVis/ldm-text2im-large-256",
                 num_diffusion_steps=50,
                 guidance_scale=5.,
                 max_num_workds=77,
                 ) -> None:

        self.model_id = model_id
        self.num_diffusion_steps = num_diffusion_steps
        self.guidance_scale = guidance_scale
        self.max_num_workds = max_num_workds
        self.setup()

        super().__init__()

    def setup(self):
        self.ldm = DiffusionPipeline.from_pretrained(
            self.model_id).to(self.device)
        self.tokenizer = self.ldm.tokenizer
        self.device = torch.device(
            'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.console = Console()

    def prompt_user(self):
        prompt = f"Enter a sentence to animate the descriptors"
        sentence = Prompt.ask(f":rocket: {prompt}", default=5)
        return sentence

    def cprint(self, msg, style=None):
        self.console.print(f'{msg}', style=style)

    def forward(self, prompt: str, descriptors: List[str]):
        pass
    #     prompts = ["A photo of a tree branch at blossom"] * 4
    #     equalizer = get_equalizer(prompts[0], word_select=("blossom",), values=(.5, .0, -.5))
    #     controller = AttentionReweight(prompts, NUM_DIFFUSION_STEPS, cross_replace_steps=1., self_replace_steps=.2, equalizer=equalizer)
    #     _ = run_and_display(prompts, controller, latent=x_t, run_baseline=False)


if __name__ == '__main__':
    p2gif = Prompt2Img()
    prompt = "The cake is red"
    p2gif.forward(prompt, destination='/Users/itamar/Git/SweepedDescriptors/test.avi')
