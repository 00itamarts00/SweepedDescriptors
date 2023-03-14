import abc
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as nnf
from diffusers import DiffusionPipeline
from IPython.display import display
from PIL import Image
from prompt_to_prompt import ptp_functional, ptp_utils, seq_aligner
from rich.console import Console
from rich.prompt import Prompt
from torch import nn


class Prompt2Img(nn.Module):
    def __init__(self,
                 model_id="CompVis/ldm-text2im-large-256",
                 num_diffusion_steps=50,
                 guidance_scale=5.,
                 max_num_workds=77,
                 image_size = [256, 256],
                 ) -> None:

        self.model_id = model_id
        self.num_diffusion_steps = num_diffusion_steps
        self.guidance_scale = guidance_scale
        self.max_num_workds = max_num_workds
        self.image_size = image_size
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
        
    def get_baseline_latent_init(self, prompts):
        height, width = self.image_size
        latent, latents = ptp_utils.init_latent(latent=None, model=self.ldm,
                                      height=height, width=width, generator=None, batch_size=len(prompts))
        return latent, latents

    def init_latent(self, prompt):
        ptp_utils.init_latent(latent=self.ldm.v, model, height, width, generator, batch_size)
        images, x_t = ptp_functional.run_and_display(prompts, controller, run_baseline=False, generator=g_cpu)


    def forward(self, prompt: str, descriptor: List[str]):
        # get baseline latent using the first prompt
        # determine length of clip
        # multiply the number of prompts by the length of the clip
        # init equilaizer with the description chosen
        # init the controller
        # generate images using ptp_utils.text2image_ldm
        # update the latent
        
        prompts = [prompt] * len(descriptor)
        latent, latents = self.get_baseline_latent_init(prompt)
        for descriptor in descriptor:
            equalizer = ptp_functional.get_equalizer(prompts[0], word_select=(descriptor), values=(.5, .0, -.5))
            controller = ptp_functional.AttentionReweight(prompts, self.num_diffusion_steps, cross_replace_steps=1., self_replace_steps=.2, equalizer=equalizer)
            _ = ptp_functional.run_and_display(prompts, controller, latent=latent, run_baseline=False)



if __name__ == '__main__':
    p2gif = Prompt2Img()
    prompt = "The cake is red"
    p2gif.forward(prompt, destination='/Users/itamar/Git/SweepedDescriptors/test.avi')
