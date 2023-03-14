import numpy as np
from rich.console import Console
from rich.prompt import Prompt
from torch import nn


class Prompt2Img(nn.Module):
    def __init__(self) -> None:
        self.console = Console()
        super().__init__()

    def prompt_user(self):
        prompt = f"Enter a sentence to animate the descriptors"
        sentence = Prompt.ask(f":rocket: {prompt}", default=5)
        return sentence

    def cprint(self, msg, style=None):
        self.console.print(f'{msg}', style=style)

    # def forward(self, prompt:str, descriptors: List[str]):
    #     prompts = ["A photo of a tree branch at blossom"] * 4
    #     equalizer = get_equalizer(prompts[0], word_select=("blossom",), values=(.5, .0, -.5))
    #     controller = AttentionReweight(prompts, NUM_DIFFUSION_STEPS, cross_replace_steps=1., self_replace_steps=.2, equalizer=equalizer)
    #     _ = run_and_display(prompts, controller, latent=x_t, run_baseline=False)


if __name__ == '__main__':
    img2gif = Prompt2Img()
    a = np.random.randint(low=0, high=255, size=(
        20, 224, 224, 3)).astype(np.uint8)
    img2gif.forward(
        a, destination='/Users/itamar/Git/SweepedDescriptors/test.avi')
