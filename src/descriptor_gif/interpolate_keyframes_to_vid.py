import os
from copy import copy
from typing import List

import cv2
import numpy as np
import torch
from frame_interpolation.ifrnet import Model
from imageio import mimsave
from rich.console import Console
from rich.prompt import Prompt
from torch import nn
from tqdm import tqdm

IFRNET_PRETRAINED_WEIGHTS = 'pretrained_models/IFRNet_GoPro.pth?dl=0'


class InterpolateKeyframes(nn.Module):
    def __init__(self, fps=30, vid_length_sec=10, method='ifrnet') -> None:
        self.console = Console()
        self.fps = fps
        self.vid_length_sec = vid_length_sec
        self.method = method
        super().__init__()

        self.model = None
        self.device = torch.device(
            'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.setup()

    def setup(self):
        if self.method == 'ifrnet' and self.model is None:
            self.model = Model().to(self.device).eval()
            self.model.load_state_dict(torch.load(
                f'./{IFRNET_PRETRAINED_WEIGHTS}', map_location=torch.device(self.device)))

    @property
    def video_frames(self):
        return self.fps * self.vid_length_sec

    def prompt_user(self):
        prompt = f"Enter a sentence to animate the descriptors"
        sentence = Prompt.ask(f":rocket: {prompt}", default=5)
        return sentence

    def cprint(self, msg, style=None):
        self.console.print(f'{msg}', style=style)

    def preprocess(self, img):
        return (torch.tensor(img.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).to(self.device)

    def postprocess(self, img):
        return (img.data.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)

    def ifrnet_interp(self, a, b, num_frames):
        prep_a = self.preprocess(a)
        prep_b = self.preprocess(b)
        embts = torch.cat([torch.tensor(i).view(1, 1, 1, 1)
                          for i in torch.linspace(0, 1, num_frames)[1:-1]], 0)
        tensor_a = torch.cat([prep_a for _ in range(num_frames-2)], 0)
        tensor_b = torch.cat([prep_b for _ in range(num_frames-2)], 0)
        imgt_pred = self.model.inference(tensor_a, tensor_b, embts)

        imgt_pred_np = np.array([self.postprocess(img) for img in imgt_pred])

        return imgt_pred_np

    def forward_ifrnet(self, keyframes):
        # keyframes should be uint8
        vid = []
        num_frames = (self.video_frames - 2) // (len(keyframes) - 1)
        self.cprint(f'Interpolating the {len(keyframes)} keyframes')
        for a, b in tqdm(zip(keyframes[:-1], keyframes[1:]), total=len(keyframes)-1):
            vid.append(a)
            interp_imgs = self.ifrnet_interp(a=a, b=b, num_frames=num_frames)
            vid.extend([i for i in interp_imgs])

        vid.append(b)
        return vid

    def classic_blend_inter(self, a, b, num_frames):
        alpha = np.linspace(0, 1, num_frames)[1:]
        beta = alpha[::-1]
        arr = np.array([np.uint8(alpha_i * a + beta_i) *
                       b for alpha_i, beta_i in zip(alpha, beta)])
        return arr

    def forward_classic(self, keyframes):
        vid = []
        num_frames = (self.video_frames - 2) // (len(keyframes) - 1)
        self.cprint(f'Interpolating the {len(keyframes)} keyframes')
        for a, b in tqdm(zip(keyframes[:-1], keyframes[1:]), total=len(keyframes)-1):
            vid.append(a)
            interp_imgs = self.classic_blend_inter(
                a=a, b=b, num_frames=num_frames)
            vid.extend([i for i in interp_imgs])

        vid.append(b)
        return vid

    def forward(self, keyframes):
        if self.method == 'ifrnet':
            return self.forward_ifrnet(keyframes=keyframes)
        if self.method == 'classic_blending':
            return self.forward_classic(keyframes=keyframes)


if __name__ == '__main__':
    interpolator = InterpolateKeyframes()
    keyframes = np.random.randint(
        low=0, high=255, size=(5, 224, 224, 3)).astype(np.uint8)
    keyframes = [i for i in keyframes]

    vid = interpolator.forward_classic(keyframes)

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video_writer = cv2.VideoWriter(
        'test_classic.mp4', fourcc, float(interpolator.fps), (224, 224))

    for frame in vid:
        video_writer.write(frame)

    video_writer.release()
