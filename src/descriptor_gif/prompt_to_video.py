import os

import torch
from descriptor_detection import DescriptorDetector
from imgs_to_mp4 import Imgs2Vid
from interpolate_keyframes_to_vid import InterpolateKeyframes
from prompt_to_image import Prompt2Img
from rich.console import Console
from rich.prompt import Prompt
from torch import nn
import numpy as np
import cv2
import imageio
import argparse


def create_arg_parser():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--exp_header', type=str, default='', help='Description of exp_header argument')
    parser.add_argument('--prompt', type=str, default='', help='Description of prompt argument')
    parser.add_argument('--dest_words', type=str, default='', help='Description of prompt argument')
    parser.add_argument('--attn_value_min', type=float, default=-.5, help='Description of attn_value_min argument')
    parser.add_argument('--attn_value_max', type=float, default=.5, help='Description of attn_value max argument')
    parser.add_argument('--num_kf', type=int, default=10, help='Description of num_kf argument')
    return parser


class Prompt2VID(nn.Module):
    def __init__(self,
                 args,
                 num_keyframes=4
                 ) -> None:
        super().__init__()
        self.num_keyframes = num_keyframes
        self.args = args
        self.dest_words = args.dest_words.split('_')
        self.setup()
        self.dest_object = args.exp_header

    def setup(self):
        self.device = torch.device(
            'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.console = Console()

        self.descriptor_detection = DescriptorDetector()
        # self.interp_keyframes = InterpolateKeyframes(method='classic_blending', vid_length_sec=4)
        self.interp_keyframes = InterpolateKeyframes(method='ifrnet', vid_length_sec=4)
        self.prompt2imgs = Prompt2Img(attn_value_min=self.args.attn_value_min, attn_value_max=self.args.attn_value_max)
        self.imgs2vid = Imgs2Vid()

    def prompt_user(self):
        prompt = f"Enter a sentence to animate the descriptors"
        sentence = Prompt.ask(f":rocket: {prompt}", default=5)
        return sentence

    def cprint(self, msg, style=None):
        self.console.print(f'{msg}', style=style)

    def forward(self, prompt: str, destination: str, fps: int):
        # descriptors = self.descriptor_detection(sentence=prompt)
        descriptors = self.dest_words
        self.cprint(f'Using descriptor {descriptors[0]} for Prompt to Video')
        descriptor = descriptors[0]
        keyframes = self.prompt2imgs(
            prompt, descriptor=descriptor, clip_len=self.num_keyframes)
        keyframes = np.roll(keyframes, -1, axis=0)
        video = self.interp_keyframes(keyframes)
        np.save(destination + '.mp4', video)
        self.imgs2vid.fps = fps
        self.imgs2vid(video, destination=f"{destination}/{self.dest_object}_video.mp4")
        
        # For DEBUG: save images, video and gif
        os.makedirs(os.path.join(destination, 'images'), exist_ok=True)
        for k_idx in range(keyframes.shape[0]):
            img_path = f"{os.path.join(destination, 'images')}/{k_idx}.png"
            img_resize = cv2.resize(keyframes[k_idx], (512, 512))
            cv2.imwrite(img_path,  cv2.cvtColor(img_resize, cv2.COLOR_RGB2BGR))
        height, width, channels = keyframes[0].shape
        fps = 60  # set desired fps
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # set desired codec
        video_writer = cv2.VideoWriter(f"{destination}/{self.dest_object}_video.mp4", fourcc, fps, (width, height))

        # Create interpolated frames
        for i in range(len(keyframes) - 1):
            for j in range(fps):
                alpha = j / fps
                interpolated_frame = cv2.addWeighted(keyframes[i], 1 - alpha, keyframes[i + 1], alpha, 0)
                video_writer.write(cv2.cvtColor(interpolated_frame, cv2.COLOR_RGB2BGR))

        # Release video writer and display final output
        video_writer.release()
        cv2.destroyAllWindows()

        gif_file = f"{destination}/{self.dest_object}_video.gif"
        imageio.mimsave(gif_file, keyframes, fps=2)  # set desired fps


def inner_main(args):
    num_kf = args.num_kf
    p2vid = Prompt2VID(args=args, num_keyframes=num_kf)
    exp_header = f"{args.exp_header}_{eh_idx}"
    p2vid.dest_object = exp_header
    exp_header = args.exp_header.replace('_', ' ')
    prompt = args.prompt.replace('_', ' ')
    destination = f"p2p/{exp_header}"
    p2vid.forward(prompt, destination=destination, fps=20)


def main(argv):
    parser = create_arg_parser()
    args = parser.parse_args()
    inner_main(args)
