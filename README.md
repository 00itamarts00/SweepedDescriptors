# Prompt-to-Video
Official implementation of Prompt-to-Video.
![p2v_images](p2v_images.png)

See at [Paper](https://github.com/00itamarts00/SweepedDescriptors/blob/main/P2V.pdf) for full report.

## Installation
For install and running this repo:

1. Install required packages:
```
python3 -m pip install requirements.txt
```

2. Run bash file:
 ```
bash install.sh
```

## Results
For generate video, use *prompt_to_video.py* with relevant arguments:
- [ ] prompt (required)

- [ ] exp_header (required): Name of experiment, results will be saved in folder with this name.

- [ ] dest_words (optional): You can specify word(s) to be sweeped. If skiped, our algorithm will detect them.

- [ ] num_kf (default: 4): number of keyframes.

- [ ] attn_value_min/max (default: -0.5/0.5): range of weights for dest words.
	
For example:
 ```
python3 SweepedDescriptors/src/descriptor_gif/prompt_to_video.py --prompt blooming_mountains --exp_header blooming --num_kf 10
```
More than 130 videos were generated and stored under [Results](https://github.com/00itamarts00/SweepedDescriptors/tree/main/results) folder.
