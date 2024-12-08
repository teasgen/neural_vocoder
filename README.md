<div align="center">

# HiFiGAN by teasgen

[\[🔥 Implementation Techinical Report\]]([src/docs/paper.pdf](https://wandb.ai/teasgen/vocoder/reports/HiFIGAN-by-teasgen--VmlldzoxMDUwNDY5Ng))

</div>

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-train">How To Train</a> •
  <a href="#how-to-evaluate">How To Evaluate</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

This repository contains scripts for training and evaluation of HiFiGAN

## Installation

Follow these steps to install the project:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html).

   ```bash
   # create env
   conda create -n project_env python=3.10

   # activate env
   conda activate project_env
   ```

1. Install all required packages:

   ```bash
   pip install -r requirements.txt
   ```

2. Install `pre-commit`:
   ```bash
   pre-commit install
   ```

## How To Train
You should have single A100-80gb GPU to exactly reproduce training, otherwise please implement and use gradient accumulation

To train a model, run the following commands and register in WandB:

Three-steps sequential training:
- 8k context
```bash
python3 train.py -cn hifigan.yaml \
  writer.run_name=hifigan_v1_my_v2_sr_22_05_len_8k_zero_to_hero \
  dataloader.batch_size=64 \
  model.hu=512 \
  trainer.n_epochs=200 \
  +datasets.train.rand_split=True \
  trainer.epoch_len=175
```

- 22k context
```bash
python3 train.py -cn hifigan_cos_sheduler.yaml \
  writer.run_name=hifigan_v1_my_v2_sr_22_05_len_8k_zero_to_hero_resume_len22k \
  dataloader.batch_size=32 \
  model.hu=512 \
  trainer.n_epochs=500 \
  +datasets.train.rand_split=True \
  trainer.epoch_len=350 \
  datasets.train.audio_length_limit=22050 \
  datasets.test.audio_length_limit=22050 \
  trainer.resume_from=<PATH_TO_SAVING_DIR>/hifigan_v1_my_v2_sr_22_05_len_8k_zero_to_hero/checkpoint-epoch60.pth
```

- 44k context
```bash
python3 train.py -cn hifigan_cos_sheduler_resume_low_lr.yaml \
  writer.run_name=hifigan_v1_my_v2_sr_22_05_len_8k_zero_to_hero_resume_2_len44k \
  dataloader.batch_size=16 \
  model.hu=512 \
  trainer.n_epochs=550 \
  +datasets.train.rand_split=True \
  trainer.epoch_len=700 \
  datasets.train.audio_length_limit=44100 \
  datasets.test.audio_length_limit=44100 \
  trainer.resume_from=<PATH_TO_SAVING_DIR>/hifigan_v1_my_v2_sr_22_05_len_8k_zero_to_hero_resume_len22k/checkpoint-epoch460.pth
```

Moreover, training logs are available in WandB

- https://wandb.ai/teasgen/vocoder

## How To Evaluate

Best model could be downloaded by link https://drive.google.com/file/d/1xe3kqva4BiXi0hAGMBaqEiT795AE16hn/view?usp=sharing
Or you may download it using CLI

```bash
gdown 1xe3kqva4BiXi0hAGMBaqEiT795AE16hn
tar xvf hifigan_v1_my_v2_sr_22_05_len_8k_zero_to_hero_resume_2_len44k.tar
```
The checkpoint will be saved into `<CURRENT_DIR>/hifigan_v1_my_v2_sr_22_05_len_8k_zero_to_hero_resume_2_len44k/checkpoint-epoch550.pth`

There are three types of evaluation

### Reproduce the Wav using Mel-spectrogram.
The input - directory with ground truth Wavs, the output - directory with generated Wavs. GT wav is being transformed to Mel-spectrogram and after using HiFiGAN-repack-by-teasgen transformed to Wav.
The example of directory with GT wavs located in this repo
- gt_wavs_js - LJSpeech dataset 5 random samples from test split
- gt_wavs - dataset with 5 random long Wavs

```bash
python3 synthesize.py -cn inference.yaml \
  inferencer.from_pretrained=hifigan_v1_my_v2_sr_22_05_len_8k_zero_to_hero_resume_2_len44k/checkpoint-epoch550.pth \
  inferencer.save_path=wav2wav_lj \
  datasets.test.wav_dir=gt_wavs_lj
```
Generated wavs will be saved into `<CURRENT_DIR>/data/test/<inferencer.save_path>`
Instead of `datasets.test.wav_dir=gt_wavs_lj` you may place custom dir: `datasets.test.wav_dir=<GT_WAVS_DIRNAME>`

### Generate the Wav using text. Directory with texts version
The input - directory with texts, the output - directory with generated Wavs. Text is being transformed to Mel-spectrogram using [Tacotron2](https://github.com/pytorch/hub/blob/master/nvidia_deeplearningexamples_waveglow.md#example) and after using HiFiGAN-repack-by-teasgen transformed to Wav.
The example of directory with GT wavs located in this repo
- test_data_text - transcriptions of `gt_wavs`
  
```bash
python3 synthesize.py -cn synthesize.yaml \
  inferencer.from_pretrained=hifigan_v1_my_v2_sr_22_05_len_8k_zero_to_hero_resume_2_len44k/checkpoint-epoch550.pth \
  inferencer.save_path=text_dir2wav \
  datasets.test.transcription_dir=test_data_text
```
Generated wavs will be saved into `<CURRENT_DIR>/data/test/<inferencer.save_path>`
Instead of `datasets.test.transcription_dir=test_data_text` you may place custom dir: `datasets.test.transcription_dir=<GT_WAVS_DIRNAME>`

### Generate the Wav using text. Text in CLI version
The input text is set via CLI, the output - directory with generated Wavs.

```bash
python3 synthesize.py -cn synthesize_text_cli.yaml\
  inferencer.from_pretrained=hifigan_v1_my_v2_sr_22_05_len_8k_zero_to_hero_resume_2_len44k/checkpoint-epoch550.pth \
  inferencer.save_path=text_cli2wav \
  datasets.test.transcription="I am Vlad\, this is my pet project"
```
Generated wavs will be saved into `<CURRENT_DIR>/data/test/<inferencer.save_path>`

### Neuro-MOS calculation
I am using https://github.com/AndreevP/wvmos

`wvmos` installation
```bash
pip install git+https://github.com/AndreevP/wvmos
```

Evaluation
```bash
python3 src/utils/mos_calculation.py --predicts-dir <PATH_TO_DIR_WITH_PREDICTIONS>
```
<PATH_TO_DIR_WITH_PREDICTIONS> is a directory with Wavs

## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
