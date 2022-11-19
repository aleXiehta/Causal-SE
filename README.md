# Inference and Denoise: Causal Inference-based Neural Speech Enhancement
This is the official implementation of "Inference and Denoise: Causal Inference-based Neural Speech Enhancement".<br>
We will update our code and additional supplementary material very soon.

## Requirements
* pytorch 1.12.1
* pytorch_lightning 1.7.5
* SOTA SE models (slightly modified to fit the usage of CISE) are credited to:
  * [MANNER](https://github.com/winddori2002/MANNER.git)
  * [CMGAN](https://github.com/ruizhecao96/CMGAN.git)
  * [DEMUCS](https://github.com/facebookresearch/denoiser.git)

## Usage
To train and inference, please run
```
python main.py \
  --save_path <path/to/checkpoint/and/tensorboard/> \
  --output_dir <path/to/prediction> \
  --use_pretrained \
  --batch_size 8 \
  --hop_length 320
```
  
## Acknowledgments
* [Bio-ASP Lab](https://bio-asplab.citi.sinica.edu.tw), CITI, Academia Sinica, Taipei, Taiwan
