# Inference and Denoise: Causal Inference-based Neural Speech Enhancement
This is the official implementation of "Inference and Denoise: Causal Inference-based Neural Speech Enhancement".<br>
We are updating our code and additional supplementary material and soon.

<img src="https://github.com/aleXiehta/Causal-SE/blob/main/causal-se-systems.PNG" width="500">


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
  
## References

```bib
@article{hsieh2022inference,
  title={Inference and Denoise: Causal Inference-based Neural Speech Enhancement},
  author={Hsieh, Tsun-An and Yang, Chao-Han Huck and Chen, Pin-Yu and Siniscalchi, Sabato Marco and Tsao, Yu},
  journal={arXiv preprint arXiv:2211.01189},
  year={2022}
}
```
  
## Acknowledgments
* [Bio-ASP Lab](https://bio-asplab.citi.sinica.edu.tw), CITI, Academia Sinica, Taipei, Taiwan

* [Georgia Tech](https://www.ece.gatech.edu/)

* [NTNU / U Kore](https://www.ntnu.edu/employees/marco.siniscalchi)

* [IBM Research AI](https://researcher.watson.ibm.com/researcher/view.php?person=ibm-Pin-Yu.Chen)
