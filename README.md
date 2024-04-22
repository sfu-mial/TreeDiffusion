# Representing Anatomical Trees by Denoising Diffusion of Implicit Neural Fields
This is the official code supporting our paper:
> [Representing Anatomical Trees by Denoising Diffusion of Implicit Neural Fields](https://arxiv.org/abs/2403.08974) </br>
> [Ashish Sinha](https://sinashish.github.io/) and [Ghassan Hamarneh](https://www.medicalimageanalysis.com/home) </br>
> Simon Fraser University </br>
> [[Paper]](./assets/paper.pdf) | [[arxiv]](https://arxiv.org/abs/2403.08974) | [[Poster]](./assets/poster.pdf) | [[Code]](./)

## Abstract

Anatomical trees play a central role in clinical diagnosis and treatment planning. However, accurately representing anatomical trees is challenging due to their varying and complex topology and geometry.
Traditional methods for representing tree structures, captured using medical imaging, while invaluable for visualizing vascular and bronchial networks, exhibit drawbacks in terms of limited resolution, flexibility, and efficiency. Recently, implicit neural representations (INRs) have emerged as a powerful tool for representing shapes accurately and efficiently. We propose a novel approach for representing anatomical trees using INR, while also capturing the distribution of a set of trees via denoising diffusion in the space of INRs. We accurately capture the intricate geometries and topologies of anatomical trees at any desired resolution. Through extensive qualitative and quantitative evaluation, we demonstrate high-fidelity tree reconstruction with arbitrary resolution yet compact storage, and versatility across anatomical sites and tree complexities.

![image](https://github.com/sinAshish/TreeDiffusion/assets/21974209/e7d49e8e-429c-45e1-99cb-9e05b238a22c)

## How to Use 

#### Setup Environment
```
python -m venv venv
source activate
pip install -r requirements.txt
```

#### Stage 1
```
python stage1.py --pcd_path /path/to/data --epochs <no. of iterations> --nsample <no. of points to sample>
```

#### Stage 2
```
python stage2.py --epochs <no. of epochs> --dset_name <dataset to use>
```

## Cite

```
@misc{sinha2024treeinr,
      title={Representing Anatomical Trees by Denoising Diffusion of Implicit Neural Fields}, 
      author={Ashish Sinha and Ghassan Hamarneh},
      year={2024},
      eprint={2403.08974},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
