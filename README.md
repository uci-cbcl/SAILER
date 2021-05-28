# SAILER

Pytorch implementation for [SAILER: Scalable and Accurate Invariant Representation Learning for Single-Cell ATAC-Seq Processing and Integration](https://www.biorxiv.org/content/10.1101/2021.01.28.428689v2)

## Setup

Clone the repository.

```
git clone https://github.com/uci-cbcl/SAILER.git
```

Navigate to the root of this repo and setup the conda environment.

```
conda env create -f deepatac.yml
```

Activate conda environment.

```
conda activate deepatac
```

## Data

Please download data [here](https://drive.google.com/file/d/1sq9QiWL3_MnMCEB3yJ_O6BW8GYjwDRMT/view?usp=sharing) and setup your data folder as the following structure:

```
SAILER
|___data  
    |___MouseAtlas
        |___...
```

## Visualizing Results

 Please download the pretrained model [here](https://drive.google.com/file/d/1L0ahnqC6bM2hOkn1ZkcOTuPU_EH-4K1X/view?usp=sharing) and setup your data folder as the following structure:

```
SAILER
|___models  
    |___MouseAtlas.pt
```

Navigate to the root of this repo and run the following command. Result will be stored under ./results directory.

```
python eval.py -l './models/MouseAtlas.pt' -d atlas
```

## Training

To train the model from scratch, use the following command. 

```
python train.py -b 400 -d atlas --name mouse_atlas
```

For more information, see

```
python train.py -h
```

