# Multimodal End-to-End Sparse Model for Emotion Recognition

[![](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/) [![CC BY 4.0][cc-by-shield]][cc-by]


<img align="right" src="img/HKUST.jpg" width="15%"/>

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

Paper accepted at the NAACL 2021:

**Multimodal End-to-End Sparse Model for Emotion Recognition**, by **[Wenliang Dai *](https://wenliangdai.github.io/)**, Samuel Cahyawijaya *, [Zihan Liu](https://zliucr.github.io/), Pascale Fung.

## Paper Abstract

> Existing works on multimodal affective computing tasks, such as emotion recognition, generally adopt a two-phase pipeline, first extracting feature representations for each single modality with hand-crafted algorithms and then performing end-to-end learning with the extracted features. However, the extracted features are fixed and cannot be further fine-tuned on different target tasks, and manually finding feature extraction algorithms does not generalize or scale well to different tasks, which can lead to sub-optimal performance. In this paper, we develop a fully end-to-end model that connects the two phases and optimizes them jointly. In addition, we restructure the current datasets to enable the fully end-to-end training. Furthermore, to reduce the computational overhead brought by the end-to-end model, we introduce a sparse cross-modal attention mechanism for the feature extraction. Experimental results show that our fully end-to-end model significantly surpasses the current state-of-the-art models based on the two-phase pipeline. Moreover, by adding the sparse cross-modal attention, our model can maintain performance with around half the computation in the feature extraction part.

If you work is inspired by our paper or code, please cite it, thanks!

<pre>
@inproceedings{Dai2021MultimodalES,
  title={Multimodal End-to-End Sparse Model for Emotion Recognition},
  author={Wenliang Dai and Samuel Cahyawijaya and Zihan Liu and Pascale Fung},
  year={2021}
}
</pre>

You can also check our blog [here](https://nlp.caire.hkust.edu.hk/blog/2021/sparse-multimodal/).

## Dataset

As mentioned in our paper, one of the contribution is that we reorganize two datasets (IEMOCAP and CMU-MOSEI) to enable training from the raw data. To the best of our knowledge, prior to our work, papers using these two datasets are based on pre-extracted features, and we did not find a way to map those features back with raw data. Therefore, we did a heavy reorganization of these datasets (refer to Section 3 of the paper for more details).

The raw data can be downloaded from [CMU-MOSEI](http://immortal.multicomp.cs.cmu.edu/raw_datasets/CMU_MOSEI.zip) (~120GB) and [IEMOCAP](https://hkustconnect-my.sharepoint.com/:u:/g/personal/wdaiai_connect_ust_hk/EdZoawxqQ01Ej38NpflFZPEB4zYR9RxIPcaAPcFU77qFgQ?e=7efIl0) (~16.5GB). However, for the IEMOCAP, you need to [request for a permission](https://sail.usc.edu/iemocap/iemocap_release.htm) from the original author, then we can give the passcode to download.

We provide two Python scripts as examples of processing the raw data in the `./preprocessing` folder. Alternatively, you can also download our processed raw data for training directly, as shown in the section below.

## Preparation

### Dataset

To run our code directly, you can download the processed data from [here](https://hkustconnect-my.sharepoint.com/:u:/g/personal/wdaiai_connect_ust_hk/EbEzVnCduqVNuT_LvuRApVQBagraPx7nGqIEHgdnGPjN7g?e=zeMtct) (88.6G). Unzip it and the tree structure of the data direcotry looks like this:

```
./data
- IEMOCAP_HCF_FEATURES
- IEMOCAP_RAW_PROCESSED
- IEMOCAP_SPLIT
- MOSEI_RAW_PROCESSED
- MOSEI_HCF_FEATURES
- MOSEI_SPLIT
```

### Environment

* Python 3.7
* PyTorch 1.6.0
* torchaudio 0.6.0
* torchvision 0.7.0
* transformers 3.4.0 ([huggingface](https://huggingface.co/transformers/))
* [sparseconvnet](https://github.com/facebookresearch/SparseConvNet)
* [facenet-pytorch](https://github.com/timesler/facenet-pytorch) 2.3.0

## Command examples for running

### Train the MME2E

```console
python main.py -lr=5e-5 -ep=40 -mod=tav -bs=8 --img-interval=500 --early-stop=6 --loss=bce --cuda=3 --model=mme2e --num-emotions=6 --trans-dim=64 --trans-nlayers=4 --trans-nheads=4 --text-lr-factor=10 --text-model-size=base --text-max-len=100
```

### Train the sparse MME2E

```console
python main.py -lr=5e-5 -ep=40 -mod=tav -bs=2 --img-interval=500 --early-stop=6 --loss=bce --cuda=3 --model=mme2e_sparse --num-emotions=6 --trans-dim=64 --trans-nlayers=4 --trans-nheads=4 --text-lr-factor=10 -st=0.8 --text-model-size=base --text-max-len=100
```

### Baselines

#### LF_RNN

```console
python main.py -lr=5e-4 -ep=60 -mod=tav -bs=32 --early-stop=8 --loss=bce --cuda=1 --model=lf_rnn --num-emotions=6 --hand-crafted --clip=2
```

#### LF_TRANSFORMER

```console
python main.py -lr=5e-4 -ep=60 -mod=tav -bs=32 --early-stop=8 --loss=bce --cuda=0 --model=lf_transformer --num-emotions=6 --hand-crafted --clip=2
```

## CLI

```
usage: main.py [-h] -bs BATCH_SIZE -lr LEARNING_RATE [-wd WEIGHT_DECAY] -ep
               EPOCHS [-es EARLY_STOP] [-cu CUDA] [-cl CLIP] [-sc] [-se SEED]
               [--loss LOSS] [--optim OPTIM] [--text-lr-factor TEXT_LR_FACTOR]
               [-mo MODEL] [--text-model-size TEXT_MODEL_SIZE]
               [--fusion FUSION] [--feature-dim FEATURE_DIM]
               [-st SPARSE_THRESHOLD] [-hfcs HFC_SIZES [HFC_SIZES ...]]
               [--trans-dim TRANS_DIM] [--trans-nlayers TRANS_NLAYERS]
               [--trans-nheads TRANS_NHEADS] [-aft AUDIO_FEATURE_TYPE]
               [--num-emotions NUM_EMOTIONS] [--img-interval IMG_INTERVAL]
               [--hand-crafted] [--text-max-len TEXT_MAX_LEN]
               [--datapath DATAPATH] [--dataset DATASET] [-mod MODALITIES]
               [--valid] [--test] [--ckpt CKPT] [--ckpt-mod CKPT_MOD]
               [-dr DROPOUT] [-nl NUM_LAYERS] [-hs HIDDEN_SIZE] [-bi] [--gru]

Multimodal End-to-End Sparse Model for Emotion Recognition

optional arguments:
  -h, --help            show this help message and exit
  -bs BATCH_SIZE, --batch-size BATCH_SIZE
                        Batch size
  -lr LEARNING_RATE, --learning-rate LEARNING_RATE
                        Learning rate
  -wd WEIGHT_DECAY, --weight-decay WEIGHT_DECAY
                        Weight decay
  -ep EPOCHS, --epochs EPOCHS
                        Number of epochs
  -es EARLY_STOP, --early-stop EARLY_STOP
                        Early stop
  -cu CUDA, --cuda CUDA
                        Cude device number
  -cl CLIP, --clip CLIP
                        Use clip to gradients
  -sc, --scheduler      Use scheduler to optimizer
  -se SEED, --seed SEED
                        Random seed
  --loss LOSS           loss function
  --optim OPTIM         optimizer function: adam/sgd
  --text-lr-factor TEXT_LR_FACTOR
                        Factor the learning rate of text model
  -mo MODEL, --model MODEL
                        Which model
  --text-model-size TEXT_MODEL_SIZE
                        Size of the pre-trained text model
  --fusion FUSION       How to fuse modalities
  --feature-dim FEATURE_DIM
                        Dimension of features outputed by each modality model
  -st SPARSE_THRESHOLD, --sparse-threshold SPARSE_THRESHOLD
                        Threshold of sparse CNN layers
  -hfcs HFC_SIZES [HFC_SIZES ...], --hfc-sizes HFC_SIZES [HFC_SIZES ...]
                        Hand crafted feature sizes
  --trans-dim TRANS_DIM
                        Dimension of the transformer after CNN
  --trans-nlayers TRANS_NLAYERS
                        Number of layers of the transformer after CNN
  --trans-nheads TRANS_NHEADS
                        Number of heads of the transformer after CNN
  -aft AUDIO_FEATURE_TYPE, --audio-feature-type AUDIO_FEATURE_TYPE
                        Hand crafted audio feature types
  --num-emotions NUM_EMOTIONS
                        Number of emotions in data
  --img-interval IMG_INTERVAL
                        Interval to sample image frames
  --hand-crafted        Use hand crafted features
  --text-max-len TEXT_MAX_LEN
                        Max length of text after tokenization
  --datapath DATAPATH   Path of data
  --dataset DATASET     Use which dataset
  -mod MODALITIES, --modalities MODALITIES
                        what modalities to use
  --valid               Only run validation
  --test                Only run test
  --ckpt CKPT           Path of checkpoint
  --ckpt-mod CKPT_MOD   Load which modality of the checkpoint
  -dr DROPOUT, --dropout DROPOUT
                        dropout
  -nl NUM_LAYERS, --num-layers NUM_LAYERS
                        num of layers of LSTM
  -hs HIDDEN_SIZE, --hidden-size HIDDEN_SIZE
                        hidden vector size of LSTM
  -bi, --bidirectional  Use Bi-LSTM
  --gru                 Use GRU rather than LSTM
```
