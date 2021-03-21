import os
import glob
import torch
import torch.nn.functional as F
import torchaudio
from torchvision import transforms
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from typing import List, Dict, Tuple, Optional
from src.utils import load, padTensor, get_max_len
from PIL import Image

audio_feature_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762]

def getEmotionDict() -> Dict[str, int]:
    return {'ang': 0, 'exc': 1, 'fru': 2, 'hap': 3, 'neu': 4, 'sad': 5}

def get_dataset_iemocap(data_folder: str, phase: str, img_interval: int, hand_crafted_features: Optional[bool] = False):
    main_folder = os.path.join(data_folder, 'IEMOCAP_RAW_PROCESSED')
    meta = load(os.path.join(main_folder, 'meta.pkl'))

    emoDict = getEmotionDict()
    uttr_ids = open(os.path.join(data_folder, 'IEMOCAP_SPLIT', f'{phase}_split.txt'), 'r').read().splitlines()
    texts = [meta[uttr_id]['text'] for uttr_id in uttr_ids]
    labels = [emoDict[meta[uttr_id]['label']] for uttr_id in uttr_ids]

    if hand_crafted_features:
        text_features = load(os.path.join(data_folder, 'IEMOCAP_HCF_FEATURES', f'{phase}_text_features.pt'))
        audio_features = load(os.path.join(data_folder, 'IEMOCAP_HCF_FEATURES', f'{phase}_audio_features.pt'))
        video_features = load(os.path.join(data_folder, 'IEMOCAP_HCF_FEATURES', f'{phase}_video_features.pt'))

        # Select only the FAUs
        for uttrId in video_features.keys():
            for imgId in video_features[uttrId].keys():
                video_features[uttrId][imgId] = video_features[uttrId][imgId][-35:]

        this_dataset = IEMOCAP_baseline(
            utterance_ids=uttr_ids,
            texts=text_features,
            video_features=video_features,
            audio_features=audio_features,
            labels=labels,
            label_annotations=list(emoDict.keys()),
            img_interval=img_interval
        )
    else:
        this_dataset = IEMOCAP(
            main_folder=main_folder,
            utterance_ids=uttr_ids,
            texts=texts,
            labels=labels,
            label_annotations=list(emoDict.keys()),
            img_interval=img_interval
        )

    return this_dataset

def get_dataset_mosei(data_folder: str, phase: str, img_interval: int, hand_crafted_features: Optional[bool] = False):
    main_folder = os.path.join(data_folder, 'MOSEI_RAW_PROCESSED')
    meta = load(os.path.join(main_folder, 'meta.pkl'))

    ids = open(os.path.join(data_folder, 'MOSEI_SPLIT', f'{phase}_split.txt'), 'r').read().splitlines()
    texts = [meta[id]['text'] for id in ids]
    labels = [meta[id]['label'] for id in ids]

    if hand_crafted_features:
        hcf = load(os.path.join(data_folder, 'MOSEI_HCF_FEATURES', f'mosei_senti_hcf_{phase}.pkl'))
        return MOSEI_baseline(
            ids=ids,
            hcf=hcf,
            labels=labels
        )

    return MOSEI(
        main_folder=main_folder,
        ids=ids,
        texts=texts,
        labels=labels,
        img_interval=img_interval
    )

class MOSEI_baseline(Dataset):
    def __init__(self, ids, hcf, labels: List[int]):
        super(MOSEI_baseline, self).__init__()
        self.ids = ids
        self.hcf_ids = hcf['id'].tolist()
        self.vision = hcf['vision']
        self.audio = hcf['audio']
        self.texts = hcf['text']
        self.labels = np.array(labels)

    def get_annotations(self) -> List[str]:
        return ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprise']

    def getPosWeight(self):
        pos_nums = self.labels.sum(axis=0)
        neg_nums = self.__len__() - pos_nums
        pos_weight = neg_nums / pos_nums
        return pos_weight

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, ind: int) -> Tuple[str, np.array, List[torch.tensor], List[int], np.array]:
        this_id = self.ids[ind]
        hcf_id_index = self.hcf_ids.index(this_id)

        video_feature = self.vision[hcf_id_index]
        audio_feature = self.audio[hcf_id_index]
        text = self.texts[hcf_id_index]
        label = self.labels[ind]

        return this_id, video_feature, audio_feature, text, label

def collate_fn_hcf_mosei(batch):
    uttrIds = []
    texts = []
    labels = []
    video_features = []
    video_lens = []
    audio_features = []
    audio_lens = []
    for dp in batch:
        uttrId, video_feature, audio_feature, words, label = dp
        uttrIds.append(uttrId)
        texts.append(torch.tensor(words, dtype=torch.float32))
        labels.append(label)
        video_features.append(torch.tensor(video_feature))

        audio_features.append(torch.tensor(audio_feature))
        video_lens.append(video_feature.shape[0])
        audio_lens.append(audio_feature.shape[0])

    video_features = torch.cat(video_features, dim=0)
    audio_features = torch.cat(audio_features, dim=0)

    return (
        uttrIds,
        video_features.float(),
        video_lens,
        audio_features.float(),
        audio_lens,
        torch.stack(texts),
        torch.tensor(labels, dtype=torch.float32)
    )

class MOSEI(Dataset):
    def __init__(self, main_folder: str, ids: List[str], texts: List[List[int]], labels: List[int], img_interval: int):
        super(MOSEI, self).__init__()
        self.ids = ids
        self.texts = texts
        self.labels = np.array(labels)
        self.main_folder = main_folder
        self.img_interval = img_interval
        self.crop = transforms.CenterCrop(360)

    def get_annotations(self) -> List[str]:
        return ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprise']

    def getPosWeight(self):
        pos_nums = self.labels.sum(axis=0)
        neg_nums = self.__len__() - pos_nums
        pos_weight = neg_nums / pos_nums
        return pos_weight

    def sample_imgs_by_interval(self, folder: str, fps: Optional[int] = 30) -> List[str]:
        files = glob.glob(f'{folder}/*')
        nums = len(files) - 1
        step = int(self.img_interval / 1000 * fps)
        sampled = [os.path.join(folder, f'image_{i}.jpg') for i in list(range(0, nums, step))]
        if len(sampled) == 0:
            step = int(self.img_interval / 1000 * fps) // 4
            sampled = [os.path.join(folder, f'image_{i}.jpg') for i in list(range(0, nums, step))]
        return sampled

    def cutSpecToPieces(self, spec, stride=32):
        # Split the audio waveform by second
        total = -(-spec.size(-1) // stride)
        specs = []
        for i in range(total):
            specs.append(spec[:, :, :, i * stride:(i + 1) * stride])

        # Pad the last piece
        lastPieceLength = specs[-1].size(-1)
        if lastPieceLength < stride:
            padRight = stride - lastPieceLength
            specs[-1] = F.pad(specs[-1], (0, padRight))

        return specs

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, ind: int) -> Tuple[str, np.array, List[torch.tensor], List[int], np.array]:
        this_id = self.ids[ind]
        sample_folder = os.path.join(self.main_folder, this_id)

        sampledImgs = []
        for imgPath in self.sample_imgs_by_interval(sample_folder):
            this_img = Image.open(imgPath)
            H = np.float32(this_img).shape[0]
            W = np.float32(this_img).shape[1]
            if H > 360:
                resize = transforms.Resize([H // 2, W // 2])
                this_img = resize(this_img)
            this_img = self.crop(this_img)
            sampledImgs.append(np.float32(this_img))
        sampledImgs = np.array(sampledImgs)

        waveform, sr = torchaudio.load(os.path.join(sample_folder, f'audio.wav'))

        # Cut Spec
        specgram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            win_length=int(float(sr) / 16000 * 400),
            n_fft=int(float(sr) / 16000 * 400)
        )(waveform).unsqueeze(0)
        specgrams = self.cutSpecToPieces(specgram)

        return this_id, sampledImgs, specgrams, self.texts[ind], self.labels[ind]

class IEMOCAP(Dataset):
    def __init__(self, main_folder: str, utterance_ids: List[str], texts: List[List[int]], labels: List[int],
                 label_annotations: List[str], img_interval: int):
        super(IEMOCAP, self).__init__()
        self.utterance_ids = utterance_ids
        self.texts = texts
        self.labels = F.one_hot(torch.tensor(labels)).numpy()
        self.label_annotations = label_annotations

        self.utteranceFolders = {
            folder.split('/')[-1]: folder
            for folder in glob.glob(os.path.join(main_folder, '**/*'))
        }
        self.img_interval = img_interval

    def get_annotations(self) -> List[str]:
        return self.label_annotations

    def use_left(self, utteranceFolder: str) -> bool:
        entries = utteranceFolder.split('_')
        return entries[0][-1] == entries[-1][0]

    def sample_imgs_by_interval(self, folder: str, imgNamePrefix: str, fps: Optional[int] = 30) -> List[str]:
        '''
        Arguments:
            @folder - utterance folder
            @imgNamePrefix - prefix of the image name (determines L/R)
            @interval - how many ms per image frame
            @fps - fps of the original video
        '''
        files = glob.glob(f'{folder}/*')
        nums = (len(files) - 5) // 2
        step = int(self.img_interval / 1000 * fps)
        sampled = [os.path.join(folder, f'{imgNamePrefix}{i}.jpg') for i in list(range(0, nums, step))]
        return sampled

    def cutWavToPieces(self, waveform, sampleRate):
        # Split the audio waveform by second
        total = int(np.ceil(waveform.size(-1) / sampleRate))
        waveformPieces = []
        for i in range(total):
            waveformPieces.append(waveform[:, i * sampleRate:(i + 1) * sampleRate])

        # Pad the last piece
        lastPieceLength = waveformPieces[-1].size(-1)
        if lastPieceLength < sampleRate:
            padLeft = (sampleRate - lastPieceLength) // 2
            padRight = sampleRate - lastPieceLength - padLeft
            waveformPieces[-1] = F.pad(waveformPieces[-1], (padLeft, padRight))
        return waveformPieces

    def cutSpecToPieces(self, spec, stride=32):
        # Split the audio waveform by second
        total = -(-spec.size(-1) // stride)
        specs = []
        for i in range(total):
            specs.append(spec[:, :, :, i * stride:(i + 1) * stride])

        # Pad the last piece
        lastPieceLength = specs[-1].size(-1)
        if lastPieceLength < stride:
            padRight = stride - lastPieceLength
            specs[-1] = F.pad(specs[-1], (0, padRight))
        return specs

    def getPosWeight(self):
        pos_nums = self.labels.sum(axis=0)
        neg_nums = self.__len__() - pos_nums
        pos_weight = neg_nums / pos_nums
        return pos_weight

    def __len__(self):
        return len(self.utterance_ids)

    def __getitem__(self, ind: int) -> Tuple[str, np.array, List[torch.tensor], List[int], np.array]:
        uttrId = self.utterance_ids[ind]
        uttrFolder = self.utteranceFolders[uttrId]
        use_left = self.use_left(uttrId)
        suffix = 'L' if use_left else 'R'
        audio_suffix = 'L' if use_left else 'R'
        imgNamePrefix = f'image_{suffix}_'

        sampledImgs = np.array([
            np.float32(Image.open(imgPath)) # .transpose()
            for imgPath in self.sample_imgs_by_interval(uttrFolder, imgNamePrefix)
        ])

        waveform, sr = torchaudio.load(os.path.join(uttrFolder, f'audio_{audio_suffix}.wav'))

        # Cut WAV
        # waveformPieces = self.cutWavToPieces(waveform, sr)
        # specgrams = [torchaudio.transforms.MelSpectrogram()(waveformPiece).unsqueeze(0) for waveformPiece in waveformPieces]

        # Cut Spec
        specgram = torchaudio.transforms.MelSpectrogram(sample_rate=sr, win_length=int(float(sr) / 16000 * 400))(waveform).unsqueeze(0)
        specgrams = self.cutSpecToPieces(specgram)

        return uttrId, sampledImgs, specgrams, self.texts[ind], self.labels[ind]

def collate_fn(batch):
    utterance_ids = []
    texts = []
    labels = []

    newSampledImgs = None
    imgSeqLens = []

    specgrams = []
    specgramSeqLens = []

    for dp in batch:
        utteranceId, sampledImgs, specgram, text, label = dp
        if sampledImgs.shape[0] == 0:
            continue
        utterance_ids.append(utteranceId)
        texts.append(text)
        labels.append(label)

        imgSeqLens.append(sampledImgs.shape[0])
        newSampledImgs = sampledImgs if newSampledImgs is None else np.concatenate((newSampledImgs, sampledImgs), axis=0)

        specgramSeqLens.append(len(specgram))
        specgrams.append(torch.cat(specgram, dim=0))

    imgs = newSampledImgs

    return (
        utterance_ids,
        imgs,
        imgSeqLens,
        torch.cat(specgrams, dim=0),
        specgramSeqLens,
        texts,
        torch.tensor(labels, dtype=torch.float32)
    )

class IEMOCAP_baseline(Dataset):
    def __init__(self, utterance_ids, texts, video_features, audio_features, labels, label_annotations, img_interval=500):
        super(IEMOCAP_baseline, self).__init__()
        self.utterance_ids = utterance_ids
        self.texts_features = texts
        self.video_features = video_features
        self.audio_features = audio_features
        self.labels = F.one_hot(torch.tensor(labels)).numpy()
        self.label_annotations = label_annotations
        self.img_interval = img_interval

    def getPosWeight(self):
        pos_nums = self.labels.sum(axis=0)
        neg_nums = self.__len__() - pos_nums
        pos_weight = neg_nums / pos_nums
        return pos_weight

    def get_annotations(self):
        return self.label_annotations

    def use_left(self, utteranceFolder: str) -> bool:
        entries = utteranceFolder.split('_')
        return entries[0][-1] == entries[-1][0]

    def sample_by_interval(self, prefix: str, dialog_uttr_features: dict, full=False) -> List[str]:
        nums = len(dialog_uttr_features) // 2
        step = 1 if full else int(self.img_interval / 1000 * 30)
        new_dialog_uttr_features = []

        for i in list(range(0, nums, step)):
            try:
                new_dialog_uttr_features.append(dialog_uttr_features[f'{prefix}{i}'])
            except KeyError:
                continue

        return new_dialog_uttr_features

    def __len__(self):
        return len(self.utterance_ids)

    def __getitem__(self, ind: int):
        uttrId = self.utterance_ids[ind]
        suffix = 'L' if self.use_left(uttrId) else 'R'
        audio_suffix = 'L' if self.use_left(uttrId) else 'R'
        img_prefix = f'image_{suffix}_'

        video_feature = self.video_features[uttrId]
        video_feature = self.sample_by_interval(img_prefix, video_feature, full=True)
        audio_feature = self.audio_features[uttrId][audio_suffix]

        text = self.texts_features[uttrId]
        label = self.labels[ind]

        return uttrId, video_feature, audio_feature, text, label

class HCFDataLoader(DataLoader):
    FEATURE_TYPE_ALL = 0
    FEATURE_TYPE_NO_BBE = 1
    FEATURE_TYPE_NO_MFCC = 2
    FEATURE_TYPE_NO_PHONOLOGICAL = 3
    FEATURE_TYPE_MEAN_STD_ALL = 4
    FEATURE_TYPE_NO_MIN_MAX_ALL = 5
    FEATURE_TYPE_PLUS = 6

    FEATURE_TYPE_DICT = {
        FEATURE_TYPE_ALL: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 100, 101],
        FEATURE_TYPE_NO_BBE: [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762],
        FEATURE_TYPE_NO_MFCC: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762],
        FEATURE_TYPE_NO_PHONOLOGICAL: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33],
        FEATURE_TYPE_MEAN_STD_ALL: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 655, 656, 661, 662, 667, 668, 673, 674, 679, 680, 685, 686, 691, 692, 697, 698, 703, 704, 709, 710, 715, 716, 721, 722, 727, 728, 733, 734, 739, 740, 745, 746, 751, 752, 757, 758],
        FEATURE_TYPE_NO_MIN_MAX_ALL: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 655, 656, 657, 658, 661, 662, 663, 664, 667, 668, 669, 670, 673, 674, 675, 676, 679, 680, 681, 682, 685, 686, 687, 688, 691, 692, 693, 694, 697, 698, 699, 700, 703, 704, 705, 706, 709, 710, 711, 712, 715, 716, 717, 718, 721, 722, 723, 724, 727, 728, 729, 730, 733, 734, 735, 736, 739, 740, 741, 742, 745, 746, 747, 748, 751, 752, 753, 754, 757, 758, 759, 760],
        FEATURE_TYPE_PLUS: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 116, 117, 118, 119, 120, 121, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 238, 239, 240, 241, 242, 243, 360, 361, 362, 363, 364, 365, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 482, 483, 484, 485, 486, 487, 552, 553, 554, 555, 556, 557, 558, 559, 562, 563, 564, 565, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762]
    }

    def __init__(self, feature_type=0, *args, **kwargs):
        super(HCFDataLoader, self).__init__(*args, **kwargs)
        self.audio_feature_indices = self.FEATURE_TYPE_DICT[feature_type]
        self.collate_fn = self.collate_fn_hcf

    def collate_fn_hcf(self, batch):
        uttrIds = []
        texts = []
        labels = []
        video_features = []
        video_lens = []
        audio_features = []
        audio_lens = []
        for dp in batch:
            uttrId, video_feature, audio_feature, word_indices, label = dp
            uttrIds.append(uttrId)
            texts.append(word_indices)
            labels.append(label)
            video_features.append(torch.tensor(video_feature))

            audio_features.append(torch.tensor(audio_feature).t())
            video_lens.append(len(video_feature))
            audio_lens.append(audio_feature.shape[1])

        text_max_len = get_max_len(texts)
        for i in range(len(texts)):
            if len(texts[i]) == 0:
                texts[i] = torch.zeros(text_max_len, 300, dtype=torch.float32)
            else:
                texts[i] = padTensor(torch.tensor(texts[i], dtype=torch.float32), text_max_len)

        video_features = torch.cat(video_features, dim=0)
        audio_features = torch.cat(audio_features, dim=0)

        audio_features = audio_features[:, self.audio_feature_indices]

        return (
            uttrIds,
            video_features.float(),
            video_lens,
            audio_features.float(),
            audio_lens,
            torch.stack(texts),
            torch.tensor(labels, dtype=torch.float32)
        )
