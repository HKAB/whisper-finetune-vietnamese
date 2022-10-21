import os
import numpy as np

import torch
import torchaudio

import pandas as pd
import whisper
import torchaudio.transforms as at
from utils import load_wave
from pathlib import Path
 
class WhisperDataCollatorWithPadding:
    """
    Using for collating many input tensors with different sizes to batch and maybe applying some processes to data.
    Input: list or dictionary of input tensors
    Output: list dictionary of batched tensors
    """
    def __call__(self, features):
        input_ids, labels, dec_input_ids, texts = [], [], [], []
        for f in features:
            input_ids.append(f["input_ids"])
            labels.append(f["labels"])
            dec_input_ids.append(f["dec_input_ids"])
            texts.append(f["text"])
        
        input_ids = torch.concat([input_id[None, :] for input_id in input_ids])  # [batch_size, seq_len]

        label_lengths = [len(lab) for lab in labels]  # same size with input_ids
        dec_input_ids_length = [len(e) for e in dec_input_ids]
        max_label_len = max(label_lengths + dec_input_ids_length)

        labels = [
            np.pad(lab, (0, max_label_len - lab_len), "constant", constant_values=-100)
            for lab, lab_len in zip(labels, label_lengths)
        ]
        dec_input_ids = [
            np.pad(e, (0, max_label_len - e_len), "constant", constant_values=50257)
            for e, e_len in zip(dec_input_ids, dec_input_ids_length)
        ]  # 50257 is eot token id

        batch = {"labels": labels, "dec_input_ids": dec_input_ids}

        batch = {
            k: torch.tensor(np.array(v), requires_grad=False) for k, v in batch.items()
        }
        batch["input_ids"] = input_ids
        batch["texts"] = texts
        return batch

class WhisperDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, sample_rate=16000) -> None:
        super().__init__()

        self.dataset = dataset
        self.sample_rate = sample_rate

        self.options = whisper.DecodingOptions(language="vi", without_timestamps=True)
        self.tokenizer = whisper.tokenizer.get_tokenizer(
            True, language="vi", task=self.options.task
        )

    def load_wave(self, wave_path, sample_rate: int = 16000) -> torch.Tensor:
        waveform, sr = torchaudio.load(wave_path, normalize=True)
        if sample_rate != sr:
            waveform = at.Resample(sr, sample_rate)(waveform)
        return waveform
    

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, id):
        audio_id, audio_path, text = self.dataset[id]

        audio = self.load_wave(audio_path, sample_rate=self.sample_rate)
        audio = whisper.pad_or_trim(audio.flatten())
        mel = whisper.log_mel_spectrogram(audio)

        text_token = [
            *self.tokenizer.sot_sequence_including_notimestamps
        ] + self.tokenizer.encode(text)
        labels = text_token[1:] + [self.tokenizer.eot]
        if len(text_token) >= 448:
            audio_id, audio_path, text = self.dataset[0]

            audio = self.load_wave(audio_path, sample_rate=self.sample_rate)
            audio = whisper.pad_or_trim(audio.flatten())
            mel = whisper.log_mel_spectrogram(audio)

            text_token = [
                *self.tokenizer.sot_sequence_including_notimestamps
            ] + self.tokenizer.encode(text)
            labels = text_token[1:] + [self.tokenizer.eot]
        return {
            "input_ids": mel,
            "labels": labels,
            "dec_input_ids": text_token,
            "text": text,
        }


def load_dataset(dataset_name, test=False):
    train_dataset = None
    test_dataset = None

    if dataset_name == 'fluers':
        print('Loading Vietnamese Fluers dataset...')
        if not os.path.exists('vi_vn.tar.gz'):
            os.system("wget https://storage.googleapis.com/xtreme_translations/FLEURS102/vi_vn.tar.gz")
            os.makedirs('fluers', exist_ok=True)
            os.system("tar -xf 'vi_vn.tar.gz' -C fluers")
        if not test:
            train_list_files = get_list_files_fluers('train')
            val_list_files = get_list_files_fluers('dev')
            train_list_files +=val_list_files
            print('Num train samples:', len(train_list_files))
            train_dataset = WhisperDataset(train_list_files)

        test_list_files = get_list_files_fluers('test')
        print('Num test samples:', len(test_list_files))
        test_dataset = WhisperDataset(test_list_files)
    
    elif dataset_name == 'vlsp2019':
        # Download VLSP2019 dataset
        print('Loading VLSP2019 dataset...')

        if not test:
            train_list_files = get_list_files_vlsp2019('train')
            print('Num train samples:', len(train_list_files))
            train_dataset = WhisperDataset(train_list_files)

        test_list_files = get_list_files_vlsp2019('test')
        print('Num test samples:', len(test_list_files))
        test_dataset = WhisperDataset(test_list_files)

    elif dataset_name == 'vin100h':
        # Download VIN100h dataset
        if not os.path.exists('downloaded_check.txt'):
            print('Loading VIN100h dataset...')
            os.system("gdown 1vUSxdORDxk-ePUt-bUVDahpoXiqKchMx")
            os.system("tar -xf 'VinBigdata-VLSP2020-100h (1).rar'")
            os.system("gdown 1Zmj9BqNysiON6Lzjqos9kY08DRanJxXv")
            os.system("unzip 'vin100h_listfiles.zip'")
            os.system("remove 'VinBigdata-VLSP2020-100h (1).rar'")
            with open('downloaded_check.txt', 'w') as f:
                f.write('True')
        else:
            print('Dataset files already downloaded!')
        if not test:
            train_list_files = get_list_files_vin100h('train')
            print('Num train samples:', len(train_list_files))
            train_dataset = WhisperDataset(train_list_files)

        test_list_files = get_list_files_vin100h('test')
        print('Num test samples:', len(test_list_files))
        test_dataset = WhisperDataset(test_list_files)

    else:
        print(dataset_name, 'is not supported, please try again!')

    return train_dataset, test_dataset

#------------------------------------FLUERS------------------------------------#

def get_list_files_fluers(phase, audio_path = 'fluers/vi_vn/audio', text_max_length=1000, audio_max_sample_length=960000, sample_rate=16000):
    audio_path = os.path.join(audio_path, phase)
    audio_transcript_pair_list = []
    if phase=='train':
        tsv_file = 'fluers/vi_vn/train.tsv'
    elif phase=='dev':
        tsv_file = 'fluers/vi_vn/dev.tsv'
    else:
        tsv_file = 'fluers/vi_vn/test.tsv'
    df = pd.read_table(tsv_file, names=("id", "file_name", "raw_transcription", "transcription", "_", "num_samples", "gender"))
    for index, row in df.iterrows():
        new_path = Path(os.path.join(audio_path, row['file_name']))
        audio_id = row['id']
        text = row['transcription']
        if new_path.exists():
            audio = load_wave(new_path, sample_rate=sample_rate)[0]
            if len(text) > text_max_length or len(audio) > audio_max_sample_length:
                print('skip file:', new_path,'with len text:', len(text), 'and len audio', len(audio))
                continue
            audio_transcript_pair_list.append((audio_id, str(new_path), text))
    return audio_transcript_pair_list



#------------------------------------VLSP2019 ASR Dataset------------------------------------#
def get_list_files_vlsp2019(phase, dataset_path = 'vlsp2019/data', text_max_length=1000, audio_max_sample_length=960000, sample_rate=16000):
    audio_transcript_pair_list = []
    if phase=='train':
      csv_file = 'vlsp2019/vlsp2019_train.csv'
    else:
      csv_file = 'vlsp2019/vlsp2019_test.csv'
    df = pd.read_csv(csv_file)
    for index, row in df.iterrows():
        new_path = Path(os.path.join(dataset_path, row['filename']+'.wav'))
        audio_id = index
        with open(Path(os.path.join(dataset_path, row['filename']+'.txt')), 'r') as f:
          text = f.readlines()[0]
        if new_path.exists():
            audio = load_wave(new_path, sample_rate=sample_rate)[0]
            if len(text) > text_max_length or len(audio) > audio_max_sample_length:
                print('skip file:', new_path,'with len text:', len(text), 'and len audio', len(audio))
                continue
            audio_transcript_pair_list.append((audio_id, str(new_path), text))
    return audio_transcript_pair_list

#------------------------------------VIN100h ASR Dataset------------------------------------#
def get_list_files_vin100h(phase, dataset_path = 'vlsp2020_train_set_02', text_max_length=1000, audio_max_sample_length=960000, sample_rate=16000):
    audio_transcript_pair_list = []
    if phase=='train':
      csv_file = 'train_vin100h.csv'
    else:
      csv_file = 'test_vin100h.csv'
    df = pd.read_csv(csv_file)
    for index, row in df.iterrows():
        new_path = Path(os.path.join(dataset_path, row['filename']+'.wav'))
        audio_id = index
        with open(Path(os.path.join(dataset_path, row['filename']+'.txt')), 'r') as f:
          text = f.readlines()[0]
        if new_path.exists():
            audio = load_wave(new_path, sample_rate=sample_rate)[0]
            if len(text) > text_max_length or len(audio) > audio_max_sample_length:
                print('skip file:', new_path,'with len text:', len(text), 'and len audio', len(audio))
                continue
            audio_transcript_pair_list.append((audio_id, str(new_path), text))
    return audio_transcript_pair_list

if __name__=='__main__':
    # load_fluers()
    print('Load dataset...')
