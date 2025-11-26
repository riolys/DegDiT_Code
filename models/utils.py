import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

import torchaudio
import random
import itertools
import numpy as np
from mix import mix


def uncapitalize(s):
    if s:
        return s[:1].lower() + s[1:]
    else:
        return ""


def normalize_wav(waveform):
    waveform = waveform - torch.mean(waveform)
    waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
    return waveform * 0.5


def pad_wav(waveform, segment_length):
    waveform_length = len(waveform)

    if segment_length is None or waveform_length == segment_length:
        return waveform
    elif waveform_length > segment_length:
        return waveform[:segment_length]
    else:
        padded_wav = torch.zeros(segment_length - waveform_length).to(waveform.device)
        waveform = torch.cat([waveform, padded_wav])
        return waveform

def augment_wav(audio_input, texts, sample_rate=44100, num_items=4):

    mixed_sounds, mixed_captions, durations = [], [], []
    combinations = list(itertools.combinations(list(range(len(texts))), 2)) 
    random.shuffle(combinations)
    if len(combinations) < num_items:
        selected_combinations = combinations
    else:
        selected_combinations = combinations[:num_items]
    
    for (i, j) in selected_combinations:
        #print("1111", audio_input[i].size(), audio_input[j].size())
        mixed_sound = mix(audio_input[i].cpu().numpy(), audio_input[j].cpu().numpy(), 0.5, sample_rate).reshape(2, -1) 
        #print("mixed_sound", np.shape(mixed_sound))
        mixed_caption = "{} and {}".format(texts[i], uncapitalize(texts[j]))
        dur = len(mixed_sound[0])/sample_rate
        mixed_sounds.append(mixed_sound)
        mixed_captions.append(mixed_caption)
        durations.append(dur)
    
    #waveform = torch.tensor(np.concatenate(mixed_sounds, 0)) 
    waveform = torch.tensor(np.stack(mixed_sounds, axis=0))
    #print("waveform", waveform.size())
    waveform = waveform / torch.max(torch.abs(waveform))
    waveform = 0.5 * waveform
    
    return waveform, mixed_captions, durations


def read_wav_file(filename, duration_sec, tgt_sample_rate):
    import os
    filename = os.path.join("experiments/PicoAudio/picoaudio", filename)
    info = torchaudio.info(filename)
    sample_rate = info.sample_rate

    # Calculate the number of frames corresponding to the desired duration
    num_frames = int(sample_rate * duration_sec)

    waveform, sr = torchaudio.load(filename, num_frames=num_frames)  # Faster!!!

    if waveform.shape[0] == 2:  ## Stereo audio
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=tgt_sample_rate)
        resampled_waveform = resampler(waveform)
        # print(resampled_waveform.shape)
        padded_left = pad_wav(resampled_waveform[0], int(tgt_sample_rate * duration_sec))  ## We pad left and right seperately
        padded_right = pad_wav(resampled_waveform[1], int(tgt_sample_rate * duration_sec))

        padded_left = normalize_wav(padded_left)
        padded_right = normalize_wav(padded_right)

        return torch.stack([padded_left, padded_right])
    else:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=tgt_sample_rate)[0]
        waveform = pad_wav(waveform, int(tgt_sample_rate * duration_sec)).unsqueeze(0)
        waveform = normalize_wav(waveform)

        return waveform


class DPOText2AudioDataset(Dataset):
    def __init__(
        self,
        dataset,
        prefix,
        text_column,
        audio_w_column,
        audio_l_column,
        duration,
        num_examples=-1,
    ):

        inputs = list(dataset[text_column])
        self.inputs = [prefix + inp for inp in inputs]
        self.audios_w = list(dataset[audio_w_column])
        self.audios_l = list(dataset[audio_l_column])
        self.durations = list(dataset[duration])
        self.indices = list(range(len(self.inputs)))

        self.mapper = {}
        for index, audio_w, audio_l, duration, text in zip(
            self.indices, self.audios_w, self.audios_l, self.durations, inputs
        ):
            self.mapper[index] = [audio_w, audio_l, duration, text]

        if num_examples != -1:
            self.inputs, self.audios_w, self.audios_l, self.durations = (
                self.inputs[:num_examples],
                self.audios_w[:num_examples],
                self.audios_l[:num_examples],
                self.durations[:num_examples],
            )
            self.indices = self.indices[:num_examples]

    def __len__(self):
        return len(self.inputs)

    def get_num_instances(self):
        return len(self.inputs)

    def __getitem__(self, index):
        s1, s2, s3, s4, s5 = (
            self.inputs[index],
            self.audios_w[index],
            self.audios_l[index],
            self.durations[index],
            self.indices[index],
        )
        return s1, s2, s3, s4, s5

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [dat[i].tolist() for i in dat]


class Text2AudioDataset(Dataset):
    def __init__(
        self, dataset, prefix, text_column, audio_column, duration, num_examples=-1
    ):

        inputs = list(dataset[text_column])
        self.inputs = [prefix + inp for inp in inputs]
        self.audios = list(dataset[audio_column])
        self.durations = list(dataset[duration])
        self.indices = list(range(len(self.inputs)))

        self.mapper = {}
        for index, audio, duration, text in zip(
            self.indices, self.audios, self.durations, inputs
        ):
            self.mapper[index] = [audio, text, duration]

        if num_examples != -1:
            self.inputs, self.audios, self.durations = (
                self.inputs[:num_examples],
                self.audios[:num_examples],
                self.durations[:num_examples],
            )
            self.indices = self.indices[:num_examples]

    def __len__(self):
        return len(self.inputs)

    def get_num_instances(self):
        return len(self.inputs)

    def __getitem__(self, index):
        s1, s2, s3, s4 = (
            self.inputs[index],
            self.audios[index],
            self.durations[index],
            self.indices[index],
        )
        # print(s1,s2,s3,s4)
        return s1, s2, s3, s4

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [dat[i].tolist() for i in dat]

class Text2AudioDataset_strong(Dataset):
    def __init__(
        self, dataset, prefix, text_column, audio_column, duration, num_examples=-1
    ):

        inputs = list(dataset[text_column])
        self.inputs = [prefix + inp for inp in inputs]
        self.audios = list(dataset[audio_column])
        self.durations = list(dataset[duration])
        self.indices = list(range(len(self.inputs)))

        self.mapper = {}
        for index, audio, duration, text in zip(
            self.indices, self.audios, self.durations, inputs
        ):
            self.mapper[index] = [audio, text, duration]

        if num_examples != -1:
            self.inputs, self.audios, self.durations = (
                self.inputs[:num_examples],
                self.audios[:num_examples],
                self.durations[:num_examples],
            )
            self.indices = self.indices[:num_examples]

    def __len__(self):
        return len(self.inputs)

    def get_num_instances(self):
        return len(self.inputs)

    def __getitem__(self, index):
        s1, s2, s3, s4 = (
            self.inputs[index],
            "audio_condition/audioset_strong/train/"+self.audios[index].split("/")[-1],
            self.durations[index],
            self.indices[index],
        )
        print(s2)
        return s1, s2, s3, s4

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [dat[i].tolist() for i in dat]
