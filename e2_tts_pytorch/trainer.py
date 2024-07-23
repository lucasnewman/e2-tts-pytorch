from __future__ import annotations

import json
import os
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import LinearLR, SequentialLR
# from torch.utils.tensorboard import SummaryWriter

import torchaudio

from einops import rearrange
from accelerate import Accelerator

import matplotlib.pyplot as plt

# from loguru import logger

from e2_tts_pytorch.e2_tts import (
    E2TTS,
    DurationPredictor,
    MelSpec
)

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# collation

def collate_fn(batch):
    mel_specs = [item['mel_spec'].squeeze(0) for item in batch]
    mel_lengths = torch.LongTensor([spec.shape[-1] for spec in mel_specs])
    max_mel_length = mel_lengths.amax()

    padded_mel_specs = []
    for spec in mel_specs:
        padding = (0, max_mel_length - spec.size(-1))
        padded_spec = F.pad(spec, padding, value = 0)
        padded_mel_specs.append(padded_spec)
    
    mel_specs = torch.stack(padded_mel_specs)

    text = [item['text'] for item in batch]
    text_lengths = torch.LongTensor([len(item) for item in text])

    return dict(
        mel = mel_specs,
        mel_lengths = mel_lengths,
        text = text,
        text_lengths = text_lengths,
    )

# dataset

class HFDataset(Dataset):
    def __init__(
        self,
        hf_dataset: Dataset,
        target_sample_rate = 22050,
        hop_length = 256
    ):
        self.data = hf_dataset
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        self.mel_spectrogram = MelSpec(sampling_rate=target_sample_rate)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        row = self.data[index]
        audio = row['audio']['array']

        print(f"Audio shape: {audio.shape}")

        sample_rate = row['audio']['sampling_rate']
        duration = audio.shape[-1] / sample_rate

        if duration > 20 or duration < 0.3:
            print(f"Skipping due to duration out of bound: {duration}")
            return self.__getitem__((index + 1) % len(self.data))
        
        audio_tensor = torch.from_numpy(audio).float()
        
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
            audio_tensor = resampler(audio_tensor)
        
        audio_tensor = rearrange(audio_tensor, 't -> 1 t')
        
        mel_spec = self.mel_spectrogram(audio_tensor)
        
        mel_spec = rearrange(mel_spec, '1 d t -> d t')
        
        text = row['transcript']
        
        return dict(
            mel_spec = mel_spec,
            text = text,
        )


class TextAudioDataset(Dataset):
    def __init__(
        self,
        folder,
        audio_extension = ".wav",
        target_sample_rate = 24_000,
        min_duration = 0.3,
        max_duration = 10
    ):
        super().__init__()
        path = Path(folder)
        assert path.exists(), 'folder does not exist'

        self.audio_extension = audio_extension

        files = list(path.glob(f'**/*{audio_extension}'))
        assert len(files) > 0, 'no files found'
        
        valid_files = []
        
        if os.path.exists(f'valid_files_{int(max_duration)}.json'):
            with open(f'valid_files_{int(max_duration)}.json', 'r') as f:
                valid_files = json.load(f)
                valid_files = [Path(file) for file in valid_files]
                print(f"Loaded {len(valid_files)} files.")
        else:
            for file in tqdm(files):
                audio, sample_rate = torchaudio.load(file)
                duration = audio.shape[-1] / sample_rate

                if duration > max_duration or duration < min_duration:
                    pass
                    # print(f"Skipping due to duration out of bound: {duration}")
                else:
                    valid_files.append(file)
        
        files = valid_files
        print(f"Using {len(files)} files.")
        
        # save the valid file listing to json
        with open(f'valid_files_{int(max_duration)}.json', 'w') as f:
            json.dump([str(file) for file in files], f)
        
        self.files = files
        self.target_sample_rate = target_sample_rate
        self.mel_spectrogram = MelSpec(
            sampling_rate = 24_000,
            filter_length = 1024,
            hop_length = 256,
            win_length = 1024,
            n_mel_channels = 100
        )
        
        # preprocess the data as needed
        text = []
        for file in tqdm(files):
            self.save_melspec(file)
            
            text_file = file.with_suffix('.normalized.txt')
            assert text_file.exists(), f'text file {text_file} does not exist'
            text.append(text_file.read_text().strip())
            
        self.text = text
        
    def save_melspec(self, file):
        mel_file = file.with_suffix('.mel')
        if not mel_file.exists():
            audio_tensor, sample_rate = torchaudio.load(file)
            # print(f"audio_tensor: {audio_tensor.shape}")
            
            if sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
                audio_tensor = resampler(audio_tensor)
            
            mel_spec = self.mel_spectrogram(audio_tensor)
            mel_spec = rearrange(mel_spec, '1 d t -> d t')
            # print(f"mel_spec: {mel_spec.shape}")
            
            torch.save(mel_spec, mel_file)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]

        # audio_tensor, sample_rate = torchaudio.load(file)
        # # audio_tensor = rearrange(audio_tensor, '1 ... -> ...')
        
        # if sample_rate != self.target_sample_rate:
        #     resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
        #     audio_tensor = resampler(audio_tensor)
        
        # mel_spec = self.mel_spectrogram(audio_tensor)
        # mel_spec = rearrange(mel_spec, '1 d t -> d t')
        
        mel_spec = torch.load(file.with_suffix('.mel'))
        
        # load the text file with .normalized.txt as the extension
        # text_file = file.with_suffix('.normalized.txt')
        # text = text_file.read_text().strip()

        return dict(
            mel_spec = mel_spec,
            text = self.text[idx],
        )

# trainer

class E2Trainer:
    def __init__(
        self,
        model: E2TTS,
        optimizer,
        num_warmup_steps=20000,
        duration_predictor: DurationPredictor | None = None,
        checkpoint_path = None,
        log_file = "logs.txt",
        max_grad_norm = 1.0,
        sample_rate = 24_000,
        tensorboard_log_dir = 'runs/e2_tts_experiment',
        accelerate_kwargs: dict = dict()
    ):
        # logger.add(log_file)

        self.accelerator = Accelerator(
            log_with="all",
            **accelerate_kwargs
        )

        self.target_sample_rate = sample_rate
        self.model = model
        self.duration_predictor = duration_predictor
        self.optimizer = optimizer
        self.num_warmup_steps = num_warmup_steps
        self.checkpoint_path = default(checkpoint_path, 'model.pth')
        self.mel_spectrogram = MelSpec(sampling_rate=self.target_sample_rate)
        self.model, self.optimizer = self.accelerator.prepare(
            self.model, self.optimizer
        )
        self.max_grad_norm = max_grad_norm
        
        # self.writer = SummaryWriter(log_dir=tensorboard_log_dir)

    def save_checkpoint(self, step, finetune=False):
        checkpoint = dict(
            model_state_dict = self.accelerator.unwrap_model(self.model).state_dict(),
            optimizer_state_dict = self.optimizer.state_dict(),
            scheduler_state_dict = self.scheduler.state_dict(),
            step = step
        )

        torch.save(checkpoint, f"e2tts_{step}.pt")

    def load_checkpoint(self):
        if not exists(self.checkpoint_path) or not os.path.exists(self.checkpoint_path):
            return 0

        checkpoint = torch.load(self.checkpoint_path)
        self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['step']

    def train(self, train_dataset, epochs, batch_size, grad_accumulation_steps = 1, num_workers = 0, save_step = 1000):
        # (todo) gradient accumulation needs to be accounted for

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, num_workers=num_workers, pin_memory=True)
        total_steps = len(train_dataloader) * epochs
        decay_steps = total_steps - self.num_warmup_steps
        warmup_scheduler = LinearLR(self.optimizer, start_factor=1e-8, end_factor=1.0, total_iters=self.num_warmup_steps)
        decay_scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=1e-8, total_iters=decay_steps)
        self.scheduler = SequentialLR(self.optimizer, 
                                      schedulers=[warmup_scheduler, decay_scheduler],
                                      milestones=[self.num_warmup_steps])
        train_dataloader, self.scheduler = self.accelerator.prepare(train_dataloader, self.scheduler)
        start_step = 0 # self.load_checkpoint()
        global_step = start_step

        for epoch in range(epochs):
            self.model.train()
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}", unit="step", disable=not self.accelerator.is_local_main_process)
            epoch_loss = 0.0

            for batch in progress_bar:
                text_inputs = batch['text']
                mel_spec = rearrange(batch['mel'], 'b d n -> b n d')
                mel_lengths = batch["mel_lengths"]
                
                if self.duration_predictor is not None:
                    dur_loss = self.duration_predictor(mel_spec, target_duration=batch.get('durations'))
                    # self.writer.add_scalar('duration loss', dur_loss.item(), global_step)
                
                loss, pred = self.model(mel_spec, text=text_inputs, lens=mel_lengths)
                self.accelerator.backward(loss)

                if self.max_grad_norm > 0:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                # if self.accelerator.is_local_main_process:
                    # print(f"step {global_step+1}: loss = {loss.item():.4f}")
                    # self.writer.add_scalar('loss', loss.item(), global_step)
                    # self.writer.add_scalar("lr", self.scheduler.get_last_lr()[0], global_step)
                
                if global_step % 100 == 0:
                    predicted = rearrange(pred[0, :].cpu(), 'n d -> d n')
                    
                    # visualize the mel spectrogram
                    plt.figure(figsize=(12, 4))
                    plt.imshow(predicted.numpy(), origin='lower', aspect='auto')
                    plt.colorbar()
                    plt.show()
                    
                    expected = rearrange(mel_spec[0, :].cpu(), 'n d -> d n')
                    
                    # visualize the mel spectrogram
                    plt.figure(figsize=(12, 4))
                    plt.imshow(expected.numpy(), origin='lower', aspect='auto')
                    plt.colorbar()
                    plt.show()
                
                global_step += 1
                epoch_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())
                
                if global_step % save_step == 0:
                    self.save_checkpoint(global_step)
            
            epoch_loss /= len(train_dataloader)
            if self.accelerator.is_local_main_process:
                print(f"epoch {epoch+1}/{epochs} - average loss = {epoch_loss:.4f}")
                # self.writer.add_scalar('epoch average loss', epoch_loss, epoch)
        
        # self.writer.close()
