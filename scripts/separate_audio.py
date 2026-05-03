import numpy as np
import librosa
import soundfile as sf
import argparse
from pathlib import Path
import torch
import torch.nn as nn

class SimpleSeparationNet(nn.Module):
    """Simplified UNet-like architecture for audio separation."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 2, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        return dec

def create_spectrogram(audio, sr=16000, n_fft=512, hop_length=128):
    """Convert audio to spectrogram."""
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    return magnitude, phase, stft

def apply_masks(magnitude, mask_voice, mask_bg):
    """Apply masks to separate sources."""
    voice_spec = magnitude * mask_voice
    bg_spec = magnitude * mask_bg
    return voice_spec, bg_spec

def separate_audio_simple(magnitude, phase):
    """
    Simple frequency-based separation.
    In production: use trained model.
    """
    # Simple frequency-based masking
    # Low frequencies = voice dominant
    # High frequencies = background dominant
    
    rows, cols = magnitude.shape
    mask_voice = np.ones_like(magnitude)
    mask_bg = np.ones_like(magnitude)
    
    freq_threshold = rows // 4
    
    # Attenuate high frequencies for voice
    for i in range(freq_threshold, rows):
        mask_voice[i, :] = 0.3
    
    # Attenuate low frequencies for background
    for i in range(freq_threshold):
        mask_bg[i, :] = 0.2
    
    return mask_voice, mask_bg

def spectrogram_to_audio(magnitude, phase, sr=16000, hop_length=128):
    """Convert spectrogram back to audio."""
    complex_spec = magnitude * np.exp(1j * phase)
    audio = librosa.istft(complex_spec, hop_length=hop_length)
    return audio

def separate(input_path, output_voice=None, output_bg=None):
    """Main separation pipeline."""
    print(f"Loading audio from: {input_path}")
    
    audio, sr = librosa.load(input_path, sr=16000)
    print(f"Audio: {len(audio)/sr:.2f}s at {sr}Hz")
    
    print("Creating spectrogram...")
    magnitude, phase, stft = create_spectrogram(audio, sr)
    print(f"Spectrogram shape: {magnitude.shape}")
    
    print("Separating...")
    mask_voice, mask_bg = separate_audio_simple(magnitude, phase)
    
    print("Applying masks...")
    voice_mag = magnitude * mask_voice
    bg_mag = magnitude * mask_bg
    
    print("Converting to audio...")
    voice_audio = spectrogram_to_audio(voice_mag, phase, sr)
    bg_audio = spectrogram_to_audio(bg_mag, phase, sr)
    
    if output_voice:
        sf.write(output_voice, voice_audio, sr)
        print(f"Voice saved: {output_voice}")
    
    if output_bg:
        sf.write(output_bg, bg_audio, sr)
        print(f"Background saved: {output_bg}")
    
    return voice_audio, bg_audio, sr

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Audio Source Separation')
    parser.add_argument('--input', type=str, default='input_audio/mixed.wav',
                        help='Input mixed audio')
    parser.add_argument('--voice', type=str, default='output_audio/voice.wav',
                        help='Output voice track')
    parser.add_argument('--background', type=str, default='output_audio/background.wav',
                        help='Output background track')
    args = parser.parse_args()
    
    separate(args.input, args.voice, args.background)