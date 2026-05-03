"""
Audio Source Separation - ONNX Conversion
Optimized for edge deployment
"""

import torch
import torch.nn as nn
import numpy as np
import os

class SimpleSeparatorModel(nn.Module):
    """Simple frequency-based separator for ONNX"""
    
    def __init__(self, split_freq=1500, sr=16000):
        super().__init__()
        self.split_freq = split_freq
        self.sr = sr
        self.n_fft = 512
        
    def forward(self, magnitude, phase):
        """
        Args:
            magnitude: Spectrogram (batch, freq, time)
            phase: Phase (batch, freq, time)
        Returns:
            voice_mag: Voice magnitude
            bg_mag: Background magnitude
        """
        batch, freq, time = magnitude.shape
        
        # Simple frequency masking
        voice_mag = magnitude.clone()
        bg_mag = magnitude.clone()
        
        # Attenuate high frequencies for voice
        voice_mag[:, freq//4:, :] *= 0.1
        bg_mag[:, :freq//4, :] *= 0.1
        
        return voice_mag, bg_mag

def convert_separator_to_onnx(model_path="separator_model.onnx"):
    """Convert separator model to ONNX"""
    
    print("Creating separator model...")
    model = SimpleSeparatorModel()
    model.eval()
    
    # Dummy inputs
    batch_size = 1
    freq_bins = 257
    time_steps = 100
    
    magnitude = torch.randn(batch_size, freq_bins, time_steps)
    phase = torch.randn(batch_size, freq_bins, time_steps)
    
    print(f"Input: magnitude={magnitude.shape}, phase={phase.shape}")
    
    # Export
    torch.onnx.export(
        model,
        (magnitude, phase),
        model_path,
        input_names=['magnitude', 'phase'],
        output_names=['voice_magnitude', 'bg_magnitude'],
        dynamic_axes={
            'magnitude': {2: 'time'},
            'phase': {2: 'time'},
            'voice_magnitude': {2: 'time'},
            'bg_magnitude': {2: 'time'}
        },
        opset_version=11
    )
    
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"✓ Saved: {model_path} ({size_mb:.4f} MB)")
    
    return model_path

if __name__ == "__main__":
    convert_separator_to_onnx()