# Lightweight Audio Source Separation

Audio source separation (voice/background) optimized for edge devices - like FL Studio stem separation.

## What It Does

- Separates voice from background music/noise
- Works like FL Studio stem separation
- Optimized for edge/CPU deployment
- Frequency domain processing

## Why It Matters

Source separation is used in:
- Music production (FL Studio, iZotope)
- Voice enhancement (isolate speech)
- Noise removal
- Audio editing
- Video conferencing

## Key Features

- Spectrogram-based processing
- Frequency masking
- Lightweight architecture
- CPU-friendly inference
- ONNX optimization

## Installation

```bash
pip install -r requirements.txt
```

Or:
```bash
pip install torch torchaudio librosa soundfile numpy matplotlib
```

## Usage

### Jupyter Notebook (Recommended)
```bash
jupyter notebook notebook_separation.ipynb
```

### Python Script
```bash
python scripts/separate_audio.py --input input_audio/mixed.wav --voice output_audio/voice.wav --background output_audio/background.wav
```

### ONNX Conversion
```bash
python scripts/convert_to_onnx.py
```

## Performance

| Metric | Full Model | Optimized |
|--------|-----------|-----------|
| Model Size | ~200 MB | ~5 MB |
| Parameters | 50M+ | ~100K |
| Platform | GPU | CPU |
| Speed | Real-time | Real-time |

## Pipeline

```
Mixed Audio → STFT → Spectrogram → Frequency Masking → ISTFT → Voice + Background
```

## Project Structure

```
lightweight-audio-source-separation/
├── notebook_separation.ipynb      # Demo notebook
├── README.md                    # This file
├── requirements.txt             # Dependencies
├── LICENSE                     # MIT License
└── scripts/
    ├── separate_audio.py       # Main separation
    ├── convert_to_onnx.py     # ONNX conversion
    └── optimize_model.py      # Model optimization
```

## Skills Demonstrated

- FFT/STFT signal processing
- Frequency domain understanding
- Audio/source separation
- Model optimization
- Edge deployment

## Related Projects

- [Edge AI Speech Enhancement](../edge-ai-speech-enhancement/)
- [Real-Time VAD](../real-time-voice-activity-detector/)

Together these show a complete audio AI pipeline!

---


*GitHub: github.com/satzgits/lightweight-audio-source-separation*
