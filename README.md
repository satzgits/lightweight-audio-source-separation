# Lightweight Audio Source Separation

A simplified version of audio stem separation (like FL Studio) optimized for edge devices. Separates mixed audio into voice and background components.

## 🎯 What It Does

- Separates voice from background noise
- Works like FL Studio stem separation
- Optimized for edge/CPU deployment
- Shows understanding of frequency domain processing

## 🧠 Why It Matters

Source separation is used in:
- Music production (FL Studio, iZotope)
- Voice enhancement (isolate speech)
- Noise removal
- Audio editing

## ⚡ Key Features

- Spectrogram-based processing
- Frequency masking
- Lightweight architecture
- CPU-friendly

## 📦 Installation

```bash
pip install -r requirements.txt
```

Or:
```bash
pip install torch torchaudio librosa soundfile numpy matplotlib
```

## 🚀 Usage

### Jupyter Notebook (Recommended)
```bash
jupyter notebook notebook_separation.ipynb
```

### Python Script
```bash
python scripts/separate_audio.py --input input_audio/mixed.wav
```

### ONNX Conversion
```bash
python scripts/convert_to_onnx.py
```

## 📊 Performance

| Metric | Full Model | Optimized |
|--------|-----------|-----------|
| Model Size | ~200 MB | ~5 MB |
| Parameters | 50M+ | ~100K |
| Platform | GPU | CPU |
| Speed | Real-time | Real-time |

## 🔁 Pipeline

```
Mixed Audio → STFT → Spectrogram → Masking → ISTFT → Voice + Background
                    ↓
            [Voice Mask] [Background Mask]
```

## 📁 Project Structure

```
lightweight-audio-source-separation/
├── notebook_separation.ipynb        # Demo notebook
├── README.md                    # This file
├── requirements.txt            # Dependencies
└── scripts/
    ├── separate_audio.py        # Main separation
    ├── convert_to_onnx.py        # ONNX conversion
    └── optimize_model.py        # Model optimization

## 🎓 Skills Demonstrated

- FFT/STFT signal processing
- Frequency domain understanding
- Audio/source separation
- Model optimization

## 🔗 Related Projects

- [Edge AI Speech Enhancement](../edge-ai-speech-enhancement/)
- [Real-Time VAD](../real-time-voice-activity-detector/)

Together these show a complete audio AI pipeline!

---

*Built for IPHIPI Technologies internship preparation*