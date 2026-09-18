import numpy as np
from scipy import signal as sp_signal


def bandpass_filter(sig: np.ndarray, fs: float = 360.0,
                    lo: float = 0.5, hi: float = 40.0) -> np.ndarray:
    """Butterworth bandpass filter for ECG baseline wander + noise removal."""
    nyq = fs / 2.0
    b, a = sp_signal.butter(4, [lo / nyq, hi / nyq], btype="band")
    return sp_signal.filtfilt(b, a, sig.astype(np.float64))


def normalize(sig: np.ndarray) -> np.ndarray:
    """Z-score normalization per segment."""
    std = sig.std()
    if std < 1e-8:
        return sig - sig.mean()
    return (sig - sig.mean()) / std


def preprocess_signal(raw: np.ndarray, fs: float = 360.0,
                      target_len: int = 3600) -> np.ndarray:
    """
    Full preprocessing pipeline:
      raw (1, N) or (N,) int16  →  float32 (1, target_len)
    """
    sig = raw.flatten().astype(np.float64)
    sig = bandpass_filter(sig, fs=fs)
    sig = normalize(sig)
    # Pad or crop to fixed length
    if len(sig) < target_len:
        sig = np.pad(sig, (0, target_len - len(sig)))
    else:
        sig = sig[:target_len]
    return sig.astype(np.float32).reshape(1, target_len)


def compute_saliency(model, x_tensor, target_class: int, device) -> np.ndarray:
    """
    Gradient-based temporal saliency: which time-steps drove the prediction.
    Returns array of shape (3600,) with absolute gradient magnitudes.
    """
    model.eval()
    x = x_tensor.clone().detach().to(device).unsqueeze(0).requires_grad_(True)
    out = model(x)
    score = out[0, target_class]
    score.backward()
    saliency = x.grad.data.abs().squeeze().cpu().numpy()
    return saliency
