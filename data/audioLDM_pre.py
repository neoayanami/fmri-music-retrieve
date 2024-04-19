import os
from librosa.util import pad_center, tiny
from scipy.signal import get_window
import librosa.util as librosa_util
import torch 
import numpy as np
from librosa.filters import mel as librosa_mel_fn
import torchaudio

sampling_rate = 16000
n_mel_channels = 64
mel_fmin = 0
mel_fmax = 8000
duration = 30.0
device = 'cpu'
filter_length = 1024
hop_length = 160
win_length = 1024 
window = 'hann'
target_length = int(duration * sampling_rate / hop_length)
pad_wav_start_sample = 0



class STFT(torch.nn.Module):
    """adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""

    def __init__(self, filter_length, hop_length, win_length, window="hann"):
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.forward_transform = None
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack(
            [np.real(fourier_basis[:cutoff, :]), np.imag(fourier_basis[:cutoff, :])]
        )

        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(
            np.linalg.pinv(scale * fourier_basis).T[:, None, :]
        )

        if window is not None:
            assert filter_length >= win_length
            # get window and zero center pad it to filter_length
            fft_window = get_window(window, win_length, fftbins=True)
            fft_window = pad_center(fft_window, size=filter_length)
            fft_window = torch.from_numpy(fft_window).float()

            # window the bases
            forward_basis *= fft_window
            inverse_basis *= fft_window

        self.register_buffer("forward_basis", forward_basis.float())
        self.register_buffer("inverse_basis", inverse_basis.float())

    def transform(self, input_data):
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)

        self.num_samples = num_samples

        # similar to librosa, reflect-pad the input
        input_data = input_data.view(num_batches, 1, num_samples)
        input_data = torch.nn.functional.pad(
            input_data.unsqueeze(1),
            (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
            mode="reflect",
        )
        input_data = input_data.squeeze(1)

        forward_transform = torch.nn.functional.conv1d(
            input_data,
            torch.autograd.Variable(self.forward_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0,
        ).cpu()

        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        magnitude = torch.sqrt(real_part**2 + imag_part**2)
        phase = torch.autograd.Variable(torch.atan2(imag_part.data, real_part.data))

        return magnitude, phase

    def inverse(self, magnitude, phase):
        recombine_magnitude_phase = torch.cat(
            [magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=1
        )

        inverse_transform = F.conv_transpose1d(
            recombine_magnitude_phase,
            torch.autograd.Variable(self.inverse_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0,
        )

        if self.window is not None:
            window_sum = window_sumsquare(
                self.window,
                magnitude.size(-1),
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_fft=self.filter_length,
                dtype=np.float32,
            )
            # remove modulation effects
            approx_nonzero_indices = torch.from_numpy(
                np.where(window_sum > tiny(window_sum))[0]
            )
            window_sum = torch.autograd.Variable(
                torch.from_numpy(window_sum), requires_grad=False
            )
            window_sum = window_sum
            inverse_transform[:, :, approx_nonzero_indices] /= window_sum[
                approx_nonzero_indices
            ]

            # scale by hop ratio
            inverse_transform *= float(self.filter_length) / self.hop_length

        inverse_transform = inverse_transform[:, :, int(self.filter_length / 2) :]
        inverse_transform = inverse_transform[:, :, : -int(self.filter_length / 2) :]

        return inverse_transform

    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction



def window_sumsquare(
    window,
    n_frames,
    hop_length,
    win_length,
    n_fft,
    dtype=np.float32,
    norm=None,
):
    """
    # from librosa 0.6
    Compute the sum-square envelope of a window function at a given hop length.

    This is used to estimate modulation effects induced by windowing
    observations in short-time fourier transforms.

    Parameters
    ----------
    window : string, tuple, number, callable, or list-like
        Window specification, as in `get_window`

    n_frames : int > 0
        The number of analysis frames

    hop_length : int > 0
        The number of samples to advance between frames

    win_length : [optional]
        The length of the window function.  By default, this matches `n_fft`.

    n_fft : int > 0
        The length of each analysis frame.

    dtype : np.dtype
        The data type of the output

    Returns
    -------
    wss : np.ndarray, shape=`(n_fft + hop_length * (n_frames - 1))`
        The sum-squared envelope of the window function
    """
    if win_length is None:
        win_length = n_fft

    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)

    # Compute the squared window at the desired length
    win_sq = get_window(window, win_length, fftbins=True)
    win_sq = librosa_util.normalize(win_sq, norm=norm) ** 2
    win_sq = librosa_util.pad_center(win_sq, n_fft)

    # Fill the envelope
    for i in range(n_frames):
        sample = i * hop_length
        x[sample : min(n, sample + n_fft)] += win_sq[: max(0, min(n_fft, n - sample))]
    return x


def griffin_lim(magnitudes, stft_fn, n_iters=30):
    """
    PARAMS
    ------
    magnitudes: spectrogram magnitudes
    stft_fn: STFT class with transform (STFT) and inverse (ISTFT) methods
    """

    angles = np.angle(np.exp(2j * np.pi * np.random.rand(*magnitudes.size())))
    angles = angles.astype(np.float32)
    angles = torch.autograd.Variable(torch.from_numpy(angles))
    signal = stft_fn.inverse(magnitudes, angles).squeeze(1)

    for i in range(n_iters):
        _, angles = stft_fn.transform(signal)
        signal = stft_fn.inverse(magnitudes, angles).squeeze(1)
    return signal


def dynamic_range_compression(x, normalize_fun=torch.log, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return normalize_fun(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C

def spectral_normalize(magnitudes, normalize_fun):
    output = dynamic_range_compression(magnitudes, normalize_fun)
    return output



stft = STFT(
    filter_length=filter_length, 
    hop_length=hop_length, 
    win_length=win_length,
    window=window
).to(device)

# magnitude, phase = stft.transform(audio)
mel_basis = librosa_mel_fn(
            sr=sampling_rate, n_fft=filter_length, n_mels=n_mel_channels, fmin=mel_fmin, fmax=mel_fmax
        )
mel_basis = torch.from_numpy(mel_basis).float()
# magnitudes = magnitude.data
# mel_output = torch.matmul(mel_basis, magnitudes)
# mel_output = spectral_normalize(mel_output, torch.log)


def get_mel_from_wav(audio):
    audio = torch.clip(torch.FloatTensor(audio).unsqueeze(0), -1, 1)
    audio = torch.autograd.Variable(audio, requires_grad=False)

    magnitudes, phases = stft.transform(audio)
    magnitudes = magnitudes.data
    mel_output = torch.matmul(mel_basis, magnitudes)
    mel_output = spectral_normalize(mel_output, torch.log)
    energy = torch.norm(magnitudes, dim=1)

    melspec = torch.squeeze(mel_output, 0).numpy().astype(np.float32)
    magnitudes = torch.squeeze(magnitudes, 0).numpy().astype(np.float32)
    energy = torch.squeeze(energy, 0).numpy().astype(np.float32)
    return melspec, magnitudes, energy

def pad_spec(log_mel_spec):
        n_frames = log_mel_spec.shape[0]
        p = target_length - n_frames
        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            log_mel_spec = m(log_mel_spec)
        elif p < 0:
            log_mel_spec = log_mel_spec[0 : target_length, :]
        if log_mel_spec.size(-1) % 2 != 0:
            log_mel_spec = log_mel_spec[..., :-1]

        return log_mel_spec

def wav_feature_extraction(waveform):
    waveform = waveform[0, ...]
    waveform = torch.FloatTensor(waveform)
    log_mel_spec, stft_extr, energy = get_mel_from_wav(waveform)
    log_mel_spec = torch.FloatTensor(log_mel_spec.T)
    stft_extr = torch.FloatTensor(stft_extr.T)
    log_mel_spec, stft_extr = pad_spec(log_mel_spec), pad_spec(stft_extr)
    return log_mel_spec, stft_extr

def read_audio_file(filename, filename2=None):
    if os.path.exists(filename):
        waveform, random_start = read_wav_file(filename)
    else:
        print(
            'Warning [dataset.py]: The wav path "',
            filename,
            '" is not find in the metadata. Use empty waveform instead.',
        )
        target_length = int(sampling_rate * duration)
        waveform = torch.zeros((1, target_length))
        random_start = 0

    mix_lambda = 0.0
    log_mel_spec, stft = wav_feature_extraction(waveform)

    return log_mel_spec, stft, mix_lambda, waveform, random_start

def random_uniform(start, end):
        val = torch.rand(1).item()
        return start + (end - start) * val

def resample(waveform, sr):
        waveform = torchaudio.functional.resample(waveform, sr, sampling_rate)
        return waveform

def normalize_wav(waveform):
        waveform = waveform - np.mean(waveform)
        waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
        return waveform * 0.5  # Manually limit the maximum amplitude into 0.5

def random_segment_wav(waveform, target_length):
        waveform_length = waveform.shape[-1]
        assert waveform_length > 100, "Waveform is too short, %s" % waveform_length

        # Too short
        if (waveform_length - target_length) <= 0:
            return waveform, 0

        random_start = int(random_uniform(0, waveform_length - target_length))
        return waveform[:, random_start : random_start + target_length], random_start

def pad_wav(waveform, target_length):
        waveform_length = waveform.shape[-1]
        assert waveform_length > 100, "Waveform is too short, %s" % waveform_length

        if waveform_length == target_length:
            return waveform

        # Pad
        temp_wav = np.zeros((1, target_length), dtype=np.float32)
        if pad_wav_start_sample is None:
            rand_start = int(random_uniform(0, target_length - waveform_length))
        else:
            rand_start = 0

        temp_wav[:, rand_start : rand_start + waveform_length] = waveform
        return temp_wav

def read_wav_file(filename):
        # waveform, sr = librosa.load(filename, sr=None, mono=True) # 4 times slower
        waveform, sr = torchaudio.load(filename)

        waveform, random_start = random_segment_wav(
            waveform, target_length=int(sr * duration)
        )

        waveform = resample(waveform, sr)
        waveform = waveform.numpy()[0, ...]

        waveform = normalize_wav(waveform)
        waveform = waveform[None, ...]
        waveform = pad_wav(
            waveform, target_length=int(sampling_rate * duration)
        )
        return waveform, random_start






