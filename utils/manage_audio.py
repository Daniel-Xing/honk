from __future__ import print_function
import argparse
import os
import random
import sys
import wave

import librosa
import numpy as np
import pcen
import pyaudio


def set_speech_format(f):
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(16000)


# MFCC（Mel Frequency Cepstral Coefficients）和 PCEN（Per-Channel Energy Normalization）是音频信号处理中常用的特征表示方法。

# MFCC 是一种用于表示音频语音特征的系数，它在音频信号的频率域和倒谱域中提取了重要的特征。
# MFCC 基于人耳的听觉感知原理，首先将音频信号转换为 Mel 频率刻度，然后计算每个频率带的能量，并在倒谱域中应用离散余弦变换（DCT）以获取最终的系数。
# MFCC 可以很好地表示音频信号的语音内容和语音质量特征，常用于语音识别、说话人识别等任务。


# PCEN 是一种用于增强音频信号的特征表示方法，它在频率域对音频信号进行归一化处理，以减小音频信号中的噪声和背景干扰。
# PCEN 通过计算每个频率带的能量，并对频率带之间的能量进行动态归一化，进而提高信号的对比度和稳定性。
# PCEN 被广泛应用于音频处理领域，特别是在语音识别和音频增强任务中，能够有效降低噪声和背景干扰对信号的影响。
class AudioPreprocessor(object):
    """
    音频预处理器类，用于计算音频的 MFCC 特征和 PCEN 转换。

    参数：
        sr: 采样率，默认为 16000
        n_dct_filters: DCT 滤波器的数量，默认为 40
        n_mels: Mel 频谱的数量，默认为 40
        f_max: 最大频率，默认为 4000
        f_min: 最小频率，默认为 20
        n_fft: FFT 窗口大小，默认为 480
        hop_ms: 帧移时间（以毫秒为单位），默认为 10

    方法：
        compute_mfccs: 计算音频的 MFCC 特征
        compute_pcen: 计算音频的 PCEN 转换
    """

    def __init__(
        self,
        sr=16000,
        n_dct_filters=40,
        n_mels=40,
        f_max=4000,
        f_min=20,
        n_fft=480,
        hop_ms=10,
    ):
        """
        初始化 AudioPreprocessor 对象。

        参数：
            sr: 采样率，默认为 16000
            n_dct_filters: DCT 滤波器的数量，默认为 40
            n_mels: Mel 频谱的数量，默认为 40
            f_max: 最大频率，默认为 4000
            f_min: 最小频率，默认为 20
            n_fft: FFT 窗口大小，默认为 480
            hop_ms: 帧移时间（以毫秒为单位），默认为 10
        """
        super().__init__()
        self.n_mels = n_mels
        self.dct_filters = librosa.filters.dct(n_dct_filters, n_mels)
        self.sr = sr
        self.f_max = f_max if f_max is not None else sr // 2
        self.f_min = f_min
        self.n_fft = n_fft
        self.hop_length = sr // 1000 * hop_ms
        self.pcen_transform = pcen.StreamingPCENTransform(
            n_mels=n_mels, n_fft=n_fft, hop_length=self.hop_length, trainable=True
        )
        # 创建 StreamingPCENTransform 对象，用于计算 PCEN 转换

    def compute_mfccs(self, data):
        """
        计算音频的 MFCC 特征。

        参数：
            data: 音频数据

        返回值：
            音频的 MFCC 特征
        """
        data = librosa.feature.melspectrogram(
            data,
            sr=self.sr,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
            fmin=self.f_min,
            fmax=self.f_max,
        )
        # 使用 librosa 库的 melspectrogram 函数计算音频的 Mel 频谱特征
        # Mel 频谱是一种在频率域中表示音频特征的表示形式

        data[data > 0] = np.log(data[data > 0])
        # 对 Mel 频谱数据中大于 0 的值应用对数变换
        # 这是为了将幅度谱转换为对数刻度，以提高特征的稳定性和表达能力
        # 对数变换可以压缩幅度范围，强调低能量成分，并提高特征的稳定性和表达能力

        data = [
            np.matmul(self.dct_filters, x)
            for x in np.split(data, data.shape[1], axis=1)
        ]
        # 将 Mel 频谱数据按列切分，并对每列应用 DCT 滤波器
        # 这是为了在倒谱域中提取更具代表性的特征
        # DCT 在音频信号处理中常用于提取倒谱系数，代表音频信号的频谱包络信息，并降低数据的维度

        data = np.array(data, order="F").astype(np.float32)
        # 将处理后的 MFCC 特征转换为 NumPy 数组，并设置数组的内存顺序为列优先（Fortran 风格）
        # 这是为了与后续的计算和操作相兼容，提高计算效率

        return data

    def compute_pcen(self, data):
        """
        计算音频的 PCEN 转换。

        参数：
            data: 音频数据

        返回值：
            音频的 PCEN 转换结果
        """
        data = self.pcen_transform(data)
        # 使用预先创建的 PCEN 转换器 self.pcen_transform 对音频数据进行转换
        # PCEN 转换可以对音频信号进行增强，并降低噪声和背景干扰的影响

        self.pcen_transform.reset()
        # 重置 PCEN 转换器，以便在下一次调用 compute_pcen 时重新初始化转换状态
        # 这是为了确保每次调用 compute_pcen 方法时都能获得正确的 PCEN 转换结果，并避免前一次调用的状态对后续调用造成影响

        return data


class AudioSnippet(object):
    _dct_filters = librosa.filters.dct(40, 40)

    def __init__(self, byte_data=b"", dtype=np.int16):
        self.byte_data = byte_data
        self.dtype = dtype
        self._compute_amps()

    def save(self, filename):
        with wave.open(filename, "wb") as f:
            set_speech_format(f)
            f.writeframes(self.byte_data)

    def generate_contrastive(self):
        snippet = self.copy()
        phoneme_chunks = snippet.chunk_phonemes()
        phoneme_chunks2 = snippet.chunk_phonemes(factor=0.8, group_threshold=500)
        joined_chunks = []
        for i in range(len(phoneme_chunks) - 1):
            joined_chunks.append(
                AudioSnippet.join([phoneme_chunks[i], phoneme_chunks[i + 1]])
            )
        if len(joined_chunks) == 1:
            joined_chunks = []
        if len(phoneme_chunks) == 1:
            phoneme_chunks = []
        if len(phoneme_chunks2) == 1:
            phoneme_chunks2 = []
        chunks = [c.copy() for c in phoneme_chunks2]
        for chunk_list in (phoneme_chunks, joined_chunks, phoneme_chunks2):
            for chunk in chunk_list:
                chunk.rand_pad(32000)
        for chunk in chunks:
            chunk.repeat_fill(32000)
            chunk.rand_pad(32000)
        chunks.extend(phoneme_chunks)
        chunks.extend(phoneme_chunks2)
        chunks.extend(joined_chunks)
        return chunks

    def chunk_phonemes(self, factor=1.0, group_threshold=1000):
        audio_data, _ = librosa.effects.trim(self.amplitudes, top_db=16)
        data = librosa.feature.melspectrogram(
            audio_data,
            sr=16000,
            n_mels=40,
            hop_length=160,
            n_fft=480,
            fmin=20,
            fmax=4000,
        )
        data[data > 0] = np.log(data[data > 0])
        data = [
            np.matmul(AudioSnippet._dct_filters, x)
            for x in np.split(data, data.shape[1], axis=1)
        ]
        data = np.array(data, order="F").squeeze(2).astype(np.float32)
        data = data[:, 1:25]
        a = []
        for i in range(data.shape[0] - 1):
            a.append(np.linalg.norm(data[i] - data[i + 1]))
        a = np.array(a)
        q75, q25 = np.percentile(a, [75, 25])
        segments = 160 * np.arange(a.shape[0])[a > q75 + factor * (q75 - q25)]
        segments = np.append(segments, [len(audio_data)])
        delete_idx = []
        for i in range(len(segments)):
            for j in range(i + 1, len(segments)):
                if segments[j] - segments[i] < group_threshold:
                    delete_idx.append(j)
                else:
                    i = j - 1
                    break
        segments = np.delete(segments, delete_idx)
        audio_segments = [
            audio_data[segments[i] : segments[i + 1]] for i in range(len(segments) - 1)
        ]
        audio_segments = [AudioSnippet.from_amps(seg) for seg in audio_segments]
        return audio_segments

    @staticmethod
    def join(snippets):
        snippet = AudioSnippet(dtype=snippets[0].dtype)
        for s in snippets:
            snippet.append(s)
        return snippet

    def copy(self):
        return AudioSnippet(self.byte_data)

    def chunk(self, size, stride=1000):
        chunks = []
        i = 0
        while i + size < len(self.byte_data):
            chunks.append(AudioSnippet(self.byte_data[i : i + size]))
            i += stride
        return chunks

    def rand_pad(self, total_length, noise_level=0.001):
        space = total_length - len(self.byte_data)
        len_a = (random.randint(0, space)) // 2 * 2
        len_b = space - len_a
        self.byte_data = b"".join(
            [b"".join([b"\x00"] * len_a), self.byte_data, b"".join([b"\x00"] * len_b)]
        )
        self._compute_amps()

    def repeat_fill(self, length):
        n_times = max(1, length // len(self.byte_data))
        self.byte_data = b"".join([self.byte_data] * n_times)[:length]

    def trim_window(self, window_size):
        nbytes = len(self.byte_data) // len(self.amplitudes)
        window_size //= nbytes
        cc_window = np.ones(window_size)
        clip_energy = np.correlate(np.abs(self.amplitudes), cc_window)
        smooth_window_size = 1000
        smooth_window = np.ones(smooth_window_size)
        scale = len(self.amplitudes) / (len(self.amplitudes) - smooth_window_size + 1)
        clip_energy2 = np.correlate(clip_energy, smooth_window)
        window_i = int(np.argmax(clip_energy2) * scale)
        window_i = max(0, window_i - window_i % nbytes)
        self.amplitudes = self.amplitudes[window_i : window_i + window_size]
        window_i *= nbytes
        self.byte_data = self.byte_data[window_i : window_i + window_size * nbytes]

    def ltrim(self, limit=0.1):
        if not self.byte_data:
            return
        i = 0
        for i in range(len(self.amplitudes)):
            if self.amplitudes[i] > limit:
                break
        nbytes = len(self.byte_data) // len(self.amplitudes)
        i = max(0, i - i % nbytes)
        self.amplitudes = self.amplitudes[i:]
        self.byte_data = self.byte_data[i * nbytes :]
        return self

    def trim(self, limit=0.1):
        self.ltrim(limit)
        self.rtrim(limit)
        return self

    def rtrim(self, limit=0.05):
        if not self.byte_data:
            return
        i = len(self.amplitudes)
        for i in range(len(self.amplitudes) - 1, -1, -1):
            if self.amplitudes[i] > limit:
                break
        nbytes = len(self.byte_data) // len(self.amplitudes)
        i = min(len(self.amplitudes), i + (nbytes - i % nbytes))
        self.amplitudes = self.amplitudes[:i]
        self.byte_data = self.byte_data[: i * nbytes]
        return self

    @classmethod
    def from_amps(cls, amps, dtype=np.int16):
        byte_data = (np.iinfo(dtype).max * amps).astype(dtype).tobytes()
        return cls(byte_data)

    def _compute_amps(self):
        self.amplitudes = (
            np.frombuffer(self.byte_data, self.dtype).astype(float)
            / np.iinfo(self.dtype).max
        )

    def append(self, snippet):
        self.byte_data = b"".join([self.byte_data, snippet.byte_data])
        self._compute_amps()
        return self

    def amplitude_rms(self, start=0, end=-1):
        return np.sqrt(np.mean([a * a for a in self.amplitudes[start:end]]))


class AudioSnippetGenerator(object):
    def __init__(self, sr=16000, fmt=pyaudio.paInt16, chunk_size=1024, channels=1):
        self.sr = sr
        self.fmt = fmt
        self.channels = channels
        self.chunk_size = chunk_size
        self.stream = None

    def __enter__(self):
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=self.fmt,
            channels=self.channels,
            rate=self.sr,
            input=True,
            frames_per_buffer=self.chunk_size,
        )
        return self

    def __exit__(self, *args):
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()
        self.stream = None

    def __iter__(self):
        if not self.stream:
            raise ValueError("Audio stream isn't open")
        return self

    def __next__(self):
        return AudioSnippet(self.stream.read(self.chunk_size))


def print_sound_level():
    with AudioSnippetGenerator() as generator:
        for audio in generator:
            print("Sound level: {}".format(audio.amplitude_rms()), end="\r")


def generate_dir(directory):
    for filename in os.listdir(directory):
        fullpath = os.path.join(os.path.abspath(directory), filename)
        try:
            with wave.open(fullpath) as f:
                n_channels = f.getnchannels()
                width = f.getsampwidth()
                rate = f.getframerate()
                snippet = AudioSnippet(f.readframes(16000))
            for i, e in enumerate(snippet.generate_contrastive()):
                gen_name = os.path.join(directory, "gen-{}-{}".format(i, filename))
                e.save(gen_name)
            print("Generated from {}".format(filename))
        except (wave.Error, IsADirectoryError, PermissionError) as e:
            pass


def clean_dir(directory=".", cutoff_ms=1000):
    """Trims all audio in directory to the loudest window of length cutoff_ms. 1 second is consistent
    with the speech command dataset.

    Args:
        directory: The directory containing all the .wav files. Should have nothing but .wav
        cutoff_ms: The length of time to trim audio to in milliseconds
    """
    for filename in os.listdir(directory):
        fullpath = os.path.join(directory, filename)
        try:
            with wave.open(fullpath) as f:
                n_channels = f.getnchannels()
                width = f.getsampwidth()
                rate = f.getframerate()
                n_samples = int((cutoff_ms / 1000) * rate)
                snippet = AudioSnippet(f.readframes(10 * n_samples))
            snippet.trim_window(n_samples * width)
            with wave.open(fullpath, "w") as f:
                f.setnchannels(n_channels)
                f.setsampwidth(width)
                f.setframerate(rate)
                f.writeframes(snippet.byte_data)
            print("Trimmed {} to {} ms".format(filename, cutoff_ms))
        except (wave.Error, IsADirectoryError, PermissionError) as e:
            pass


def main():
    parser = argparse.ArgumentParser()
    commands = dict(trim=clean_dir, listen=print_sound_level)
    commands["generate-contrastive"] = generate_dir
    parser.add_argument("subcommand")

    def print_sub_commands():
        print("Subcommands: {}".format(", ".join(commands.keys())))

    if len(sys.argv) <= 1:
        print_sub_commands()
        return
    subcommand = sys.argv[1]
    if subcommand == "generate-contrastive":
        parser.add_argument(
            "directory",
            type=str,
            default=".",
            help="Generate from the directory's audio files",
        )
        flags, _ = parser.parse_known_args()
        generate_dir(flags.directory)
    elif subcommand == "trim":
        parser.add_argument(
            "directory",
            type=str,
            nargs="?",
            default=".",
            help="Trim the directory's audio files",
        )
        flags, _ = parser.parse_known_args()
        clean_dir(flags.directory)
    elif subcommand == "listen":
        print_sound_level()
    else:
        print_sub_commands()


if __name__ == "__main__":
    main()
