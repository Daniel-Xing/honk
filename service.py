import io
import os
import shutil
import subprocess
import uuid
import threading
import wave

import librosa
import numpy as np
try:
    import torch
    import torch.nn.functional as F
except ImportError:
    pass
try:
    import onnx
    import onnx_caffe2.backend
except ImportError:
    pass

from utils.manage_audio import AudioSnippet, AudioPreprocessor
try:
    import utils.model as model
except ImportError:
    pass

def _softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

class LabelService(object):
    def evaluate(self, speech_dirs, indices=[]):
        dir_labels = {}
        if indices:
            real_labels = [self.labels[i] for i in indices]
        else:
            real_labels = [os.dirname(d) for d in speech_dirs]
        for i, label in enumerate(real_labels):
            if label not in self.labels:
                real_labels[i] = "_unknown_"
            dir_labels[speech_dirs[i]] = real_labels[i]
        accuracy = []
        for folder in speech_dirs:
            for filename in os.listdir(folder):
                fp = os.path.join(folder, filename)
                with wave.open(fp) as f:
                    b_data = f.readframes(16000)
                label, _ = self.label(b_data)
                accuracy.append(int(label == dir_labels[folder]))
        return sum(accuracy) / len(accuracy)                

    def label(self, wav_data):
        raise NotImplementedError

class Caffe2LabelService(LabelService):
    def __init__(self, onnx_filename, labels):
        self.labels = labels
        self.model_filename = onnx_filename
        self.audio_processor = AudioPreprocessor()
        self._graph = onnx.load(onnx_filename)
        self._in_name = self._graph.graph.input[0].name
        self.model = onnx_caffe2.backend.prepare(self._graph)

    def label(self, wav_data):
        wav_data = np.frombuffer(wav_data, dtype=np.int16) / 32768.
        model_in = np.expand_dims(self.audio_processor.compute_mfccs(wav_data).squeeze(2), 0)
        model_in = np.expand_dims(model_in, 0)
        model_in = model_in.astype(np.float32)
        predictions = _softmax(self.model.run({self._in_name: model_in})[0])
        return (self.labels[np.argmax(predictions)], np.max(predictions))

class TorchLabelService(LabelService):
    """
    使用 PyTorch 实现的标签服务类。

    参数：
        model_filename: 模型文件名
        no_cuda: 是否禁用 CUDA，默认为 False
        labels: 训练的标签列表，默认为 ["_silence_", "_unknown_", "command", "random"]

    方法：
        __init__: 初始化 TorchLabelService 对象
        reload: 重新加载模型
        label: 对音频数据进行标签预测
    """

    def __init__(self, model_filename, no_cuda=False, labels=["_silence_", "_unknown_", "command", "random"]):
        """
        初始化 TorchLabelService 对象。

        参数：
            model_filename: 模型文件名
            no_cuda: 是否禁用 CUDA，默认为 False
            labels: 训练的标签列表，默认为 ["_silence_", "_unknown_", "command", "random"]
        """
        self.labels = labels
        self.model_filename = model_filename
        self.no_cuda = no_cuda
        self.audio_processor = AudioPreprocessor()  # 创建音频预处理器对象
        self.reload()  # 加载模型

    def reload(self):
        """
        重新加载模型。
        """
        config = model.find_config(model.ConfigType.CNN_TRAD_POOL2)  # 获取模型配置
        config["n_labels"] = len(self.labels)  # 设置标签数
        self.model = model.SpeechModel(config)  # 创建语音模型对象
        if not self.no_cuda:
            self.model.cuda()  # 将模型移动到 CUDA 上（如果可用）
        self.model.load(self.model_filename)  # 加载模型
        self.model.eval()  # 设置模型为评估模式

    def label(self, wav_data):
        """
        对音频数据进行标签预测。

        参数：
            wav_data: 需要进行标签预测的音频数据

        返回值：
            最可能的标签和对应的概率的元组
        """
        wav_data = np.frombuffer(wav_data, dtype=np.int16) / 32768.  # 将音频数据转换为浮点数
        model_in = torch.from_numpy(self.audio_processor.compute_mfccs(wav_data).squeeze(2)).unsqueeze(0)
        # 计算音频的 MFCC 特征并转换为模型输入的格式
        model_in = torch.autograd.Variable(model_in, requires_grad=False)
        if not self.no_cuda:
            model_in = model_in.cuda()  # 将输入移动到 CUDA 上（如果可用）
        predictions = F.softmax(self.model(model_in).squeeze(0).cpu()).data.numpy()
        # 使用模型进行预测，并将结果转换为概率分布
        return (self.labels[np.argmax(predictions)], np.max(predictions))
        # 返回最可能的标签和对应的概率的元组

def stride(array, stride_size, window_size):
    i = 0
    while i + window_size <= len(array):
        yield array[i:i + window_size]
        i += stride_size

class TrainingService(object):
    def __init__(self, train_script, speech_dataset_path, options):
        """
        初始化训练服务对象。

        参数：
            train_script: 训练脚本的路径
            speech_dataset_path: 语音数据集的路径
            options: 训练选项

        """
        self.train_script = train_script
        self.neg_directory = os.path.join(speech_dataset_path, "random")
        self.pos_directory = os.path.join(speech_dataset_path, "command")
        self.options = options
        self._run_lck = threading.Lock()
        self.script_running = False
        self._create_dirs()

    def _create_dirs(self):
        """
        创建用于存储训练数据的目录。
        """
        if not os.path.exists(self.neg_directory):
            os.makedirs(self.neg_directory)
        if not os.path.exists(self.pos_directory):
            os.makedirs(self.pos_directory)

    def generate_contrastive(self, data):
        """
        生成对比训练样本。

        参数：
            data: 音频数据

        返回值：
            对比训练样本的列表
        """
        snippet = AudioSnippet(data)
        phoneme_chunks = AudioSnippet(data).chunk_phonemes()
        phoneme_chunks2 = AudioSnippet(data).chunk_phonemes(factor=0.8, group_threshold=500)
        joined_chunks = []
        for i in range(len(phoneme_chunks) - 1):
            joined_chunks.append(AudioSnippet.join([phoneme_chunks[i], phoneme_chunks[i + 1]]))
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

    def clear_examples(self, positive=True, tag=""):
        """
        清除训练样本。

        参数：
            positive: 是否清除正例样本，默认为 True
            tag: 样本标签，默认为空

        """
        directory = self.pos_directory if positive else self.neg_directory
        if not tag:
            shutil.rmtree(directory)
            self._create_dirs()
        else:
            for name in os.listdir(directory):
                if name.startswith("{}-".format(tag)):
                    os.unlink(os.path.join(directory, name))

    def write_example(self, wav_data, positive=True, filename=None, tag=""):
        """
        写入训练样本。

        参数：
            wav_data: 音频数据
            positive: 是否为正例样本，默认为 True
            filename: 文件名，默认为 None
            tag: 样本标签，默认为空

        """
        if tag:
            tag = "{}-".format(tag)
        if not filename:
            filename = "{}{}.wav".format(tag, str(uuid.uuid4()))
        directory = self.pos_directory if positive else self.neg_directory
        filename = os.path.join(directory, filename)
        AudioSnippet(wav_data).save(filename)

    def _run_script(self, script, options):
        """
        运行外部脚本。

        参数：
            script: 脚本的路径
            options: 选项参数

        """
        cmd_strs = ["python", script]
        for option, value in options.items():
            cmd_strs.append("--{}={}".format(option, value))
        subprocess.run(cmd_strs)

    def _run_training_script(self, callback):
        """
        运行训练脚本。

        参数：
            callback: 回调函数

        """
        with self._run_lck:
            self.script_running = True
        self._run_script(self.train_script, self.options)
        if callback:
            callback()
        self.script_running = False

    def run_train_script(self, callback=None):
        """
        运行训练脚本。

        参数：
            callback: 训练结束时的回调函数

        返回值：
            是否成功启动训练脚本的标志

        """
        if self.script_running:
            return False
        threading.Thread(target=self._run_training_script, args=(callback,)).start()
        return True

