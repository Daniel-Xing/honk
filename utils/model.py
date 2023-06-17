from enum import Enum
import hashlib
import math
import os
import random
import re

from chainmap import ChainMap
from torch.autograd import Variable
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from .manage_audio import AudioPreprocessor


class SimpleCache(dict):
    def __init__(self, limit):
        super().__init__()
        self.limit = limit
        self.n_keys = 0

    def __setitem__(self, key, value):
        if key in self.keys():
            super().__setitem__(key, value)
        elif self.n_keys < self.limit:
            self.n_keys += 1
            super().__setitem__(key, value)
        return value


class ConfigType(Enum):
    CNN_TRAD_POOL2 = "cnn-trad-pool2"  # default full model (TF variant)
    CNN_ONE_STRIDE1 = "cnn-one-stride1"  # default compact model (TF variant)
    CNN_ONE_FPOOL3 = "cnn-one-fpool3"
    CNN_ONE_FSTRIDE4 = "cnn-one-fstride4"
    CNN_ONE_FSTRIDE8 = "cnn-one-fstride8"
    CNN_TPOOL2 = "cnn-tpool2"
    CNN_TPOOL3 = "cnn-tpool3"
    CNN_TSTRIDE2 = "cnn-tstride2"
    CNN_TSTRIDE4 = "cnn-tstride4"
    CNN_TSTRIDE8 = "cnn-tstride8"
    RES15 = "res15"
    RES26 = "res26"
    RES8 = "res8"
    RES15_NARROW = "res15-narrow"
    RES8_NARROW = "res8-narrow"
    RES26_NARROW = "res26-narrow"


def find_model(conf):
    if isinstance(conf, ConfigType):
        conf = conf.value
    if conf.startswith("res"):
        return SpeechResModel
    else:
        return SpeechModel


def find_config(conf):
    if isinstance(conf, ConfigType):
        conf = conf.value
    return _configs[conf]


def truncated_normal(tensor, std_dev=0.01):
    tensor.zero_()
    tensor.normal_(std=std_dev)
    while torch.sum(torch.abs(tensor) > 2 * std_dev) > 0:
        t = tensor[torch.abs(tensor) > 2 * std_dev]
        t.zero_()
        tensor[torch.abs(tensor) > 2 * std_dev] = torch.normal(t, std=std_dev)


class SerializableModule(nn.Module):
    def __init__(self):
        super().__init__()

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(
            torch.load(filename, map_location=lambda storage, loc: storage)
        )


class SpeechResModel(SerializableModule):
    def __init__(self, config):
        super().__init__()
        n_labels = config["n_labels"]
        n_maps = config["n_feature_maps"]
        self.conv0 = nn.Conv2d(1, n_maps, (3, 3), padding=(1, 1), bias=False)
        if "res_pool" in config:
            self.pool = nn.AvgPool2d(config["res_pool"])

        self.n_layers = n_layers = config["n_layers"]
        dilation = config["use_dilation"]
        if dilation:
            self.convs = [
                nn.Conv2d(
                    n_maps,
                    n_maps,
                    (3, 3),
                    padding=int(2 ** (i // 3)),
                    dilation=int(2 ** (i // 3)),
                    bias=False,
                )
                for i in range(n_layers)
            ]
        else:
            self.convs = [
                nn.Conv2d(n_maps, n_maps, (3, 3), padding=1, dilation=1, bias=False)
                for _ in range(n_layers)
            ]
        for i, conv in enumerate(self.convs):
            self.add_module("bn{}".format(i + 1), nn.BatchNorm2d(n_maps, affine=False))
            self.add_module("conv{}".format(i + 1), conv)
        self.output = nn.Linear(n_maps, n_labels)

    def forward(self, x):
        x = x.unsqueeze(1)
        for i in range(self.n_layers + 1):
            y = F.relu(getattr(self, "conv{}".format(i))(x))
            if i == 0:
                if hasattr(self, "pool"):
                    y = self.pool(y)
                old_x = y
            if i > 0 and i % 2 == 0:
                x = y + old_x
                old_x = x
            else:
                x = y
            if i > 0:
                x = getattr(self, "bn{}".format(i))(x)
        x = x.view(x.size(0), x.size(1), -1)  # shape: (batch, feats, o3)
        x = torch.mean(x, 2)
        return self.output(x)


class SpeechModel(SerializableModule):
    def __init__(self, config):
        """
        初始化语音模型对象。

        参数：
            config: 配置参数

        """
        super().__init__()
        n_labels = config["n_labels"]
        n_featmaps1 = config["n_feature_maps1"]

        conv1_size = config["conv1_size"]  # 第一卷积层的大小 (time, frequency)
        conv1_pool = config["conv1_pool"]  # 第一卷积层的池化尺寸
        conv1_stride = tuple(config["conv1_stride"])  # 第一卷积层的步幅
        dropout_prob = config["dropout_prob"]  # Dropout 层的丢弃概率
        width = config["width"]  # 输入图像的宽度
        height = config["height"]  # 输入图像的高度
        self.conv1 = nn.Conv2d(1, n_featmaps1, conv1_size, stride=conv1_stride)  # 第一卷积层
        tf_variant = config.get("tf_variant")  # 是否使用 TensorFlow 变体
        self.tf_variant = tf_variant
        if tf_variant:
            truncated_normal(self.conv1.weight.data)  # 对卷积层的权重进行截断正态分布初始化
            self.conv1.bias.data.zero_()  # 将卷积层的偏置初始化为零
        self.pool1 = nn.MaxPool2d(conv1_pool)  # 第一池化层

        x = Variable(torch.zeros(1, 1, height, width), volatile=True)  # 创建一个输入示例
        x = self.pool1(self.conv1(x))  # 将输入示例传递给卷积层和池化层，获取特征图

        conv_net_size = x.view(1, -1).size(1)  # 计算卷积后特征图的大小
        last_size = conv_net_size

        if "conv2_size" in config:
            conv2_size = config["conv2_size"]  # 第二卷积层的大小
            conv2_pool = config["conv2_pool"]  # 第二卷积层的池化尺寸
            conv2_stride = tuple(config["conv2_stride"])  # 第二卷积层的步幅
            n_featmaps2 = config["n_feature_maps2"]  # 第二卷积层的特征图数量
            self.conv2 = nn.Conv2d(
                n_featmaps1, n_featmaps2, conv2_size, stride=conv2_stride
            )  # 第二卷积层
            if tf_variant:
                truncated_normal(self.conv2.weight.data)  # 对卷积层的权重进行截断正态分布初始化
                self.conv2.bias.data.zero_()  # 将卷积层的偏置初始化为零
            self.pool2 = nn.MaxPool2d(conv2_pool)  # 第二池化层

            x = self.pool2(self.conv2(x))  # 将特征图传递给第二卷积层和第二池化层
            conv_net_size = x.view(1, -1).size(1)  # 计算卷积后特征图的大小
            last_size = conv_net_size

        if not tf_variant:
            self.lin = nn.Linear(conv_net_size, 32)  # 全连接层

        if "dnn1_size" in config:
            dnn1_size = config["dnn1_size"]  # 第一个全连接层的大小
            last_size = dnn1_size
            if tf_variant:
                self.dnn1 = nn.Linear(conv_net_size, dnn1_size)  # 第一个全连接层
                truncated_normal(self.dnn1.weight.data)  # 对权重进行截断正态分布初始化
                self.dnn1.bias.data.zero_()  # 将偏置初始化为零
            else:
                self.dnn1 = nn.Linear(32, dnn1_size)  # 第一个全连接层
            if "dnn2_size" in config:
                dnn2_size = config["dnn2_size"]  # 第二个全连接层的大小
                last_size = dnn2_size
                self.dnn2 = nn.Linear(dnn1_size, dnn2_size)  # 第二个全连接层
                if tf_variant:
                    truncated_normal(self.dnn2.weight.data)  # 对权重进行截断正态分布初始化
                    self.dnn2.bias.data.zero_()  # 将偏置初始化为零

        self.output = nn.Linear(last_size, n_labels)  # 输出层
        if tf_variant:
            truncated_normal(self.output.weight.data)  # 对权重进行截断正态分布初始化
            self.output.bias.data.zero_()  # 将偏置初始化为零
        self.dropout = nn.Dropout(dropout_prob)  # Dropout 层

    def forward(self, x):
        """
        定义模型的前向传播过程。

        参数：
            x: 输入数据

        返回值：
            模型的输出

        """
        x = F.relu(self.conv1(x.unsqueeze(1)))  # 第一卷积层
        x = self.dropout(x)  # Dropout 层
        x = self.pool1(x)  # 第一池化层
        if hasattr(self, "conv2"):
            x = F.relu(self.conv2(x))  # 第二卷积层
            x = self.dropout(x)  # Dropout 层
            x = self.pool2(x)  # 第二池化层
        x = x.view(x.size(0), -1)  # 展开特征图
        if hasattr(self, "lin"):
            x = self.lin(x)  # 全连接层
        if hasattr(self, "dnn1"):
            x = self.dnn1(x)  # 第一个全连接层
            if not self.tf_variant:
                x = F.relu(x)  # ReLU 激活函数
            x = self.dropout(x)  # Dropout 层
        if hasattr(self, "dnn2"):
            x = self.dnn2(x)  # 第二个全连接层
            x = self.dropout(x)  # Dropout 层
        return self.output(x)  # 输出层


class DatasetType(Enum):
    TRAIN = 0
    DEV = 1
    TEST = 2


class SpeechDataset(data.Dataset):
    LABEL_SILENCE = "__silence__"
    LABEL_UNKNOWN = "__unknown__"

    def __init__(self, data, set_type, config):
        super().__init__()
        self.audio_files = list(data.keys())
        self.set_type = set_type
        self.audio_labels = list(data.values())
        config["bg_noise_files"] = list(
            filter(lambda x: x.endswith("wav"), config.get("bg_noise_files", []))
        )
        self.bg_noise_audio = [
            librosa.core.load(file, sr=16000)[0] for file in config["bg_noise_files"]
        ]
        self.unknown_prob = config["unknown_prob"]
        self.silence_prob = config["silence_prob"]
        self.noise_prob = config["noise_prob"]
        self.input_length = config["input_length"]
        self.timeshift_ms = config["timeshift_ms"]
        self._audio_cache = SimpleCache(config["cache_size"])
        self._file_cache = SimpleCache(config["cache_size"])
        n_unk = len(list(filter(lambda x: x == 1, self.audio_labels)))
        self.n_silence = int(self.silence_prob * (len(self.audio_labels) - n_unk))
        self.audio_processor = AudioPreprocessor(
            n_mels=config["n_mels"], n_dct_filters=config["n_dct_filters"], hop_ms=10
        )
        self.audio_preprocess_type = config["audio_preprocess_type"]

    @staticmethod
    def default_config():
        config = {}
        config["group_speakers_by_id"] = True
        config["silence_prob"] = 0.1
        config["noise_prob"] = 0.8
        config["n_dct_filters"] = 40
        config["input_length"] = 16000
        config["n_mels"] = 40
        config["timeshift_ms"] = 100
        config["unknown_prob"] = 0.1
        config["train_pct"] = 80
        config["dev_pct"] = 10
        config["test_pct"] = 10
        config["wanted_words"] = ["command", "random"]
        config["data_folder"] = "/data/speech_dataset"
        config["audio_preprocess_type"] = "MFCCs"
        return config

    def collate_fn(self, data):
        x = None
        y = []
        for audio_data, label in data:
            if self.audio_preprocess_type == "MFCCs":
                audio_tensor = torch.from_numpy(
                    self.audio_processor.compute_mfccs(audio_data).reshape(1, -1, 40)
                )
                x = audio_tensor if x is None else torch.cat((x, audio_tensor), 0)
            elif self.audio_preprocess_type == "PCEN":
                audio_tensor = torch.from_numpy(np.expand_dims(audio_data, axis=0))
                audio_tensor = self.audio_processor.compute_pcen(audio_tensor)
                x = audio_tensor if x is None else torch.cat((x, audio_tensor), 0)
            y.append(label)
        return x, torch.tensor(y)

    def _timeshift_audio(self, data):
        shift = (16000 * self.timeshift_ms) // 1000
        shift = random.randint(-shift, shift)
        a = -min(0, shift)
        b = max(0, shift)
        data = np.pad(data, (a, b), "constant")
        return data[: len(data) - a] if a else data[b:]

    def load_audio(self, example, silence=False):
        if silence:
            example = "__silence__"
        if random.random() < 0.7 or not self.set_type == DatasetType.TRAIN:
            try:
                return self._audio_cache[example]
            except KeyError:
                pass
        in_len = self.input_length
        if self.bg_noise_audio:
            bg_noise = random.choice(self.bg_noise_audio)
            a = random.randint(0, len(bg_noise) - in_len - 1)
            bg_noise = bg_noise[a : a + in_len]
        else:
            bg_noise = np.zeros(in_len)

        if silence:
            data = np.zeros(in_len, dtype=np.float32)
        else:
            file_data = self._file_cache.get(example)
            data = (
                librosa.core.load(example, sr=16000)[0]
                if file_data is None
                else file_data
            )
            self._file_cache[example] = data
        data = np.pad(data, (0, max(0, in_len - len(data))), "constant")
        if self.set_type == DatasetType.TRAIN:
            data = self._timeshift_audio(data)

        if random.random() < self.noise_prob or silence:
            a = random.random() * 0.1
            data = np.clip(a * bg_noise + data, -1, 1)

        self._audio_cache[example] = data
        return data

    @classmethod
    def splits(cls, config):
        # 从配置中获取必要的参数
        print("开始从配置中获取必要的参数...")
        folder = config["data_folder"]
        wanted_words = config["wanted_words"]
        unknown_prob = config["unknown_prob"]
        train_pct = config["train_pct"]
        dev_pct = config["dev_pct"]
        test_pct = config["test_pct"]

        # 创建一个字典，其中每个需要的单词都映射到一个整数
        print("开始创建字典...")
        words = {word: i + 2 for i, word in enumerate(wanted_words)}
        # 将silence和unknown标签添加到字典中
        words.update({cls.LABEL_SILENCE: 0, cls.LABEL_UNKNOWN: 1})
        print("字典创建完成  ", words)

        # 初始化训练、开发和测试的空集
        sets = [{}, {}, {}]
        # 初始化未知单词的计数器
        unknowns = [0] * 3
        # 初始化背景噪声和未知文件的列表
        bg_noise_files = []
        unknown_files = []

        print("开始遍历数据文件夹...")
        for folder_name in os.listdir(folder):
            # 文件夹的完整路径
            path_name = os.path.join(folder, folder_name)
            is_bg_noise = False
            # 检查路径是否为文件
            if os.path.isfile(path_name):
                continue
            # 在文件夹中标记文件
            if folder_name in words:
                label = words[folder_name]
            elif folder_name == "_background_noise_":
                is_bg_noise = True
            else:
                label = words[cls.LABEL_UNKNOWN]

            print(f"开始遍历文件夹: {folder_name}...")
            for filename in os.listdir(path_name):
                # 文件的完整路径
                wav_name = os.path.join(path_name, filename)
                print("wav_name:", wav_name)  # 打印文件的完整路径
                # 如果文件是背景噪声，将其添加到bg_noise_files中并继续
                if is_bg_noise and os.path.isfile(wav_name):
                    bg_noise_files.append(wav_name)
                    print("Added to bg_noise_files:", wav_name)  # 打印已添加到bg_noise_files中的文件
                    continue
                # 如果文件是未知的，将其添加到unknown_files中并继续
                elif label == words[cls.LABEL_UNKNOWN]:
                    unknown_files.append(wav_name)
                    print("Added to unknown_files:", wav_name)  # 打印已添加到unknown_files中的文件
                    continue
                # 如果配置为按ID分组说话者，则进行此操作
                if config["group_speakers_by_id"]:
                    hashname = re.sub(r"_nohash_.*$", "", filename)
                    print("hashname:", hashname)  # 打印经过处理的哈希名称
                # 对文件名进行哈希以决定其进入哪个集合
                max_no_wavs = 2**27 - 1
                bucket = int(hashlib.sha1(hashname.encode()).hexdigest(), 16)
                bucket = (bucket % (max_no_wavs + 1)) * (100.0 / max_no_wavs)
                print("bucket:", bucket)  # 打印文件的哈希桶值
                # 根据bucket值决定文件进入哪个集合
                if bucket < dev_pct:
                    tag = DatasetType.DEV
                elif bucket < test_pct + dev_pct:
                    tag = DatasetType.TEST
                else:
                    tag = DatasetType.TRAIN
                print("tag:", tag)  # 打印文件的标签类型
                # 将文件和其标签添加到正确的集合中
                sets[tag.value][wav_name] = label
                print("Added to sets[{}]: {}, Label: {}".format(tag.value, wav_name, label))  # 打印已添加到集合中的文件和标签


        # 计算每个集合中未知文件的数量
        for tag in range(len(sets)):
            unknowns[tag] = int(unknown_prob * len(sets[tag]))

        # 随机打乱unknown_files以保证随机性
        random.shuffle(unknown_files)
        a = 0
        # 使用未知文件更新每个集合
        print("开始更新每个集合...")
        for i, dataset in enumerate(sets):
            b = a + unknowns[i]
            unk_dict = {u: words[cls.LABEL_UNKNOWN] for u in unknown_files[a:b]}
            dataset.update(unk_dict)
            a = b

        # 为训练和测试阶段创建配置
        train_cfg = ChainMap(dict(bg_noise_files=bg_noise_files), config)
        test_cfg = ChainMap(dict(bg_noise_files=bg_noise_files, noise_prob=0), config)
        # 创建数据集
        print("开始创建数据集...")
        datasets = (
            cls(sets[0], DatasetType.TRAIN, train_cfg),
            cls(sets[1], DatasetType.DEV, test_cfg),
            cls(sets[2], DatasetType.TEST, test_cfg),
        )
        # 返回数据集
        print("完成创建数据集，开始返回结果...")
        return datasets


    def __getitem__(self, index):
        if index >= len(self.audio_labels):
            return self.load_audio(None, silence=True), 0
        return self.load_audio(self.audio_files[index]), self.audio_labels[index]

    def __len__(self):
        return len(self.audio_labels) + self.n_silence


_configs = {
    ConfigType.CNN_TRAD_POOL2.value: dict(
        dropout_prob=0.5,
        height=101,
        width=40,
        n_labels=4,
        n_feature_maps1=64,
        n_feature_maps2=64,
        conv1_size=(20, 8),
        conv2_size=(10, 4),
        conv1_pool=(2, 2),
        conv1_stride=(1, 1),
        conv2_stride=(1, 1),
        conv2_pool=(1, 1),
        tf_variant=True,
    ),
    ConfigType.CNN_ONE_STRIDE1.value: dict(
        dropout_prob=0.5,
        height=101,
        width=40,
        n_labels=4,
        n_feature_maps1=186,
        conv1_size=(101, 8),
        conv1_pool=(1, 1),
        conv1_stride=(1, 1),
        dnn1_size=128,
        dnn2_size=128,
        tf_variant=True,
    ),
    ConfigType.CNN_TSTRIDE2.value: dict(
        dropout_prob=0.5,
        height=101,
        width=40,
        n_labels=4,
        n_feature_maps1=78,
        n_feature_maps2=78,
        conv1_size=(16, 8),
        conv2_size=(9, 4),
        conv1_pool=(1, 3),
        conv1_stride=(2, 1),
        conv2_stride=(1, 1),
        conv2_pool=(1, 1),
        dnn1_size=128,
        dnn2_size=128,
    ),
    ConfigType.CNN_TSTRIDE4.value: dict(
        dropout_prob=0.5,
        height=101,
        width=40,
        n_labels=4,
        n_feature_maps1=100,
        n_feature_maps2=78,
        conv1_size=(16, 8),
        conv2_size=(5, 4),
        conv1_pool=(1, 3),
        conv1_stride=(4, 1),
        conv2_stride=(1, 1),
        conv2_pool=(1, 1),
        dnn1_size=128,
        dnn2_size=128,
    ),
    ConfigType.CNN_TSTRIDE8.value: dict(
        dropout_prob=0.5,
        height=101,
        width=40,
        n_labels=4,
        n_feature_maps1=126,
        n_feature_maps2=78,
        conv1_size=(16, 8),
        conv2_size=(5, 4),
        conv1_pool=(1, 3),
        conv1_stride=(8, 1),
        conv2_stride=(1, 1),
        conv2_pool=(1, 1),
        dnn1_size=128,
        dnn2_size=128,
    ),
    ConfigType.CNN_TPOOL2.value: dict(
        dropout_prob=0.5,
        height=101,
        width=40,
        n_labels=4,
        n_feature_maps1=94,
        n_feature_maps2=94,
        conv1_size=(21, 8),
        conv2_size=(6, 4),
        conv1_pool=(2, 3),
        conv1_stride=(1, 1),
        conv2_stride=(1, 1),
        conv2_pool=(1, 1),
        dnn1_size=128,
        dnn2_size=128,
    ),
    ConfigType.CNN_TPOOL3.value: dict(
        dropout_prob=0.5,
        height=101,
        width=40,
        n_labels=4,
        n_feature_maps1=94,
        n_feature_maps2=94,
        conv1_size=(15, 8),
        conv2_size=(6, 4),
        conv1_pool=(3, 3),
        conv1_stride=(1, 1),
        conv2_stride=(1, 1),
        conv2_pool=(1, 1),
        dnn1_size=128,
        dnn2_size=128,
    ),
    ConfigType.CNN_ONE_FPOOL3.value: dict(
        dropout_prob=0.5,
        height=101,
        width=40,
        n_labels=4,
        n_feature_maps1=54,
        conv1_size=(101, 8),
        conv1_pool=(1, 3),
        conv1_stride=(1, 1),
        dnn1_size=128,
        dnn2_size=128,
    ),
    ConfigType.CNN_ONE_FSTRIDE4.value: dict(
        dropout_prob=0.5,
        height=101,
        width=40,
        n_labels=4,
        n_feature_maps1=186,
        conv1_size=(101, 8),
        conv1_pool=(1, 1),
        conv1_stride=(1, 4),
        dnn1_size=128,
        dnn2_size=128,
    ),
    ConfigType.CNN_ONE_FSTRIDE8.value: dict(
        dropout_prob=0.5,
        height=101,
        width=40,
        n_labels=4,
        n_feature_maps1=336,
        conv1_size=(101, 8),
        conv1_pool=(1, 1),
        conv1_stride=(1, 8),
        dnn1_size=128,
        dnn2_size=128,
    ),
    ConfigType.RES15.value: dict(
        n_labels=12, use_dilation=True, n_layers=13, n_feature_maps=45
    ),
    ConfigType.RES8.value: dict(
        n_labels=12, n_layers=6, n_feature_maps=45, res_pool=(4, 3), use_dilation=False
    ),
    ConfigType.RES26.value: dict(
        n_labels=12, n_layers=24, n_feature_maps=45, res_pool=(2, 2), use_dilation=False
    ),
    ConfigType.RES15_NARROW.value: dict(
        n_labels=12, use_dilation=True, n_layers=13, n_feature_maps=19
    ),
    ConfigType.RES8_NARROW.value: dict(
        n_labels=12, n_layers=6, n_feature_maps=19, res_pool=(4, 3), use_dilation=False
    ),
    ConfigType.RES26_NARROW.value: dict(
        n_labels=12, n_layers=24, n_feature_maps=19, res_pool=(2, 2), use_dilation=False
    ),
}
