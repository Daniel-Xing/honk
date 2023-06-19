from collections import ChainMap  # 导入ChainMap模块，用于创建联合视图
import argparse  # 导入argparse模块，用于解析命令行参数
import os  # 导入os模块，提供与操作系统交互的功能
import random  # 导入random模块，用于生成伪随机数
import sys  # 导入sys模块，提供与Python解释器交互的功能

from torch.autograd import Variable  # 从torch.autograd模块导入Variable类，用于包装张量并自动计算梯度
import numpy as np  # 导入numpy模块并将其重命名为np，提供多维数组和矩阵运算的功能
import torch  # 导入torch模块，PyTorch深度学习框架的主要包
import torch.nn as nn  # 导入torch.nn模块，PyTorch中的神经网络模块
import torch.utils.data as data  # 导入torch.utils.data模块，提供用于处理数据集的工具
import copy  # 导入copy模块，提供对象的复制操作

from . import model as mod  # 导入自定义模块.model并将其重命名为mod
from .manage_audio import AudioPreprocessor  # 导入自定义模块.manage_audio中的AudioPreprocessor类
from torch.utils.tensorboard import SummaryWriter
# default `log_dir` is "logs" - we'll be more specific here
writer = SummaryWriter('logs/speech')


class ConfigBuilder(object):
    """
    配置构建器类，用于构建和解析配置。

    """

    def __init__(self, *default_configs):
        """
        初始化配置构建器。

        参数：
            *default_configs: 一个或多个默认配置的字典

        """
        self.default_config = ChainMap(*default_configs)  # 使用ChainMap创建联合视图，合并多个默认配置字典

    def build_argparse(self):
        """
        构建argparse.ArgumentParser对象，用于解析命令行参数。

        返回值：
            argparse.ArgumentParser对象

        """
        parser = argparse.ArgumentParser()  # 创建argparse.ArgumentParser对象
        for key, value in self.default_config.items():
            key = "--{}".format(key)  # 构建参数的名称
            if isinstance(value, tuple):
                parser.add_argument(
                    key, default=list(value), nargs=len(value), type=type(value[0])
                )
            elif isinstance(value, list):
                parser.add_argument(key, default=value, nargs="+", type=type(value[0]))
            elif isinstance(value, bool) and not value:
                parser.add_argument(key, action="store_true")
            else:
                parser.add_argument(key, default=value, type=type(value))
        return parser

    def config_from_argparse(self, parser=None):
        """
        根据解析命令行参数后的结果和默认配置，返回合并后的配置字典。

        参数：
            parser: argparse.ArgumentParser对象，用于解析命令行参数（可选）

        返回值：
            合并后的配置字典

        """
        if not parser:
            parser = self.build_argparse()  # 如果未提供parser对象，则调用build_argparse()方法创建一个
        args = vars(parser.parse_known_args()[0])  # 解析命令行参数并获取结果
        return ChainMap(args, self.default_config)  # 将解析后的结果与默认配置进行合并创建ChainMap对象


def print_eval(name, scores, labels, loss, end="\n", global_step = 0):
    """
    打印评估结果。

    参数：
        name: 评估名称
        scores: 预测分数
        labels: 真实标签
        loss: 损失值
        end: 结束符（可选，默认为换行符）

    返回值：
        accuracy的值

    """
    batch_size = labels.size(0)  # 批次大小
    accuracy = (
        torch.max(scores, 1)[1].view(batch_size).data == labels.data
    ).float().sum() / batch_size  # 计算准确率
    loss = loss.item()  # 获取损失值
    print(
        "{} accuracy: {:>5}, loss: {:<25}".format(name, accuracy, loss), end=end
    )  # 打印评估结果
    
    if global_step != 0:
        print("写入tensorBoard   ", "global_step:{:>5} accuracy: {:>5}, loss: {:<25}".format(global_step, accuracy, loss), end=end)
        writer.add_scalar('Loss/train', loss, global_step)
        writer.add_scalar('Accuracy/train', accuracy, global_step)
        
    return accuracy.item()  # 返回准确率的值


def set_seed(config):
    """
    设置随机种子。

    参数：
        config: 配置字典

    """
    seed = config["seed"]  # 获取种子值
    torch.manual_seed(seed)  # 设置PyTorch的随机种子
    np.random.seed(seed)  # 设置NumPy的随机种子
    if not config["no_cuda"]:
        torch.cuda.manual_seed(seed)  # 如果不禁用CUDA，则设置CUDA的随机种子
    random.seed(seed)  # 设置Python内置的随机种子


def evaluate(config, model=None, test_loader=None):
    """
    执行模型的评估过程。

    参数：
        config: 包含评估配置的字典
        model: 要评估的模型对象（可选）
        test_loader: 测试数据加载器（可选）

    """
    if not test_loader:
        # 如果没有提供测试数据加载器
        _, _, test_set = mod.SpeechDataset.splits(config)  # 将配置中的数据集分割为训练集、开发集和测试集
        test_loader = data.DataLoader(
            test_set,
            batch_size=len(test_set),  # 将整个测试集作为一个批次
            collate_fn=test_set.collate_fn,
        )  # 使用数据集的collate_fn函数对数据进行处理

    if not config["no_cuda"]:
        # 如果未禁用CUDA
        torch.cuda.set_device(config["gpu_no"])  # 设置CUDA设备为指定的GPU编号

    if not model:
        # 如果未提供模型对象
        model = config["model_class"](config)  # 根据配置中指定的模型类创建新的模型对象
        model.load(config["input_file"])  # 从指定的输入文件中加载模型参数

    if not config["no_cuda"]:
        # 如果未禁用CUDA
        torch.cuda.set_device(config["gpu_no"])  # 设置CUDA设备为指定的GPU编号
        model.cuda()  # 将模型移动到CUDA设备上

    model.eval()  # 将模型设置为评估模式
    criterion = nn.CrossEntropyLoss()  # 定义交叉熵损失函数
    results = []  # 存储评估结果
    total = 0  # 总样本数

    for model_in, labels in test_loader:
        # 遍历测试数据加载器
        model_in = Variable(model_in, requires_grad=False)
        if not config["no_cuda"]:
            model_in = model_in.cuda()
            labels = labels.cuda()

        scores = model(model_in)  # 通过模型进行前向传播得到预测结果
        labels = Variable(labels, requires_grad=False)
        loss = criterion(scores, labels)  # 计算损失值
        results.append(
            print_eval("test", scores, labels, loss) * model_in.size(0)
        )  # 计算并存储准确率
        total += model_in.size(0)  # 更新总样本数

    print("final test accuracy: {}".format(sum(results) / total))  # 打印最终的测试准确率


def train(config):
    """
    执行模型的训练过程。

    参数：
        config: 包含训练配置的字典

    """
    output_dir = os.path.dirname(os.path.abspath(config["output_file"]))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_set, dev_set, test_set = mod.SpeechDataset.splits(
        config
    )  # 将配置中的数据集分割为训练集、开发集和测试集
    print(train_set.audio_files)
    
    model = config["model_class"](config)  # 根据配置中指定的模型类创建新的模型对象
    if config["input_file"]:
        model.load(config["input_file"])  # 如果指定了输入文件，则加载模型参数
    if not config["no_cuda"]:
        torch.cuda.set_device(config["gpu_no"])  # 设置CUDA设备为指定的GPU编号
        model.cuda()  # 将模型移动到CUDA设备上

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["lr"][0],
        nesterov=config["use_nesterov"],
        weight_decay=config["weight_decay"],
        momentum=config["momentum"],
    )  # 定义优化器
    schedule_steps = config["schedule"]  # 定义学习率调整步骤
    schedule_steps.append(np.inf)  # 在调整步骤末尾添加无穷大的步骤
    sched_idx = 0  # 当前调整步骤索引
    criterion = nn.CrossEntropyLoss()  # 定义交叉熵损失函数
    max_acc = 0  # 最大准确率

    train_loader = data.DataLoader(
        train_set,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
        collate_fn=train_set.collate_fn,
    )  # 创建训练数据加载器
    dev_loader = data.DataLoader(
        dev_set,
        batch_size=min(len(dev_set), 16),
        shuffle=False,
        collate_fn=dev_set.collate_fn,
    )  # 创建开发数据加载器
    test_loader = data.DataLoader(
        test_set,
        batch_size=len(test_set),
        shuffle=False,
        collate_fn=test_set.collate_fn,
    )  # 创建测试数据加载器
    step_no = 0  # 当前训练步骤编号

    for epoch_idx in range(config["n_epochs"]):
        # 遍历每个训练周期
        # initialize the running loss for visualization
        for batch_idx, (model_in, labels) in enumerate(train_loader):
            # 遍历训练数据加载器中的批次数据
            model.train()  # 设置模型为训练模式
            optimizer.zero_grad()  # 清除梯度
            if not config["no_cuda"]:
                model_in = model_in.cuda()
                labels = labels.cuda()
            model_in = Variable(model_in, requires_grad=False)
            scores = model(model_in)  # 通过模型进行前向传播得到预测结果
            labels = Variable(labels, requires_grad=False)
            loss = criterion(scores, labels)  # 计算损失值
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新模型参数
            step_no += 1  # 更新训练步骤编号

            if step_no > schedule_steps[sched_idx]:
                # 如果达到了调整学习率的步骤
                sched_idx += 1  # 更新调整步骤索引
                print("changing learning rate to {}".format(config["lr"][sched_idx]))
                optimizer = torch.optim.SGD(
                    model.parameters(),
                    lr=config["lr"][sched_idx],
                    nesterov=config["use_nesterov"],
                    momentum=config["momentum"],
                    weight_decay=config["weight_decay"],
                )  # 更新优化器的学习率

            print_eval(
                "train step #{}".format(step_no), scores, labels, loss, global_step=step_no
            )  # 打印训练过程的评估结果
            

        if epoch_idx % config["dev_every"] == config["dev_every"] - 1:
            # 如果达到了进行开发集评估的周期
            model.eval()  # 设置模型为评估模式
            accs = []
            for model_in, labels in dev_loader:
                model_in = Variable(model_in, requires_grad=False)
                if not config["no_cuda"]:
                    model_in = model_in.cuda()
                    labels = labels.cuda()
                scores = model(model_in)  # 通过模型进行前向传播得到预测结果
                labels = Variable(labels, requires_grad=False)
                loss = criterion(scores, labels)  # 计算损失值
                accs.append(print_eval("dev", scores, labels, loss))  # 计算并存储准确率
                
            avg_acc = np.mean(accs)  # 计算平均准确率
            print("final dev accuracy: {}".format(avg_acc))  # 打印最终的开发集准确率

            if avg_acc > max_acc:
                # 如果当前准确率超过最大准确率
                print("saving best model...")
                max_acc = avg_acc  # 更新最大准确率
                model.save(config["output_file"])  # 保存模型参数
                best_model = copy.deepcopy(model)  # 复制最佳模型

    evaluate(config, best_model, test_loader)  # 在最佳模型上进行测试集评估


def main():
    output_file = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "model", "model.pt"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=[x.value for x in list(mod.ConfigType)],
        default="cnn-trad-pool2",
        type=str,
    )
    config, _ = parser.parse_known_args()

    global_config = dict(
        no_cuda=False,
        n_epochs=500,
        lr=[0.001],
        schedule=[np.inf],
        batch_size=64,
        dev_every=10,
        seed=0,
        use_nesterov=False,
        input_file="",
        output_file=output_file,
        gpu_no=1,
        cache_size=32768,
        momentum=0.9,
        weight_decay=0.00001,
    )
    mod_cls = mod.find_model(config.model)
    builder = ConfigBuilder(
        mod.find_config(config.model), mod.SpeechDataset.default_config(), global_config
    )
    parser = builder.build_argparse()
    parser.add_argument("--type", choices=["train", "eval"], default="train", type=str)
    config = builder.config_from_argparse(parser)
    config["model_class"] = mod_cls
    set_seed(config)
    if config["type"] == "train":
        train(config)
    elif config["type"] == "eval":
        evaluate(config)
        
    # close writer
    writer.close()


if __name__ == "__main__":
    main()
