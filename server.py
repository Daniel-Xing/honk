import base64
import json
import io
import os
import re
import threading
import wave
import zlib

import cherrypy
import numpy as np

from service import Caffe2LabelService, TorchLabelService, TrainingService
from service import stride

def json_in(f):
    def merge_dicts(x, y):
        z = x.copy()
        z.update(y)
        return z
    def wrapper(*args, **kwargs):
        cl = cherrypy.request.headers["Content-Length"]
        data = json.loads(cherrypy.request.body.read(int(cl)).decode("utf-8"))
        kwargs = merge_dicts(kwargs, data)
        return f(*args, **kwargs)
    return wrapper

class TrainEndpoint(object):
    exposed = True
    def __init__(self, train_service, label_service):
        self.train_service = train_service
        self.label_service = label_service

    @cherrypy.tools.json_out()
    def POST(self):
        return dict(success=self.train_service.run_train_script(callback=self.label_service.reload))

    @cherrypy.tools.json_out()
    def GET(self):
        return dict(in_progress=self.train_service.script_running)

class DataEndpoint(object):
    exposed = True
    def __init__(self, train_service):
        self.train_service = train_service

    @cherrypy.tools.json_out()
    @json_in
    def POST(self, **kwargs):
        wav_data = zlib.decompress(base64.b64decode(kwargs["wav_data"]))
        positive = kwargs["positive"]
        for _ in range(3):
            self.train_service.write_example(wav_data, positive=positive)
        success = dict(success=True)
        if not positive:
            return success
        neg_examples = self.train_service.generate_contrastive(wav_data)
        if not neg_examples:
            return success
        for example in neg_examples:
            self.train_service.write_example(example.byte_data, positive=False, tag="gen")
        return success

    @cherrypy.tools.json_out()
    def DELETE(self):
        self.train_service.clear_examples(positive=True)
        self.train_service.clear_examples(positive=False, tag="gen")
        return dict(success=True)

class EvaluateEndpoint(object):
    exposed = True
    def __init__(self, label_service):
        self.label_service = label_service

    @cherrypy.tools.json_out()
    @json_in
    def POST(self, **kwargs):
        folders = kwargs["folders"]
        indices = kwargs.get("indices", [])
        accuracy = self.label_service.evaluate(folders, indices)
        return dict(accuracy=accuracy)
        
class ListenEndpoint(object):
    exposed = True
    def __init__(self, label_service, stride_size=500, min_keyword_prob=0.85, keyword="command"):
        """The REST API endpoint that determines if audio contains the keyword.

        Args:
            label_service: The labelling service to use
            stride_size: The stride in milliseconds of the 1-second window to use. It should divide 1000 ms.
            min_keyword_prob: The minimum probability the keyword must take in order to be classified as such
            keyword: The keyword
        """
        self.label_service = label_service
        self.stride_size = stride_size
        self.min_keyword_prob = min_keyword_prob
        self.keyword = keyword

    @cherrypy.tools.json_out()
    @json_in
    def POST(self, **kwargs):
        wav_data = zlib.decompress(base64.b64decode(kwargs["wav_data"]))
        labels = {}
        for data in stride(wav_data, int(2 * 16000 * self.stride_size / 1000), 2 * 16000):
            label, prob = self.label_service.label(data)
            try:
                labels[label] += float(prob)
            except KeyError:
                labels[label] = float(prob)
            if label == "command" and prob >= self.min_keyword_prob and kwargs["method"] == "command_tagging":
                return dict(contains_command=True)
        return dict(contains_command=False) if kwargs["method"] == "command_tagging" else labels

def make_abspath(rel_path):
    if not os.path.isabs(rel_path):  # 检查给定的路径是否为绝对路径
        rel_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), rel_path)
        # 如果是相对路径，则使用当前脚本文件的路径和相对路径拼接，生成绝对路径
    return rel_path  # 返回生成的绝对路径

def load_service(config):
    """
    加载标签服务

    参数：
        config: 配置对象

    返回值：
        lbl_service: 标签服务对象
    """
    # 获取模型路径并转换为绝对路径
    model_path = make_abspath(config["model_path"])

    # 创建命令列表，初始包含 "__silence__" 和 "__unknown__"
    # 并将配置中的命令以逗号分隔加入列表中
    commands = ["__silence__", "__unknown__"]
    commands.extend(config["commands"].split(","))

    # 获取后端类型
    backend = config["backend"]

    if backend.lower() == "caffe2":  # 如果后端为 Caffe2
        lbl_service = Caffe2LabelService(model_path, commands)  # 创建 Caffe2LabelService 对象
    elif backend.lower() == "pytorch":  # 如果后端为 PyTorch
        lbl_service = TorchLabelService(model_path, labels=commands, no_cuda=config["model_options"]["no_cuda"])
        # 创建 TorchLabelService 对象，指定标签、是否使用 GPU 等参数
    else:
        raise ValueError("Backend {} not supported!".format(backend))  # 抛出错误，不支持的后端类型

    return lbl_service  # 返回标签服务对象


def start(config):
    cherrypy.config.update({
        "environment": "production",  # 设置 CherryPy 的运行环境为 "production"
        "log.screen": True  # 在屏幕上显示日志信息
    })
    cherrypy.config.update(config["server"])  # 使用配置文件中的服务器配置更新 CherryPy 配置
    rest_config = {"/": {
        "request.dispatch": cherrypy.dispatch.MethodDispatcher()  # 将请求调度方式设置为方法调度器
    }}
    train_script = make_abspath(config["train_script"])  # 使用配置文件中的训练脚本路径生成绝对路径
    speech_dataset_path = make_abspath(config["speech_dataset_path"])  # 使用配置文件中的语音数据集路径生成绝对路径
    
    lbl_service = load_service(config)  # 加载配置文件中的标签服务
    
    train_service = TrainingService(train_script, speech_dataset_path, config["model_options"])
    
    # 创建训练服务对象，使用训练脚本路径、语音数据集路径和模型选项作为参数
    cherrypy.tree.mount(ListenEndpoint(lbl_service), "/listen", rest_config)
    
    # 将 ListenEndpoint 对象挂载到 "/listen" 路径上
    cherrypy.tree.mount(DataEndpoint(train_service), "/data", rest_config)
    
    # 将 DataEndpoint 对象挂载到 "/data" 路径上
    cherrypy.tree.mount(EvaluateEndpoint(lbl_service), "/evaluate", rest_config)
    
    # 将 EvaluateEndpoint 对象挂载到 "/evaluate" 路径上
    cherrypy.tree.mount(TrainEndpoint(train_service, lbl_service), "/train", rest_config)
    
    # 将 TrainEndpoint 对象挂载到 "/train" 路径上
    cherrypy.engine.start()  # 启动 CherryPy 引擎
    cherrypy.engine.block()  # 阻塞主线程，使 CherryPy 运行

# CherryPy是一个Python的Web应用程序开发框架，它旨在简化Web应用程序的开发过程。
# 它提供了一组工具和抽象，使开发者能够轻松地构建、部署和管理Web应用程序。

# CherryPy具有以下特点和功能：
# - 轻量级：CherryPy是一个轻量级框架，它的核心库非常小巧，但功能丰富。这使得它易于学习和使用，并且对于小型应用程序或API开发非常适用。
# - 易于使用：CherryPy使用Pythonic的API设计，提供了简洁、直观的语法，使开发人员能够快速构建Web应用程序。
# - 路由和URL映射：CherryPy提供了路由功能，可以将URL映射到相应的处理器函数或类。这样，开发人员可以定义不同URL路径的行为。
# - 内置的Web服务器：CherryPy内置了一个简单而高效的Web服务器，因此您无需额外安装和配置Web服务器即可运行CherryPy应用程序。
# - 插件支持：CherryPy支持插件机制，允许开发人员扩展框架的功能。这使得您可以根据需要添加各种功能，如数据库访问、会话管理等。
# - 可扩展性：CherryPy具有良好的可扩展性，允许您以模块化的方式构建和组织应用程序。您可以将应用程序拆分为多个组件，并在需要时添加新功能。

# start() 函数使用 CherryPy 框架来启动和管理 Web 应用程序的运行。
# 具体步骤如下：
# 1. 设置 CherryPy 的运行环境为 "production"，这表示在生产环境下运行应用程序。
# 2. 在屏幕上显示日志信息。
# 3. 使用配置文件中的服务器配置更新 CherryPy 的配置。
# 4. 创建一个请求调度方式为方法调度器的配置字典。
# 5. 生成训练脚本和语音数据集的绝对路径。
# 6. 加载配置文件中的标签服务。
# 7. 创建训练服务对象，使用训练脚本路径、语音数据集路径和模型选项作为参数。
# 8. 将不同的端点（Endpoint）对象挂载到不同的路径上，用于处理对应路径的请求。
# 9. 启动 CherryPy 引擎，开始监听和处理请求。
# 10. 阻塞主线程，使 CherryPy 运行，直到应用程序停止或退出。



