import argparse  # 用于解析命令行参数
import base64  # 用于进行Base64编解码
import json  # 用于处理JSON数据
import math  # 提供数学函数和常量
import threading  # 用于多线程编程
import time  # 提供时间相关函数
import zlib  # 用于数据的压缩和解压缩

from OpenGL.GL import *  # 提供OpenGL的函数和常量
from OpenGL.GLUT import *  # 提供OpenGL Utility Toolkit的函数和常量
from OpenGL.GLU import *  # 提供OpenGL Utility Library的函数和常量
from PIL import Image  # 用于图像处理
import librosa  # 用于音频处理
import pyaudio  # 用于音频输入和输出
import numpy as np  # 提供数组和矩阵运算的功能
import requests  # 用于发送HTTP请求和处理响应

textures = {}  # 存储纹理对象的字典
labels = [
    "unknown",
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
]  # 标签列表


def load_texture(filename, width, height):
    """
    加载纹理图片并生成纹理对象。

    参数：
        filename: 图片文件路径
        width: 纹理宽度
        height: 纹理高度

    返回值：
        纹理对象的 ID

    """
    im = Image.open(filename)  # 打开图片文件
    im.convert("RGBA")  # 转换图片格式为RGBA
    data = im.getdata()  # 获取图片数据
    pixels = []  # 存储像素值的列表
    for pixel in reversed(data):  # 遍历像素值（以反向顺序）
        pixels.extend(pixel)  # 将像素值添加到列表中
    pixels = np.array(pixels, np.uint8)  # 将像素值列表转换为NumPy数组，并指定数据类型为无符号8位整数
    tex_id = glGenTextures(1)  # 生成纹理对象的ID

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)  # 设置像素存储模式
    glBindTexture(GL_TEXTURE_2D, tex_id)  # 绑定纹理对象
    glTexImage2D(
        GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels
    )  # 加载纹理数据
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)  # 设置纹理放大过滤方式
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)  # 设置纹理缩小过滤方式
    return tex_id


def draw_spectrogram(audio_data):
    """
    绘制音频的频谱图。

    参数：
        audio_data: 音频数据

    """
    audio_data = (
        np.frombuffer(audio_data, np.int16).astype(np.float32) / 32768
    )  # 将音频数据转换为浮点数表示
    spectrogram = np.absolute(np.fft.fft(audio_data)[: len(audio_data) // 2])  # 计算音频的频谱
    spectrogram = np.power(np.clip(spectrogram, 0, 200) / 200.0, 0.7)  # 对频谱进行幅度归一化和指数变换
    glColor3f(0.3, 0.3, 0.3)  # 设置绘制颜色为灰色
    h = 7  # 每个频谱块的高度
    s = 4  # 频谱块之间的间隔
    for i, energy in enumerate(spectrogram.tolist()):  # 遍历频谱能量值
        glBegin(GL_QUADS)  # 开始绘制四边形
        glVertex2f(0, i * (h + s))  # 左上角顶点
        glVertex2f(int(energy * 150), i * (h + s))  # 右上角顶点
        glVertex2f(int(energy * 150), i * (h + s) + h)  # 右下角顶点
        glVertex2f(0, i * (h + s) + h)  # 左下角顶点
        glEnd()  # 结束绘制四边形
        glBegin(GL_QUADS)  # 开始绘制四边形
        glVertex2f(800, i * (h + s))  # 右上角顶点
        glVertex2f(800 - int(energy * 150), i * (h + s))  # 左上角顶点
        glVertex2f(800 - int(energy * 150), i * (h + s) + h)  # 左下角顶点
        glVertex2f(800, i * (h + s) + h)  # 右下角顶点
        glEnd()  # 结束绘制四边形


def draw_text(text, x, y):
    """
    在指定位置绘制文本。

    参数：
        text: 要绘制的文本
        x: 文本的横坐标
        y: 文本的纵坐标

    """
    m = 0.0385  # 字母之间的间距
    glColor3f(0.9, 0.9, 0.9)  # 设置绘制颜色为浅灰色
    for i, c in enumerate(text):  # 遍历文本中的每个字符
        idx = ord(c) - ord("a")  # 计算字符在字母表中的索引
        glEnable(GL_TEXTURE_2D)  # 启用纹理映射
        glEnable(GL_BLEND)  # 启用混合
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)  # 设置混合函数
        glBindTexture(GL_TEXTURE_2D, textures["font"])  # 绑定字体纹理
        glBegin(GL_QUADS)  # 开始绘制四边形
        glTexCoord2f(m * idx, 0)  # 设置纹理坐标（左下角）
        glVertex2f(x, y)  # 设置顶点坐标（左下角）
        glTexCoord2f(m * idx, 1)  # 设置纹理坐标（左上角）
        glVertex2f(x, y + 15)  # 设置顶点坐标（左上角）
        glTexCoord2f(m * (idx + 1), 1)  # 设置纹理坐标（右上角）
        glVertex2f(x + 10, y + 15)  # 设置顶点坐标（右上角）
        glTexCoord2f(m * (idx + 1), 0)  # 设置纹理坐标（右下角）
        glVertex2f(x + 10, y)  # 设置顶点坐标（右下角）
        x += 10  # 更新下一个字符的横坐标位置
        glEnd()  # 结束绘制四边形
        glDisable(GL_TEXTURE_2D)  # 禁用纹理映射


def create_rot_matrix(rads):
    """
    创建旋转矩阵。

    参数：
        rads: 旋转角度（弧度）

    返回值：
        旋转矩阵的NumPy数组表示

    """
    return np.array(
        [[math.cos(rads), -math.sin(rads)], [math.sin(rads), math.cos(rads)]]
    )


def draw_vertices(vertices):
    """
    绘制顶点集合。

    参数：
        vertices: 顶点列表

    """
    for vertex in vertices:  # 遍历顶点集合中的每个顶点
        glVertex2f(*vertex)  # 设置顶点坐标


class LerpStepper(object):
    def __init__(self, a, b, speed):
        """
        初始化线性插值器。

        参数：
            a: 起始值
            b: 结束值
            speed: 插值速度

        """
        self.last_time = time.time()  # 上次步进时间
        self.val = a  # 当前值
        self.b = b  # 结束值
        self.speed = speed  # 插值速度
        self.running = False  # 是否正在运行

    def reset(self, val, b=None):
        """
        重置插值器的状态。

        参数：
            val: 新的起始值
            b: 新的结束值（可选）

        """
        self.val = val  # 更新当前值
        if b:
            self.b = b  # 更新结束值

    def step(self):
        """
        执行一步线性插值。

        """
        if self.val < self.b:  # 如果当前值小于结束值
            self.val += self.speed  # 根据插值速度增加当前值


class Indicator(object):
    indicators = []  # 存储所有指示器的列表

    def __init__(self, text, midpoint, index, radius=250, n_slices=len(labels)):
        """
        初始化指示器。

        参数：
            text: 指示器文本
            midpoint: 指示器的中点坐标
            index: 指示器的索引
            radius: 指示器的半径（默认为250）
            n_slices: 划分圆周的切片数（默认为标签的数量）

        """
        self.text = text  # 指示器文本
        self._state = 0.0  # 内部状态变量
        self.midpoint = midpoint  # 中点坐标
        self.radius = radius  # 半径
        self.index = index  # 索引
        self.color = (0.4, 0.4, 0.4)  # 指示器的颜色
        self._color_lerp = LerpStepper(1.0, 1.0, 0.01)  # 颜色渐变器
        Indicator.indicators.append(self)  # 将指示器添加到指示器列表中

        slice_rads = (2 * math.pi) / n_slices  # 每个切片的弧度
        rads = index * slice_rads  # 当前切片的起始弧度
        self._rotmat1 = create_rot_matrix(rads)  # 旋转矩阵1
        rads = (index + 0.5) * slice_rads  # 当前切片的中间弧度
        self._rotmat15 = create_rot_matrix(rads)  # 旋转矩阵1.5
        rads = (index + 1) * slice_rads  # 当前切片的结束弧度
        self._rotmat2 = create_rot_matrix(rads)  # 旋转矩阵2
        self._init_shape()  # 初始化指示器的形状

    def highlight(self, intensity):
        """
        设置指示器的高亮度。

        参数：
            intensity: 高亮度强度

        """
        self._color_lerp.reset(1 - intensity)  # 重置颜色渐变器的状态

    def _init_shape(self):
        """
        初始化指示器的形状。

        """
        bp = np.array([0, self.radius])  # 基础点
        bp2 = np.array([0, self.radius * 0.8])  # 基础点2
        m = np.array(self.midpoint)  # 中点坐标
        p1 = np.matmul(self._rotmat1, bp) + m  # 顶点1
        p2 = np.matmul(self._rotmat2, bp) + m  # 顶点2
        self.text_pos = np.matmul(self._rotmat15, bp2) + m  # 文本位置
        self.text_pos[0] -= len(self.text) * 5  # 调整文本位置
        p1 = p1.tolist()
        p2 = p2.tolist()
        self.vertices = [self.midpoint, p1, p2, self.midpoint]  # 顶点列表

    def tick(self):
        """
        更新指示器的状态。

        """
        self._color_lerp.step()  # 颜色渐变器进行步进

    def draw(self):
        """
        绘制指示器。

        """
        draw_text(self.text, self.text_pos[0], self.text_pos[1])  # 绘制指示器文本
        glEnable(GL_LINE_SMOOTH)
        glLineWidth(2)
        glBegin(GL_LINE_STRIP)
        color = [0.4 + min(1 - self._color_lerp.val, 0.6)] * 3  # 计算线条的颜色
        glColor3f(*color)
        draw_vertices(self.vertices)  # 绘制顶点
        glEnd()
        glDisable(GL_LINE_SMOOTH)

    def highlight(self, intensity):
        """
        设置指示器的高亮度。

        参数：
            intensity: 高亮度强度

        """
        self._color_lerp.reset(1 - intensity)  # 重置颜色渐变器的状态


class LabelClient(object):
    def __init__(self, server_endpoint):
        """
        初始化标签客户端。

        参数：
            server_endpoint: 服务器端点地址

        """
        self.endpoint = server_endpoint  # 服务器端点地址
        self.chunk_size = 1000  # 音频数据块大小
        self._audio = pyaudio.PyAudio()  # 创建PyAudio对象
        self._audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._on_audio,
        )  # 打开音频输入流，并设置参数和回调函数
        self.last_data = np.zeros(1000)  # 最近的音频数据
        self._audio_buf = []  # 音频缓冲区

    def _on_audio(self, in_data, frame_count, time_info, status):
        """
        音频回调函数，处理音频数据。

        参数：
            in_data: 输入音频数据
            frame_count: 音频帧数量
            time_info: 时间信息
            status: 状态信息

        返回值：
            音频数据和状态信息

        """
        data_ok = (in_data, pyaudio.paContinue)
        self.last_data = in_data  # 保存最近的音频数据
        self._audio_buf.append(in_data)  # 将音频数据添加到缓冲区
        if len(self._audio_buf) != 16:
            return data_ok  # 如果缓冲区中的音频数据不足16个块，则继续录制
        audio_data = base64.b64encode(
            zlib.compress(b"".join(self._audio_buf))
        )  # 压缩并编码音频数据
        self._audio_buf = []  # 清空音频缓冲区
        response = requests.post(
            "{}/listen".format(self.endpoint),
            json=dict(wav_data=audio_data.decode(), method="all_label"),
        )  # 发送POST请求给服务器，传递压缩的音频数据和方法参数
        response = json.loads(response.content.decode())  # 解析服务器响应
        if not response:
            return data_ok  # 如果服务器响应为空，则继续录制

        max_key = max(response.items(), key=lambda x: x[1])[0]  # 获取概率最大的标签
        for key in response:
            p = response[key]  # 获取标签对应的概率
            if p < 0.5 and key != "__unknown__":
                continue  # 如果概率小于0.5且不是"__unknown__"标签，则忽略该标签
            key = key.replace("_", "")  # 去除标签中的下划线
            try:
                Indicator.indicators[labels.index(key)].highlight(1.0)
            except ValueError:
                continue  # 如果指示器列表中不存在对应标签的指示器，则继续处理下一个标签
        return data_ok  # 返回音频数据和状态信息


class DemoApplication(object):
    def __init__(self, label_client):
        """
        初始化演示应用程序。

        参数：
            label_client: 标签客户端对象

        """
        glutInit()  # 初始化OpenGL环境
        self.width, self.height = 800, 600  # 窗口宽度和高度
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH)  # 设置显示模式
        glutInitWindowSize(self.width, self.height)  # 设置窗口大小
        glutInitWindowPosition(100, 100)  # 设置窗口位置
        self.window = glutCreateWindow(b"Google Speech Dataset Demo")  # 创建窗口并设置窗口标题
        self.label_client = label_client  # 标签客户端对象

        glutDisplayFunc(self.draw)  # 设置绘制函数
        glutIdleFunc(self.draw)  # 设置空闲时的绘制函数
        glutReshapeFunc(self._on_resize)  # 设置窗口调整大小时的回调函数
        glClearColor(0.12, 0.12, 0.15, 1.0)  # 设置清除颜色
        font_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "fonts.png"
        )  # 字体文件路径
        textures["font"] = load_texture(font_path, 208, 15)  # 加载字体纹理
        self.children = [
            Indicator(labels[i], [400, 300], i) for i in range(len(labels))
        ]  # 创建指示器对象列表

    def _refresh(self):
        """
        刷新OpenGL视口。

        """
        glViewport(0, 0, self.width, self.height)  # 设置视口大小
        glMatrixMode(GL_PROJECTION)  # 设置投影矩阵模式
        glLoadIdentity()  # 重置投影矩阵
        glOrtho(0.0, self.width, 0.0, self.height, 0.0, 1.0)  # 设置正交投影矩阵
        glMatrixMode(GL_MODELVIEW)  # 设置模型视图矩阵模式
        glLoadIdentity()  # 重置模型视图矩阵

    def _on_resize(self, width, height):
        """
        窗口调整大小时的回调函数。

        参数：
            width: 新的窗口宽度
            height: 新的窗口高度

        """
        glutReshapeWindow(self.width, self.height)  # 设置窗口大小

    def draw(self):
        """
        绘制函数，用于绘制场景。

        """
        self.children = sorted(
            self.children, key=lambda x: -x._color_lerp.val
        )  # 对指示器对象列表进行排序
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)  # 清除颜色缓冲区和深度缓冲区
        glLoadIdentity()  # 重置模型视图矩阵
        self._refresh()  # 刷新OpenGL视口
        self._draw()  # 绘制音频频谱图
        for obj in self.children:
            obj.draw()  # 绘制指示器对象
        glutSwapBuffers()  # 交换前后缓冲区

    def _tick(self):
        """
        内部使用的计时函数。

        """
        pass

    def tick(self):
        """
        计时函数，用于更新场景。

        """
        self._tick()  # 内部使用的计时函数
        for obj in self.children:
            obj.tick()  # 更新指示器对象

    def _do_tick(self, hz=60):
        """
        执行计时循环。

        参数：
            hz: 每秒计时次数

        """
        delay = 1 / hz  # 每次计时的延迟时间
        while True:
            a = time.time()  # 记录起始时间
            self.tick()  # 执行计时函数
            dt = time.time() - a  # 计算实际执行时间
            time.sleep(max(0, delay - dt))  # 控制计时周期

    def _draw(self):
        """
        绘制音频频谱图。

        """
        draw_spectrogram(self.label_client.last_data)  # 调用绘制音频频谱图的函数

    def run(self):
        """
        运行应用程序。

        """
        threading.Thread(target=self._do_tick).start()  # 启动计时循环的线程
        glutMainLoop()  # 进入OpenGL主循环


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server-endpoint",
        type=str,
        default="http://127.0.0.1:16888",
        help="The endpoint to use",
    )
    flags = parser.parse_args()
    app = DemoApplication(LabelClient(flags.server_endpoint))  # 创建应用程序对象
    app.run()  # 运行应用程序


if __name__ == "__main__":
    main()
