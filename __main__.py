import argparse
import server
import json
import os

import argparse  # 导入 argparse 模块，用于处理命令行参数
import os  # 导入 os 模块，用于处理文件和目录路径
import json  # 导入 json 模块，用于处理 JSON 数据

def main():
    parser = argparse.ArgumentParser()  # 创建一个 ArgumentParser 对象，用于解析命令行参数
    parser.add_argument(
        "--config",  # 添加一个名为 "--config" 的命令行参数
        type=str,
        default="",  # 如果命令行中没有提供 "--config" 参数，则使用空字符串作为默认值
        help="The config file to use")  # 设置关于 "--config" 参数的帮助信息
    flags, _ = parser.parse_known_args()  # 解析命令行参数并将结果存储在 flags 变量中
    if not flags.config:  # 如果 "--config" 参数没有提供值
        flags.config = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
        # 使用当前脚本文件的路径和 "config.json" 的文件名构建配置文件的路径
    with open(flags.config) as f:  # 打开配置文件
        config = json.loads(f.read())  # 读取配置文件内容并将其解析为 JSON 格式
    server.start(config)  # 使用解析后的配置启动服务器

if __name__ == "__main__":
    main()  # 调用 main() 函数，开始执行程序
