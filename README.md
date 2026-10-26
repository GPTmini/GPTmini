# From Neuron to GPT - 从零开始大模型

一套从单个神经元出发，只用 NumPy 一步步手搓出 GPT 的中文教程。

不依赖 PyTorch、TensorFlow 等深度学习框架：自动求导、网络层、优化器、注意力、Transformer、训练、推理、微调，全部从零实现，并配有逐步讲解的Jupyer Notebook。

## 你将学到什么

教程共 18 章加一篇框架总结，由浅入深：

### 第一部分：神经网络基础

| 章节            | 内容                                        |
|---------------|-------------------------------------------|
| 多层神经网络（01-10） | 线性回归、损失函数、梯度下降、自动微分、学习率、迭代、批次、轮次、隐藏层、激活函数 |
| 循环神经网络（11-16） | 编码、嵌入、交叉熵损失、循环网络、Adam 优化器、学习率调度器          |
| 核心框架          | 总结完整的核心深度学习框架                             |

### 第二部分：大语言模型基础

| 章节            | 内容                                   |
|---------------|--------------------------------------|
| 编码器（01-02）    | 词元嵌入、位置嵌入                            |
| 自注意力机制（03-06） | 自注意力、因果注意力、加权注意力、多头注意力               |
| 预训练（07-10）    | 残差连接、归一化、随机失活、模型持久化                  |
| 推理（11-14）     | 采样策略（温度、Top-k、Top-p）、停止条件、KV 缓存、批量推理 |
| 微调（15-18）     | 监督微调（SFT）、直接偏好优化（DPO）                |

## 项目结构│

```text
.
├── data/                       # 训练数据（tinyshakespeare）及训练产生的模型参数文件
├── src/                        
│   ├── core/                   # 核心框架：张量、网络层、激活函数、损失函数、优化器、学习率调度器、数据集、训练器
│   ├── gpt/                    # GPT 模型、数据集、损失函数、训练器
│   ├── sft/                    # 监督微调
│   └── dpo/                    # 直接偏好优化
├── tutorial/zh/                # 中文教程（Jupyter Notebook）
│   ├── artificial-neural-network/
│   └── large-language-model/
├── myst.yml                    # 教程站点（MyST / Jupyter Book）配置
└── requirements.txt
```

## 快速开始

### 环境要求

* Python 3.10 及以上
* 依赖：`numpy`、`matplotlib`、`requests`、`jupyter-book`

### 安装

```bash
git clone https://github.com/gptmini/from-neuron-to-gpt.git
cd from-neuron-to-gpt

python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
```

### 运行教程

Notebook 中通过 `from src.core import *` 的方式导入代码，因此请在**项目根目录**启动 Jupyter，或在 VS Code 中打开项目根目录后运行：

```bash
pip install jupyterlab
jupyter lab
```

然后按编号顺序打开 `tutorial/zh` 下的 Notebook 即可。

## 推荐学习路线

1. 先依次学习第一部分，神经网络基础，下的 01-多层神经网络、02-循环神经网络和 03-训练框架总结，理解自动微分、计算图、训练循环等基础概念；
2. 再依次学习第二部分，大语言模型基础。下的 5 个部分：编码器、注意力、预训练、推理和微调；
3. 每章末尾的课后练习建议动手完成，并观察对比结果。

大语言模型部分默认使用 `data/tinyshakespeare.txt` 作为训练数据。预训练、SFT、DPO 会依次生成 3 个模型参数文件，后一阶段依赖前一阶段的模型参数文件，请按顺序运行。

全部代码基于 Mac mini 4M / 16G 开发完成，完全运行在 CPU 上。

“模型持久化”章节使用以下列出的模型超参数配置，目的在于训练一个在推理和微调阶段可用的模型，需要训练 1-2 小时。

## 模型规模

默认的 GPT 模型配置如下：

| 参数 |   值 |
| --- |----:|
| 上下文尺寸 | 256 |
| 嵌入尺寸 | 384 |
| 注意力头数 |   6 |
| Transformer 模块数 |   6 |
| 批大小|  16 |
| 训练轮次 |  20 |

模型完全在 CPU 上训练，规模较小，目的是看清每一步的原理，而不是追求生成质量。

## 许可证

本项目基于 [MIT License](LICENSE) 开源。