# -*- coding: utf-8 -*-
# Copyright 2023 XuMing(xuming624@qq.com) and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
配置数据类：包含模型、数据、训练和脚本参数的定义
"""

from dataclasses import dataclass, field
from typing import Literal, Optional

import torch
from transformers import Seq2SeqTrainingArguments


@dataclass
class ModelArguments:
    """
    模型相关的参数配置
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "模型检查点的路径，用于权重初始化。如果要从头训练模型，则不要设置此项。"
            )
        },
    )
    load_in_8bit: bool = field(default=False, metadata={"help": "是否以8位模式加载模型"})
    load_in_4bit: bool = field(default=False, metadata={"help": "是否以4位模式加载模型"})
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "分词器的路径，用于权重初始化。如果要从头训练模型，则不要设置此项。"
            )
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "存储从 huggingface.co 下载的预训练模型的目录"},
    )
    model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "要使用的特定模型版本（可以是分支名、标签名或提交ID）"},
    )
    hf_hub_token: Optional[str] = field(default=None, metadata={"help": "登录 Hugging Face Hub 的认证令牌"})
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "是否使用快速分词器（由 tokenizers 库支持）"},
    )
    torch_dtype: Optional[str] = field(
        default="float16",
        metadata={
            "help": (
                "覆盖默认的 `torch.dtype` 并在此数据类型下加载模型。如果传递 `auto`，"
                "数据类型将从模型的权重中自动推导。"
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    device_map: Optional[str] = field(
        default="auto",
        metadata={"help": "将模型映射到的设备。如果传递 `auto`，设备将自动选择。"},
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "从远程检查点加载模型时是否信任远程代码"},
    )
    rope_scaling: Optional[Literal["linear", "dynamic"]] = field(
        default=None,
        metadata={"help": "采用缩放的旋转位置编码（RoPE）"}
    )
    flash_attn: Optional[bool] = field(
        default=False,
        metadata={"help": "启用 FlashAttention-2 以加快训练速度"}
    )
    shift_attn: Optional[bool] = field(
        default=False,
        metadata={"help": "启用 LongLoRA 提出的移位稀疏注意力（S^2-Attn）"}
    )
    neft_alpha: Optional[float] = field(
        default=0,
        metadata={"help": "控制 NEFTune 中噪声幅度的 alpha 参数，值可以设为 5"}
    )

    def __post_init__(self):
        if self.model_name_or_path is None:
            raise ValueError("必须指定有效的 model_name_or_path 才能运行训练")


@dataclass
class DataArguments:
    """
    数据相关的参数配置
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "要使用的数据集名称（通过 datasets 库）"}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "要使用的数据集配置名称（通过 datasets 库）"}
    )
    train_file_dir: Optional[str] = field(default=None, metadata={"help": "训练 jsonl 数据文件目录"})
    validation_file_dir: Optional[str] = field(default=None, metadata={"help": "评估 jsonl 数据文件目录"})
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "出于调试目的或更快训练，将训练示例数量截断到此值（如果设置）"
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "出于调试目的或更快训练，将评估示例数量截断到此值（如果设置）"
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={"help": "是否只忽略填充 token。这假设已定义 `config.pad_token_id`。"},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "覆盖缓存的训练和评估集"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "如果没有验证集，用作验证集的训练集百分比"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=0,
        metadata={"help": "用于预处理的进程数"},
    )

    def __post_init__(self):
        from loguru import logger
        if self.max_train_samples is not None and 0 < self.max_train_samples <= 1000:
            logger.warning("在生产环境中，可以将 max_train_samples 设置为 -1 以运行所有样本")


@dataclass
class ScriptArguments:
    """
    脚本特定的参数配置
    """

    use_peft: bool = field(default=True, metadata={"help": "是否使用 PEFT（参数高效微调）"})
    train_on_inputs: bool = field(default=False, metadata={"help": "是否在输入上进行训练"})
    target_modules: Optional[str] = field(default="all", metadata={"help": "LoRA 目标模块，用逗号分隔，'all' 表示自动查找所有线性层"})
    lora_rank: Optional[int] = field(default=8, metadata={"help": "LoRA 秩"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "LoRA dropout 比率"})
    lora_alpha: Optional[float] = field(default=32.0, metadata={"help": "LoRA alpha 参数"})
    modules_to_save: Optional[str] = field(default=None, metadata={"help": "除了 LoRA 适配器外还要保存的模块，用逗号分隔"})
    peft_path: Optional[str] = field(default=None, metadata={"help": "PEFT 模型的路径"})
    qlora: bool = field(default=False, metadata={"help": "是否使用 QLoRA（4位量化+LoRA）"})
    model_max_length: int = field(
        default=512,
        metadata={"help": "模型最大上下文长度。建议值：8192 * 4, 8192 * 2, 8192, 4096, 2048, 1024, 512"}
    )
    template_name: Optional[str] = field(default="vicuna", metadata={"help": "对话模板名称"})

    def __post_init__(self):
        if self.model_max_length < 60:
            raise ValueError("必须指定有效的 model_max_length >= 60 才能运行训练")
