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
监督微调主入口文件：对因果语言模型（GPT, LLaMA, Bloom 等）进行微调

支持从 json 文件或数据集进行微调

部分代码修改自 https://github.com/shibing624/textgen
"""

import os
import sys

import torch
from loguru import logger
from transformers import HfArgumentParser, Seq2SeqTrainingArguments, set_seed
from transformers.trainer_pt_utils import LabelSmoother

from template import get_conv_template

# 导入本地模块
from config import ModelArguments, DataArguments, ScriptArguments
from data_processing import (
    DataPreprocessor,
    load_datasets,
    prepare_train_dataset,
    prepare_eval_dataset,
)
from model_utils import (
    TokenizerManager,
    ModelLoader,
    PEFTManager,
    print_trainable_parameters,
    check_and_optimize_memory,
)
from trainer import TrainerFactory, TrainingRunner


def parse_arguments():
    """
    解析命令行参数

    Returns:
        元组 (model_args, data_args, training_args, script_args)
    """
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, Seq2SeqTrainingArguments, ScriptArguments)
    )

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # 从 JSON 文件加载
        return parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        # 解析命令行参数
        return parser.parse_args_into_dataclasses(look_for_args_file=False)


def setup_environment(training_args, model_args, data_args, script_args):
    """
    设置训练环境

    Args:
        training_args: 训练参数
        model_args: 模型参数
        data_args: 数据参数
        script_args: 脚本参数

    Returns:
        元组 (is_main_process, tokenizer, prompt_template, raw_datasets)
    """
    # 处理 DeepSpeed 配置
    if training_args.deepspeed is not None:
        training_args.distributed_state.deepspeed_plugin = None

    # 判断是否为主进程
    is_main_process = training_args.local_rank in [-1, 0]

    # 在主进程上记录配置
    if is_main_process:
        logger.info(f"模型参数: {model_args}")
        logger.info(f"数据参数: {data_args}")
        logger.info(f"训练参数: {training_args}")
        logger.info(f"脚本参数: {script_args}")
        logger.info(
            f"进程秩: {training_args.local_rank}, 设备: {training_args.device}, "
            f"n_gpu: {training_args.n_gpu}"
            f" 分布式训练: {bool(training_args.local_rank != -1)}, "
            f"16位训练: {training_args.fp16}"
        )

    # 设置随机种子以保证可重复性
    set_seed(training_args.seed)

    # 检查并优化 GPU 内存
    check_and_optimize_memory()

    # 加载分词器
    tokenizer_manager = TokenizerManager(model_args, script_args)
    tokenizer = tokenizer_manager.load_tokenizer()

    # 获取对话模板
    prompt_template = get_conv_template(script_args.template_name)

    # 加载数据集
    raw_datasets = load_datasets(data_args, model_args.cache_dir)

    return is_main_process, tokenizer, prompt_template, raw_datasets


def prepare_datasets(
    training_args,
    data_args,
    script_args,
    raw_datasets,
    tokenizer,
    prompt_template,
    is_main_process
):
    """
    准备训练和评估数据集

    Args:
        training_args: 训练参数
        data_args: 数据参数
        script_args: 脚本参数
        raw_datasets: 原始数据集
        tokenizer: 分词器
        prompt_template: 对话模板
        is_main_process: 是否为主进程

    Returns:
        元组 (train_dataset, eval_dataset, max_train_samples, max_eval_samples, ignore_index)
    """
    # 计算标签的忽略索引
    ignore_index = (
        LabelSmoother.ignore_index
        if data_args.ignore_pad_token_for_loss
        else tokenizer.pad_token_id
    )

    # 创建数据预处理器
    preprocessor = DataPreprocessor(tokenizer, prompt_template, script_args, ignore_index)

    # 准备数据集
    train_dataset = None
    eval_dataset = None
    max_train_samples = 0
    max_eval_samples = 0

    if training_args.do_train:
        train_dataset, max_train_samples = prepare_train_dataset(
            raw_datasets,
            data_args,
            script_args,
            training_args,
            preprocessor,
            is_main_process
        )

    if training_args.do_eval:
        eval_dataset, max_eval_samples = prepare_eval_dataset(
            raw_datasets,
            data_args,
            script_args,
            training_args,
            preprocessor
        )

    return train_dataset, eval_dataset, max_train_samples, max_eval_samples, ignore_index


def load_and_prepare_model(
    model_args,
    script_args,
    training_args,
    tokenizer
):
    """
    加载并准备模型用于训练

    Args:
        model_args: 模型参数
        script_args: 脚本参数
        training_args: 训练参数
        tokenizer: 分词器

    Returns:
        准备好的模型
    """
    # 加载模型
    model_loader = ModelLoader(model_args, script_args, training_args)
    model = model_loader.load_model()

    # 如果启用 PEFT，则应用 PEFT
    if script_args.use_peft:
        peft_manager = PEFTManager(model_args, script_args, training_args)
        model = peft_manager.apply_peft(model)
    else:
        logger.info("微调方法: 全参数训练")
        model = model.float()
        print_trainable_parameters(model)

    # 配置梯度检查点
    if training_args.gradient_checkpointing and getattr(model, "supports_gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        logger.info("已启用梯度检查点")
    else:
        model.config.use_cache = True
        logger.info("已禁用梯度检查点")

    model.enable_input_require_grads()

    # 处理多 GPU（非 DDP）情况
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if not (world_size != 1 or torch.cuda.device_count() <= 1):
        model.is_parallelizable = True
        model.model_parallel = True

    return model


def main():
    """
    监督微调的主入口点
    """
    # 解析参数
    model_args, data_args, training_args, script_args = parse_arguments()

    # 设置环境
    is_main_process, tokenizer, prompt_template, raw_datasets = setup_environment(
        training_args, model_args, data_args, script_args
    )

    # 准备数据集
    train_dataset, eval_dataset, max_train_samples, max_eval_samples, ignore_index = prepare_datasets(
        training_args, data_args, script_args, raw_datasets, tokenizer, prompt_template, is_main_process
    )

    # 加载并准备模型
    model = load_and_prepare_model(model_args, script_args, training_args, tokenizer)

    # 创建训练器
    trainer = TrainerFactory.create_trainer(
        model=model,
        tokenizer=tokenizer,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        ignore_index=ignore_index
    )

    # 运行训练
    if training_args.do_train:
        training_runner = TrainingRunner(trainer, tokenizer, ignore_index)
        training_runner.run_training(max_train_samples)

    # 运行评估
    if training_args.do_eval:
        training_runner = TrainingRunner(trainer, tokenizer, ignore_index)
        training_runner.run_evaluation(max_eval_samples)


if __name__ == "__main__":
    main()
