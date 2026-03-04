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
训练器类和监督微调的模型保存工具
"""

import math
import os

import torch
from loguru import logger
from transformers import DataCollatorForSeq2Seq, Trainer
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.trainer import TRAINING_ARGS_NAME


class SavePeftModelTrainer(Trainer):
    """
    用于 LoRA 模型的训练器，处理模型保存
    """

    def save_model(self, output_dir=None, _internal_call=False):
        """
        保存 LoRA 模型

        Args:
            output_dir: 输出目录
            _internal_call: 是否为内部调用
        """
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        self.model.save_pretrained(output_dir)


def save_model(model, tokenizer, args):
    """
    保存模型和分词器

    Args:
        model: 要保存的模型
        tokenizer: 要保存的分词器
        args: 包含输出目录的训练参数
    """
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # 处理分布式/并行训练
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def save_model_zero3(model, tokenizer, args, trainer):
    """
    为 DeepSpeed ZeRO-3 保存模型

    参考:
    https://github.com/lm-sys/FastChat/blob/main/fastchat/train/train_lora.py#L209

    Args:
        model: 要保存的模型
        tokenizer: 要保存的分词器
        args: 包含输出目录的训练参数
        trainer: 包含 ZeRO-3 状态字典的训练器实例
    """
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    state_dict_zero3 = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(args.output_dir, state_dict=state_dict_zero3)
    tokenizer.save_pretrained(output_dir)


class TrainerFactory:
    """
    用于创建训练器实例的工厂类
    """

    @staticmethod
    def create_trainer(
        model,
        tokenizer,
        training_args,
        train_dataset,
        eval_dataset,
        ignore_index: int
    ) -> SavePeftModelTrainer:
        """
        创建训练器实例

        Args:
            model: 要训练的模型
            tokenizer: 分词器
            training_args: 训练参数
            train_dataset: 训练数据集（可以为 None）
            eval_dataset: 评估数据集（可以为 None）
            ignore_index: 填充时忽略的标签索引

        Returns:
            配置好的训练器实例
        """
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            label_pad_token_id=ignore_index,
            pad_to_multiple_of=4 if tokenizer.padding_side == "right" else None,
        )

        trainer = SavePeftModelTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            processing_class=tokenizer,
            data_collator=data_collator,
        )

        return trainer


class TrainingRunner:
    """
    用于训练和评估操作的运行器
    """

    def __init__(self, trainer, tokenizer, ignore_index):
        """
        初始化训练运行器

        Args:
            trainer: 训练器实例
            tokenizer: 分词器
            ignore_index: 填充时忽略的标签索引
        """
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index

    def run_training(self, max_train_samples: int):
        """
        运行训练循环

        Args:
            max_train_samples: 处理的训练样本数

        Returns:
            训练结果
        """
        if not self.trainer.args.do_train:
            return None

        if self.trainer.is_world_process_zero():
            logger.info("*** 开始训练 ***")
            self._log_train_sample()

        checkpoint = None
        if self.trainer.args.resume_from_checkpoint is not None:
            checkpoint = self.trainer.args.resume_from_checkpoint

        train_result = self.trainer.train(resume_from_checkpoint=checkpoint)

        # 处理并保存指标
        self._process_train_metrics(train_result, max_train_samples)

        # 保存模型
        if self.trainer.is_world_process_zero():
            logger.info(f"保存模型检查点到 {self.trainer.args.output_dir}")
            self._save_model()

        return train_result

    def _log_train_sample(self):
        """
        记录训练数据加载器的样本用于调试
        """
        sample = next(iter(self.trainer.get_train_dataloader()))
        logger.debug(f"训练数据加载器示例: {sample}")
        logger.debug(
            f"input_ids:\n{list(sample['input_ids'])[:3]}, "
            f"\nlabels:\n{list(sample['labels'])[:3]}"
        )
        logger.debug(f"解码 input_ids[0]:\n{self.tokenizer.decode(sample['input_ids'][0])}")
        replaced_labels = [
            label if label != self.ignore_index else self.tokenizer.pad_token_id
            for label in sample['labels'][0]
        ]
        logger.debug(f"解码 labels[0]:\n{self.tokenizer.decode(replaced_labels)}")

    def _process_train_metrics(self, train_result, max_train_samples):
        """
        处理并记录训练指标

        Args:
            train_result: 训练结果
            max_train_samples: 最大训练样本数
        """
        metrics = train_result.metrics
        metrics["train_samples"] = max_train_samples
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        self.trainer.save_state()

        # 训练后启用缓存
        self.trainer.model.config.use_cache = True
        self.tokenizer.padding_side = "left"
        self.tokenizer.init_kwargs["padding_side"] = "left"

    def _save_model(self):
        """
        保存训练好的模型
        """
        if is_deepspeed_zero3_enabled():
            model = self.trainer.model.module if hasattr(self.trainer.model, "module") else self.trainer.model
            save_model_zero3(model, self.tokenizer, self.trainer.args, self.trainer)
        else:
            model = self.trainer.model.module if hasattr(self.trainer.model, "module") else self.trainer.model
            save_model(model, self.tokenizer, self.trainer.args)

    def run_evaluation(self, max_eval_samples: int):
        """
        运行评估循环

        Args:
            max_eval_samples: 处理的评估样本数

        Returns:
            评估指标
        """
        if not self.trainer.args.do_eval:
            return None

        if self.trainer.is_world_process_zero():
            logger.info("*** 开始评估 ***")

        metrics = self.trainer.evaluate(metric_key_prefix="eval")
        metrics["eval_samples"] = max_eval_samples

        # 计算困惑度
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        self.trainer.log_metrics("eval", metrics)
        self.trainer.save_metrics("eval", metrics)

        if self.trainer.is_world_process_zero():
            logger.debug(f"评估指标: {metrics}")

        return metrics
