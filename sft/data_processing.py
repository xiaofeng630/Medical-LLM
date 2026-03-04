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
数据加载和预处理工具，用于监督微调（SFT）
"""

from glob import glob
from typing import List, Optional

from datasets import Dataset, load_dataset
from loguru import logger

from config import DataArguments, ScriptArguments


def load_datasets(
    data_args: DataArguments,
    cache_dir: Optional[str] = None
) -> Dataset:
    """
    从本地文件或 HuggingFace Hub 加载训练和验证数据集

    Args:
        data_args: 数据配置参数
        cache_dir: 数据集缓存目录

    Returns:
        包含训练和验证分割的已加载数据集
    """
    if data_args.dataset_name is not None:
        # 从 HuggingFace Hub 下载数据集
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=cache_dir,
        )
        # 如果没有验证集，则从训练集中分割
        if "validation" not in raw_datasets.keys():
            raw_datasets = _split_train_to_validation(
                raw_datasets,
                data_args.validation_split_percentage
            )
    else:
        # 从本地文件加载数据集
        data_files = _get_data_files(data_args)
        raw_datasets = load_dataset(
            'json',
            data_files=data_files,
            cache_dir=cache_dir,
        )
        # 如果没有验证集，则从训练集中分割
        if "validation" not in raw_datasets.keys():
            raw_datasets = _split_train_to_validation(
                raw_datasets,
                data_args.validation_split_percentage
            )

    logger.info(f"原始数据集: {raw_datasets}")
    return raw_datasets


def _get_data_files(data_args: DataArguments) -> dict:
    """
    收集训练和验证数据文件的路径

    Args:
        data_args: 数据配置参数

    Returns:
        包含训练和验证文件路径的字典
    """
    data_files = {}

    if data_args.train_file_dir is not None:
        # 查找所有训练数据文件（支持 .json 和 .jsonl）
        train_data_files = glob(
            f'{data_args.train_file_dir}/**/*.json', recursive=True
        ) + glob(
            f'{data_args.train_file_dir}/**/*.jsonl', recursive=True
        )
        logger.info(f"训练文件: {train_data_files}")
        data_files["train"] = train_data_files

    if data_args.validation_file_dir is not None:
        # 查找所有验证数据文件
        eval_data_files = glob(
            f'{data_args.validation_file_dir}/**/*.json', recursive=True
        ) + glob(
            f'{data_args.validation_file_dir}/**/*.jsonl', recursive=True
        )
        logger.info(f"评估文件: {eval_data_files}")
        data_files["validation"] = eval_data_files

    return data_files


def _split_train_to_validation(
    raw_datasets: Dataset,
    validation_split_percentage: float
) -> Dataset:
    """
    将训练数据集分割为训练集和验证集

    Args:
        raw_datasets: 原始数据集
        validation_split_percentage: 用作验证集的百分比

    Returns:
        分割后的数据集，包含 train 和 validation 键
    """
    shuffled_train_dataset = raw_datasets["train"].shuffle(seed=42)
    split = shuffled_train_dataset.train_test_split(
        test_size=validation_split_percentage / 100,
        seed=42
    )
    raw_datasets["train"] = split["train"]
    raw_datasets["validation"] = split["test"]
    return raw_datasets


class DataPreprocessor:
    """
    SFT 训练数据的预处理器

    负责将对话数据转换为模型训练所需的格式
    """

    def __init__(
        self,
        tokenizer,
        prompt_template,
        script_args: ScriptArguments,
        ignore_index: int
    ):
        """
        初始化数据预处理器

        Args:
            tokenizer: 分词器实例
            prompt_template: 对话模板
            script_args: 包含 model_max_length 和 train_on_inputs 的脚本参数
            ignore_index: 损失计算时忽略的标签索引
        """
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template
        self.max_length = script_args.model_max_length
        self.train_on_inputs = script_args.train_on_inputs
        self.ignore_index = ignore_index
        self.roles = ["human", "gpt"]  # 对话角色：用户和助手

    def preprocess_function(self, examples):
        """
        预处理数据集示例用于训练

        部分代码修改自 https://github.com/lm-sys/FastChat

        Args:
            examples: 包含多个对话样本的批次

        Returns:
            包含 input_ids, attention_mask, labels 的字典
        """
        input_ids_list = []
        attention_mask_list = []
        targets_list = []

        # 遍历处理每个对话
        for dialog in self._get_dialogs(examples):
            input_ids, labels = self._process_dialog(dialog)
            input_ids_list.append(input_ids)
            attention_mask_list.append([1] * len(input_ids))
            targets_list.append(labels)

        return dict(
            input_ids=input_ids_list,
            attention_mask=attention_mask_list,
            labels=targets_list,
        )

    def _get_dialogs(self, examples):
        """
        从对话示例中生成对话字符串

        Args:
            examples: 包含 conversations 字段的示例批次

        Yields:
            对话字符串列表（交替的用户和助手消息）
        """
        system_prompts = examples.get("system_prompt", "")

        for i, source in enumerate(examples['conversations']):
            if len(source) < 2:
                continue

            system_prompt = ""
            messages = []

            # 处理系统提示词
            data_role = source[0].get("from", "")
            if data_role == "system":
                system_prompt = source[0]["value"]
                source = source[1:]
                data_role = source[0].get("from", "")

            # 如果第一条消息不是来自用户，则跳过
            if data_role not in self.roles or data_role != self.roles[0]:
                source = source[1:]

            if len(source) < 2:
                continue

            # 处理对话消息
            for j, sentence in enumerate(source):
                data_role = sentence.get("from", "")
                if data_role not in self.roles:
                    logger.warning(f"未知角色: {data_role}, {i}. (已忽略)")
                    break
                if data_role == self.roles[j % 2]:
                    messages.append(sentence["value"])

            # 确保消息数量为偶数（成对出现）
            if len(messages) % 2 != 0:
                continue

            # 转换为对话对
            history_messages = [
                [messages[k], messages[k + 1]]
                for k in range(0, len(messages), 2)
            ]

            # 使用对话特定的或全局系统提示词
            if not system_prompt:
                system_prompt = system_prompts[i] if system_prompts else ""

            yield self.prompt_template.get_dialog(
                history_messages,
                system_prompt=system_prompt
            )

    def _process_dialog(self, dialog: List[str]) -> tuple:
        """
        处理单个对话，生成 input_ids 和 labels

        Args:
            dialog: 对话字符串列表（交替的用户/助手消息）

        Returns:
            元组 (input_ids, labels)
        """
        input_ids = []
        labels = []

        for i in range(len(dialog) // 2):
            # 分词用户消息
            source_ids = self.tokenizer.encode(
                text=dialog[2 * i],
                add_special_tokens=(i == 0)  # 只在第一个用户消息时添加特殊 token
            )
            # 分词助手回复
            target_ids = self.tokenizer.encode(
                text=dialog[2 * i + 1],
                add_special_tokens=False
            )

            # 计算按比例的长度限制
            total_len = len(source_ids) + len(target_ids)
            max_source_len = int(self.max_length * (len(source_ids) / total_len))
            max_target_len = int(self.max_length * (len(target_ids) / total_len))

            # 必要时截断
            source_ids = self._truncate_source(source_ids, max_source_len)
            target_ids = self._truncate_target(target_ids, max_target_len)

            # 检查添加此轮对话是否会超过最大长度
            if len(input_ids) + len(source_ids) + len(target_ids) + 1 > self.max_length:
                break

            # 添加到序列
            eos_id = self.tokenizer.eos_token_id
            input_ids += source_ids + target_ids + [eos_id]

            if self.train_on_inputs:
                # 在输入上训练：用户和助手部分都有损失
                labels += source_ids + target_ids + [eos_id]
            else:
                # 只在输出上训练：用户部分损失被忽略
                labels += [self.ignore_index] * len(source_ids) + target_ids + [eos_id]

        return input_ids, labels

    def _truncate_source(self, source_ids: List[int], max_len: int) -> List[int]:
        """
        截断源（用户）token

        Args:
            source_ids: 源 token 列表
            max_len: 最大长度

        Returns:
            截断后的 token 列表
        """
        if len(source_ids) > max_len:
            source_ids = source_ids[:max_len]
        # 移除开头的 EOS token（如果有）
        if len(source_ids) > 0 and source_ids[0] == self.tokenizer.eos_token_id:
            source_ids = source_ids[1:]
        return source_ids

    def _truncate_target(self, target_ids: List[int], max_len: int) -> List[int]:
        """
        截断目标（助手回复）token（为 EOS token 预留空间）

        Args:
            target_ids: 目标 token 列表
            max_len: 最大长度

        Returns:
            截断后的 token 列表
        """
        # 预留一个位置给 EOS token
        if len(target_ids) > max_len - 1:
            target_ids = target_ids[:max_len - 1]
        # 移除结尾的 EOS token（如果有），我们会手动添加
        if len(target_ids) > 0 and target_ids[-1] == self.tokenizer.eos_token_id:
            target_ids = target_ids[:-1]
        return target_ids


def filter_empty_labels(example, ignore_index: int) -> bool:
    """
    过滤掉所有标签都被忽略的示例

    Args:
        example: 包含 labels 的数据集示例
        ignore_index: 表示被忽略标签的索引值

    Returns:
        如果示例至少有一个未被忽略的标签，则返回 True
    """
    return not all(label == ignore_index for label in example["labels"])


def prepare_train_dataset(
    raw_datasets: Dataset,
    data_args: DataArguments,
    script_args: ScriptArguments,
    training_args,
    preprocessor: DataPreprocessor,
    is_main_process: bool
) -> tuple:
    """
    准备训练数据集

    Args:
        raw_datasets: 包含 train 分割的原始数据集
        data_args: 数据参数
        script_args: 脚本参数
        training_args: 训练参数
        preprocessor: 数据预处理器实例
        is_main_process: 是否为主进程

    Returns:
        元组 (train_dataset, max_train_samples)
    """
    if "train" not in raw_datasets:
        raise ValueError("--do_train 需要训练数据集")

    train_dataset = raw_datasets['train'].shuffle(seed=42)
    max_train_samples = len(train_dataset)

    # 如果指定了最大训练样本数，则截断
    if data_args.max_train_samples is not None and data_args.max_train_samples > 0:
        max_train_samples = min(len(train_dataset), data_args.max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))

    if is_main_process:
        logger.debug(f"训练数据集示例[0]: {train_dataset[0]}")

    # 分词并处理数据集
    with training_args.main_process_first(desc="训练数据集分词"):
        tokenized_dataset = train_dataset.map(
            preprocessor.preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=train_dataset.column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="正在对数据集进行分词" if is_main_process else None,
        )

        # 过滤掉空标签的样本
        train_dataset = tokenized_dataset.filter(
            lambda x: filter_empty_labels(x, preprocessor.ignore_index),
            num_proc=data_args.preprocessing_num_workers
        )

    if is_main_process:
        logger.debug(f"训练样本数: {len(train_dataset)}")
        logger.debug("分词后的训练示例:")
        logger.debug(f"解码 input_ids[0]:\n{preprocessor.tokenizer.decode(train_dataset[0]['input_ids'])}")
        # 将忽略的标签替换为 pad_token_id 以便解码查看
        replaced_labels = [
            label if label != preprocessor.ignore_index else preprocessor.tokenizer.pad_token_id
            for label in list(train_dataset[0]['labels'])
        ]
        logger.debug(f"解码 labels[0]:\n{preprocessor.tokenizer.decode(replaced_labels)}")

    return train_dataset, max_train_samples


def prepare_eval_dataset(
    raw_datasets: Dataset,
    data_args: DataArguments,
    script_args: ScriptArguments,
    training_args,
    preprocessor: DataPreprocessor
) -> tuple:
    """
    准备评估数据集

    Args:
        raw_datasets: 包含 validation 分割的原始数据集
        data_args: 数据参数
        script_args: 脚本参数
        training_args: 训练参数
        preprocessor: 数据预处理器实例

    Returns:
        元组 (eval_dataset, max_eval_samples)
    """
    if "validation" not in raw_datasets:
        raise ValueError("--do_eval 需要验证数据集")

    eval_dataset = raw_datasets["validation"]
    max_eval_samples = len(eval_dataset)

    # 如果指定了最大评估样本数，则截断
    if data_args.max_eval_samples is not None and data_args.max_eval_samples > 0:
        max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
        eval_dataset = eval_dataset.select(range(max_eval_samples))

    eval_size = len(eval_dataset)
    logger.debug(f"评估样本数: {eval_size}")

    if eval_size > 500:
        logger.warning(
            f"评估样本数较大: {eval_size}, "
            f"训练会变慢，建议通过 `--max_eval_samples=50` 减少评估样本数"
        )

    logger.debug(f"评估数据集示例[0]: {eval_dataset[0]}")

    # 分词评估数据集
    eval_dataset = eval_dataset.map(
        preprocessor.preprocess_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=eval_dataset.column_names,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="正在对验证数据集进行分词",
    )

    # 过滤掉空标签的样本
    eval_dataset = eval_dataset.filter(
        lambda x: filter_empty_labels(x, preprocessor.ignore_index),
        num_proc=data_args.preprocessing_num_workers
    )

    logger.debug(f"评估样本数: {len(eval_dataset)}")
    logger.debug("分词后的评估示例:")
    logger.debug(preprocessor.tokenizer.decode(eval_dataset[0]['input_ids']))

    return eval_dataset, max_eval_samples
