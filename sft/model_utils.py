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
模型加载、配置和监督微调工具函数
"""

import math
import os
from types import MethodType
from typing import Optional, Tuple

import torch
from loguru import logger
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.utils.versions import require_version

from config import ModelArguments, ScriptArguments


def print_trainable_parameters(model):
    """
    打印模型中可训练参数的数量

    Args:
        model: 要检查的模型
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"可训练参数: {trainable_params} || 总参数: {all_param} || "
        f"可训练比例: {100 * trainable_params / all_param:.2f}%"
    )


def find_all_linear_names(model, int4=False, int8=False):
    """
    查找模型中所有线性层的名称

    参考 QLoRA 论文

    Args:
        model: 要搜索的模型
        int4: 是否使用 4 位量化
        int8: 是否使用 8 位量化

    Returns:
        排序后的线性层名称列表
    """
    cls = torch.nn.Linear
    if int4 or int8:
        import bitsandbytes as bnb
        if int4:
            cls = bnb.nn.Linear4bit
        elif int8:
            cls = bnb.nn.Linear8bitLt

    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            # 跳过最后一层（lm_head 和 output_layer）
            if 'lm_head' in name or 'output_layer' in name:
                continue
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    return sorted(lora_module_names)


def check_and_optimize_memory():
    """
    检查并优化 GPU 内存使用
    """
    if not torch.cuda.is_available():
        return

    logger.info("检查 GPU 内存状态...")

    # 清空缓存
    torch.cuda.empty_cache()

    # 检查每个 GPU 的内存状态
    num_gpus = torch.cuda.device_count()
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        total_memory = props.total_memory / 1024 ** 3
        allocated = torch.cuda.memory_allocated(i) / 1024 ** 3
        cached = torch.cuda.memory_reserved(i) / 1024 ** 3
        free = total_memory - allocated - cached

        logger.info(f"GPU {i} ({props.name}):")
        logger.info(f"  总内存: {total_memory:.1f}GB")
        logger.info(f"  已分配: {allocated:.1f}GB")
        logger.info(f"  已缓存: {cached:.1f}GB")
        logger.info(f"  可用: {free:.1f}GB")

    # 启用内存优化
    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(True)
        logger.info("已启用 Flash Attention 优化")

    if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        logger.info("已启用内存高效注意力机制")


def log_model_distribution(model):
    """
    记录模型在设备上的分布情况

    Args:
        model: 要记录的模型
    """
    logger.info("模型分布:")

    if hasattr(model, 'hf_device_map') and model.hf_device_map:
        logger.info("使用 HuggingFace 设备映射:")
        for module_name, device in model.hf_device_map.items():
            logger.info(f"  {module_name}: {device}")

        # 统计每个设备上的模块数量
        device_count = {}
        for device in model.hf_device_map.values():
            device_str = str(device)
            device_count[device_str] = device_count.get(device_str, 0) + 1

        logger.info("设备使用统计:")
        for device, count in device_count.items():
            logger.info(f"  {device}: {count} 个模块")
    else:
        # 检查参数的设备分布
        device_params = {}
        total_params = 0
        for name, param in model.named_parameters():
            device = str(param.device)
            if device not in device_params:
                device_params[device] = {'count': 0, 'size': 0}
            device_params[device]['count'] += 1
            device_params[device]['size'] += param.numel()
            total_params += param.numel()

        logger.info("参数设备分布:")
        for device, info in device_params.items():
            param_size_gb = info['size'] * 4 / 1024 ** 3  # 假设 float32
            percentage = info['size'] / total_params * 100
            logger.info(
                f"  {device}: {info['count']} 个参数组, "
                f"{param_size_gb:.2f}GB ({percentage:.1f}%)"
            )


def log_gpu_memory():
    """
    记录 GPU 内存使用情况
    """
    if torch.cuda.is_available():
        logger.info("GPU 内存使用情况:")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024 ** 3
            cached = torch.cuda.memory_reserved(i) / 1024 ** 3
            total = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3
            logger.info(
                f"  GPU {i}: 已分配={allocated:.1f}GB, "
                f"已缓存={cached:.1f}GB, 总计={total:.1f}GB"
            )


class ModelLoader:
    """
    预训练语言模型的加载器，支持各种配置
    """

    def __init__(
        self,
        model_args: ModelArguments,
        script_args: ScriptArguments,
        training_args
    ):
        """
        初始化模型加载器

        Args:
            model_args: 模型配置参数
            script_args: 脚本特定参数
            training_args: 训练参数
        """
        self.model_args = model_args
        self.script_args = script_args
        self.training_args = training_args
        self.flash_attn_available = self._check_flash_attn()

    def _check_flash_attn(self) -> bool:
        """
        检查 FlashAttention-2 是否可用

        Returns:
            如果可用返回 True，否则返回 False
        """
        try:
            from flash_attn import flash_attn_func, flash_attn_varlen_func
            return True
        except ImportError:
            return False

    def load_model(self):
        """
        加载并配置模型

        Returns:
            加载的模型
        """
        # 获取 torch 数据类型
        torch_dtype = self._get_torch_dtype()

        # 分布式训练设置
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        ddp = world_size != 1
        if ddp:
            self.model_args.device_map = None
        if self.model_args.device_map in ["None", "none", ""]:
            self.model_args.device_map = None

        # 检查 QLoRA 兼容性
        if (self.script_args.qlora and
                (len(self.training_args.fsdp) > 0 or is_deepspeed_zero3_enabled())):
            logger.warning("FSDP 和 DeepSpeed ZeRO-3 目前与 QLoRA 不兼容")

        # 加载模型配置
        config = self._load_model_config()

        # 配置注意力机制
        self._configure_attention(config)

        # 获取量化配置
        quantization_config = self._get_quantization_config()

        # 准备模型参数
        model_kwargs = {
            "config": config,
            "torch_dtype": torch_dtype,
            "trust_remote_code": self.model_args.trust_remote_code,
            "quantization_config": quantization_config,
            "low_cpu_mem_usage": True,
            "device_map": self._get_device_map(),
        }

        # 为多 GPU（非 DDP）添加 max_memory
        if self.model_args.device_map == 'auto' and torch.cuda.device_count() > 1 and not ddp:
            model_kwargs["max_memory"] = self._get_max_memory_config()

        logger.info(f"模型训练配置: {model_kwargs}")

        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            self.model_args.model_name_or_path,
            **model_kwargs
        )

        logger.info("模型加载成功")

        # 记录模型分布
        log_model_distribution(model)
        log_gpu_memory()

        # 应用模型特定修复
        self._apply_model_fixes(model, config)

        # 应用 NEFTune（如果配置）
        self._apply_neftune(model)

        # 为 MoE 模型应用 DeepSpeed 补丁
        self._apply_deepspeed_patches(model, config)

        return model

    def _get_torch_dtype(self):
        """
        获取模型加载的 torch 数据类型

        Returns:
            torch 数据类型或 "auto"
        """
        if self.model_args.torch_dtype in ["auto", None]:
            return self.model_args.torch_dtype
        return getattr(torch, self.model_args.torch_dtype)

    def _load_model_config(self):
        """
        加载模型配置

        Returns:
            模型配置对象
        """
        config_kwargs = {
            "trust_remote_code": self.model_args.trust_remote_code,
            "cache_dir": self.model_args.cache_dir,
            "revision": self.model_args.model_revision,
            "token": self.model_args.hf_hub_token,
        }
        config = AutoConfig.from_pretrained(
            self.model_args.model_name_or_path,
            **config_kwargs
        )

        # 配置 RoPE 缩放
        self._configure_rope_scaling(config)

        return config

    def _configure_rope_scaling(self, config):
        """
        配置 RoPE（旋转位置编码）缩放

        Args:
            config: 模型配置对象
        """
        if self.model_args.rope_scaling is None:
            return

        if not hasattr(config, "rope_scaling"):
            logger.warning("当前模型不支持 RoPE 缩放")
            return

        if self.model_args.rope_scaling == "dynamic":
            logger.warning(
                "动态 NTK 在微调时可能效果不佳。"
                "详见: https://github.com/huggingface/transformers/pull/24653"
            )

        current_max_length = getattr(config, "max_position_embeddings", None)
        if current_max_length and self.script_args.model_max_length > current_max_length:
            scaling_factor = float(math.ceil(
                self.script_args.model_max_length / current_max_length
            ))
        else:
            logger.warning(
                f"model_max_length({self.script_args.model_max_length}) "
                f"小于最大长度({current_max_length})。"
                "建议增加 model_max_length。"
            )
            scaling_factor = 1.0

        setattr(
            config, "rope_scaling",
            {"type": self.model_args.rope_scaling, "factor": scaling_factor}
        )
        logger.info(
            f"使用 {self.model_args.rope_scaling} 缩放策略，"
            f"缩放因子设置为 {scaling_factor}"
        )

    def _configure_attention(self, config):
        """
        配置注意力机制（FlashAttention-2, S^2-Attn）

        Args:
            config: 模型配置对象
        """
        # FlashAttention-2
        if self.model_args.flash_attn:
            if self.flash_attn_available:
                self._flash_attn_enabled = True
                logger.info("使用 FlashAttention-2 以加快训练和推理速度")
            else:
                logger.warning("FlashAttention-2 未安装")

        # 移位稀疏注意力（S^2-Attn）
        if self.model_args.shift_attn:
            if getattr(config, "model_type", None) == "llama":
                setattr(config, "group_size_ratio", 0.25)
                logger.info("使用移位稀疏注意力，group_size_ratio=1/4")
            else:
                logger.warning("当前模型不支持移位稀疏注意力")

    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """
        获取量化配置（如果启用）

        Returns:
            量化配置对象，如果未启用则返回 None
        """
        load_in_4bit = self.model_args.load_in_4bit
        load_in_8bit = self.model_args.load_in_8bit

        if not (load_in_8bit or load_in_4bit):
            return None

        if load_in_8bit and load_in_4bit:
            raise ValueError("load_in_4bit 和 load_in_8bit 不能同时设置")

        logger.info(f"正在量化模型，load_in_4bit: {load_in_4bit}, load_in_8bit: {load_in_8bit}")

        if is_deepspeed_zero3_enabled():
            raise ValueError("DeepSpeed ZeRO-3 与量化不兼容")

        torch_dtype = self._get_torch_dtype()

        if load_in_8bit:
            return BitsAndBytesConfig(load_in_8bit=True)

        # load_in_4bit
        if self.script_args.qlora:
            # QLoRA 使用 NF4 量化
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        else:
            # 普通 4 位量化
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
            )

    def _get_device_map(self):
        """
        获取模型加载的设备映射

        Returns:
            设备映射字符串
        """
        if self.model_args.device_map == "auto" and torch.cuda.device_count() > 1:
            # 检查是否使用 DDP
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            if world_size == 1:
                return "auto"
        return self.model_args.device_map

    def _get_max_memory_config(self) -> dict:
        """
        获取多 GPU 的最大内存配置

        Returns:
            包含每个 GPU 最大内存的字典
        """
        max_memory = {}
        num_gpus = torch.cuda.device_count()

        for i in range(num_gpus):
            gpu_props = torch.cuda.get_device_properties(i)
            total_mem = gpu_props.total_memory
            # 为梯度和优化器状态预留 20% 内存
            usable_mem = int(total_mem * 0.8)
            max_memory[i] = f"{usable_mem // (1024 ** 3)}GiB"

        return max_memory

    def _apply_model_fixes(self, model, config):
        """
        应用模型特定的修复

        Args:
            model: 要修复的模型
            config: 模型配置
        """
        model_type = getattr(config, "model_type", None)

        # 修复 ChatGLM 和 ChatGLM2 的 LM head
        if model_type in ["chatglm", "internlm2"]:
            setattr(model, "lm_head", model.transformer.output_layer)
            setattr(model, "_keys_to_ignore_on_save", ["lm_head.weight"])

    def _apply_neftune(self, model):
        """
        应用 NEFTune（噪声嵌入微调）（如果配置）

        Args:
            model: 要应用 NEFTune 的模型
        """
        if self.model_args.neft_alpha <= 0:
            return

        input_embed = model.get_input_embeddings()
        if not isinstance(input_embed, torch.nn.Embedding):
            logger.warning(
                "输入嵌入不是标准的 nn.Embedding，"
                "无法转换为噪声嵌入"
            )
            return

        def noisy_forward(self: torch.nn.Embedding, x: torch.Tensor) -> torch.Tensor:
            embeddings = input_embed.__class__.forward(self, x)
            dims = self.num_embeddings * self.embedding_dim
            mag_norm = self.model_args.neft_alpha / (dims ** 0.5)
            embeddings += torch.zeros_like(embeddings).uniform_(-mag_norm, mag_norm)
            return embeddings

        input_embed.forward = MethodType(noisy_forward, input_embed)
        logger.info(f"使用噪声嵌入，alpha={self.model_args.neft_alpha:.2f}")

    def _apply_deepspeed_patches(self, model, config):
        """
        为 MoE 模型应用 DeepSpeed 补丁

        Args:
            model: 要打补丁的模型
            config: 模型配置
        """
        if not is_deepspeed_zero3_enabled():
            return

        model_type = getattr(config, "model_type", None)

        # 为 Mixtral MOE 模型打补丁
        if model_type == "mixtral":
            require_version("deepspeed>=0.13.0", "修复方法: pip install deepspeed>=0.13.0")
            from deepspeed.utils import set_z3_leaf_modules
            from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
            set_z3_leaf_modules(model, [MixtralSparseMoeBlock])

        # 为 DeepSeek-V3 MoE 模块打补丁
        elif model_type == "deepseek_v3":
            require_version("deepspeed>=0.13.0", "修复方法: pip install deepspeed>=0.13.0")
            for layer in model.model.layers:
                if 'DeepseekV3MoE' in str(type(layer.mlp)):
                    layer.mlp._z3_leaf = True


class TokenizerManager:
    """
    分词器初始化和配置管理器
    """

    def __init__(
        self,
        model_args: ModelArguments,
        script_args: ScriptArguments
    ):
        """
        初始化分词器管理器

        Args:
            model_args: 模型配置参数
            script_args: 脚本特定参数
        """
        self.model_args = model_args
        self.script_args = script_args

    def load_tokenizer(self) -> AutoTokenizer:
        """
        加载并配置分词器

        Returns:
            配置好的分词器
        """
        from template import get_conv_template

        tokenizer_kwargs = {
            "cache_dir": self.model_args.cache_dir,
            "use_fast": self.model_args.use_fast_tokenizer,
            "trust_remote_code": self.model_args.trust_remote_code,
        }

        tokenizer_name_or_path = self.model_args.tokenizer_name_or_path
        if not tokenizer_name_or_path:
            tokenizer_name_or_path = self.model_args.model_name_or_path

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, **tokenizer_kwargs)

        # 配置特殊 token
        self._configure_tokens(tokenizer)

        logger.debug(f"分词器: {tokenizer}")
        return tokenizer

    def _configure_tokens(self, tokenizer):
        """
        配置特殊 token（EOS, BOS, PAD）

        Args:
            tokenizer: 要配置的分词器
        """
        from template import get_conv_template
        prompt_template = get_conv_template(self.script_args.template_name)

        # 配置 EOS token
        if tokenizer.eos_token_id is None:
            tokenizer.eos_token = prompt_template.stop_str
            tokenizer.add_special_tokens({"eos_token": tokenizer.eos_token})
            logger.info(
                f"添加 eos_token: {tokenizer.eos_token}, "
                f"eos_token_id: {tokenizer.eos_token_id}"
            )

        # 配置 BOS token
        if tokenizer.bos_token_id is None:
            tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
            tokenizer.bos_token_id = tokenizer.eos_token_id
            logger.info(
                f"添加 bos_token: {tokenizer.bos_token}, "
                f"bos_token_id: {tokenizer.bos_token_id}"
            )

        # 配置 PAD token
        if tokenizer.pad_token_id is None:
            if tokenizer.unk_token_id is not None:
                tokenizer.pad_token = tokenizer.unk_token
            else:
                tokenizer.pad_token = tokenizer.eos_token
            logger.info(
                f"添加 pad_token: {tokenizer.pad_token}, "
                f"pad_token_id: {tokenizer.pad_token_id}"
            )


class PEFTManager:
    """
    参数高效微调（PEFT）管理器
    """

    def __init__(
        self,
        model_args: ModelArguments,
        script_args: ScriptArguments,
        training_args
    ):
        """
        初始化 PEFT 管理器

        Args:
            model_args: 模型配置参数
            script_args: 脚本特定参数
            training_args: 训练参数
        """
        self.model_args = model_args
        self.script_args = script_args
        self.training_args = training_args

    def apply_peft(self, model):
        """
        对模型应用 PEFT（LoRA）

        Args:
            model: 要应用 PEFT 的模型

        Returns:
            PEFT 包装后的模型
        """
        logger.info("微调方法: LoRA(PEFT)")

        # 为 lm_head 设置 fp32 前向钩子
        self._set_lm_head_fp32_hook(model)

        # 加载或创建 LoRA 模型
        if self.script_args.peft_path is not None:
            model = self._load_existing_peft_model(model)
        else:
            model = self._create_new_peft_model(model)

        # 将可训练参数转换为 fp32
        for param in filter(lambda p: p.requires_grad, model.parameters()):
            param.data = param.data.to(torch.float32)

        model.print_trainable_parameters()
        return model

    def _set_lm_head_fp32_hook(self, model):
        """
        为 lm_head 设置 fp32 前向钩子

        Args:
            model: 要设置钩子的模型
        """
        output_layer = getattr(model, "lm_head")
        if not isinstance(output_layer, torch.nn.Linear):
            return

        if output_layer.weight.dtype == torch.float32:
            return

        def fp32_forward_post_hook(
            module: torch.nn.Module,
            args: Tuple[torch.Tensor],
            output: torch.Tensor
        ) -> torch.Tensor:
            return output.to(torch.float32)

        output_layer.register_forward_hook(fp32_forward_post_hook)

    def _load_existing_peft_model(self, model):
        """
        从路径加载现有的 PEFT 模型

        Args:
            model: 基础模型

        Returns:
            加载的 PEFT 模型
        """
        logger.info(f"从预训练模型加载 PEFT: {self.script_args.peft_path}")
        return PeftModel.from_pretrained(
            model,
            self.script_args.peft_path,
            is_trainable=True
        )

    def _create_new_peft_model(self, model):
        """
        使用 LoRA 配置创建新的 PEFT 模型

        Args:
            model: 基础模型

        Returns:
            应用 PEFT 后的模型
        """
        logger.info("初始化新的 PEFT 模型")

        # 如果需要，为 k-bit 训练准备模型
        load_in_8bit = self.model_args.load_in_8bit
        load_in_4bit = self.model_args.load_in_4bit

        if load_in_8bit or load_in_4bit:
            model = prepare_model_for_kbit_training(
                model,
                self.training_args.gradient_checkpointing
            )

        # 获取目标模块
        target_modules = self._get_target_modules(model, load_in_8bit, load_in_4bit)
        logger.info(f"PEFT 目标模块: {target_modules}")

        # 获取要保存的模块
        modules_to_save = self._get_modules_to_save()

        # 创建 LoRA 配置
        logger.info(f"PEFT lora_rank: {self.script_args.lora_rank}")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            inference_mode=False,
            r=self.script_args.lora_rank,
            lora_alpha=self.script_args.lora_alpha,
            lora_dropout=self.script_args.lora_dropout,
            modules_to_save=modules_to_save
        )

        return get_peft_model(model, peft_config)

    def _get_target_modules(self, model, load_in_8bit, load_in_4bit):
        """
        获取 LoRA 的目标模块

        Args:
            model: 模型
            load_in_8bit: 是否使用 8 位量化
            load_in_4bit: 是否使用 4 位量化

        Returns:
            目标模块列表
        """
        target_modules = self.script_args.target_modules

        if not target_modules:
            return None

        target_modules = target_modules.split(',')

        if 'all' in target_modules:
            # 自动查找所有线性层
            return find_all_linear_names(model, int4=load_in_4bit, int8=load_in_8bit)

        return target_modules

    def _get_modules_to_save(self):
        """
        获取除了 LoRA 适配器外要保存的模块

        Returns:
            模块名称列表或 None
        """
        modules_to_save = self.script_args.modules_to_save

        if modules_to_save is None:
            return None

        return modules_to_save.split(',')
