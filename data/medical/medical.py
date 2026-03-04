# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:

Natural Language Generation Chinese Corpus.(medical)
"""

import os
import json
import datasets
_DESCRIPTION = """纯文本数据，中文医疗数据集，包含预训练数据的百科数据，指令微调数据和奖励模型数据。"""
_HOMEPAGE = "https://github.com/shibing624/MedicalGPT"
_CITATION = ""
_LICENSE = ""
_BASE_URL = "https://huggingface.co/datasets/shibing624/medical/resolve/main/"
# file url: https://huggingface.co/datasets/shibing624/medical/resolve/main/finetune/test_zh_0.json

class NewDataset(datasets.GeneratorBasedBuilder):
    """Medical Chinese Version"""

    VERSION = datasets.Version("1.0.1")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="pretrain", version=VERSION, description="pretrain data"),
        datasets.BuilderConfig(name="finetune", version=VERSION, description="finetune data"),
        datasets.BuilderConfig(name="reward", version=VERSION, description="reward data"),
    ]

    def _info(self):
        if self.config.name == "pretrain":
            features = datasets.Features(
                {
                    "text": datasets.Value("string")
                }
            )
        elif self.config.name == 'finetune': 
            features = datasets.Features(
                {
                    "instruction": datasets.Value("string"),
                    "input": datasets.Value("string"),
                    "output": datasets.Value("string")
                }
            )
        elif self.config.name == 'reward': 
            features = datasets.Features(
                {
                    "question": datasets.Value("string"),
                    "response_chosen": datasets.Value("string"),
                    "response_rejected": datasets.Value("string")
                }
            )
        
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # supervised_keys=("sentence", "label"),
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        data_url = _BASE_URL + self.config.name

        if self.config.name == 'pretrain':
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "filepath": dl_manager.download_and_extract(f"{data_url}/train_encyclopedia.json"),
                        "split": "train"
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "filepath": dl_manager.download_and_extract(f"{data_url}/valid_encyclopedia.json"),
                        "split": "dev"
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "filepath": dl_manager.download_and_extract(f"{data_url}/test_encyclopedia.json"),
                        "split": "test"
                    },
                ),
            ]
        elif self.config.name == 'finetune':
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "filepath": dl_manager.download_and_extract([f"{data_url}/train_zh_0.json", f"{data_url}/train_en_1.json"]),
                        "split": "train"
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "filepath": dl_manager.download_and_extract([f"{data_url}/valid_zh_0.json", f"{data_url}/valid_en_1.json"]),
                        "split": "dev"
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "filepath": dl_manager.download_and_extract([f"{data_url}/test_zh_0.json", f"{data_url}/test_en_1.json"]),
                        "split": "test"
                    },
                ),
            ]
        elif self.config.name == 'reward':
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "filepath": dl_manager.download_and_extract(f"{data_url}/train.json"),
                        "split": "train"
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "filepath": dl_manager.download_and_extract(f"{data_url}/valid.json"),
                        "split": "dev"
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "filepath": dl_manager.download_and_extract(f"{data_url}/test.json"),
                        "split": "test"
                    },
                ),
            ]
        
    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):
        id = 0
        if isinstance(filepath, str):
            filepath = [filepath]
        for file in filepath:
            with open(file, encoding="utf-8") as f:
                for key, row in enumerate(f):
                    data = json.loads(row)
                    if self.config.name == "pretrain":
                        yield id, {
                            "text": data["text"]
                        }
                    elif self.config.name == 'finetune':
                        yield id, {
                            "instruction": data["instruction"],
                            "input": data["input"],
                            "output": data["output"]
                        }
                    elif self.config.name == 'reward':
                        yield id, {
                            "question": data["question"],
                            "response_chosen": data["response_chosen"],
                            "response_rejected": data["response_rejected"]
                        }
                    id += 1


