# Description: Custom Trainer class for multi eval model training
# Author
import json
import numpy as np
import torch

from datasets import load_dataset, concatenate_datasets, Dataset
from evaluate import load
from pathlib import Path
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    GenerationConfig,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from transformers.utils import logging
from typer import echo, Option, Typer
from typing import Dict, List, Optional, Union


logger = logging.get_logger(__name__)
app = Typer()


def parse_kv(kv: str) -> dict:
    """
    Parses a string of key-value pairs separated by commas and ensures proper format.
    Raises ValueError if the format is incorrect.
    """
    kv_dict = {}
    pairs = kv.split(",")
    for item in pairs:
        if "=" not in item:
            raise ValueError("Each key-value pair must contain an '='.")
        key, value = item.split("=", 1)  # Split only on the first '=', allowing for '=' in values
        if not key.strip() or not value.strip():
            raise ValueError("Both key and value must be non-empty and not just whitespace.")
        kv_dict[key.strip()] = value.strip()
    return kv_dict


def get_prf(gold_tags: list, predicted_tags: list):
    # convert to sets
    gold_tags_set = set(gold_tags)
    predicted_tags_set = set(predicted_tags)
    # calculate true positives
    true_positives = len(gold_tags_set.intersection(predicted_tags_set))
    # calculate precision
    precision = true_positives / len(predicted_tags_set) if len(predicted_tags_set) > 0 else 0
    # calculate recall
    recall = true_positives / len(gold_tags_set) if len(gold_tags_set) > 0 else 0
    # calculate f1
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return {"precision": precision, "recall": recall, "f1": f1}


def pre_process_eos(dataset, eos_token):
    prompts = [doc for doc in dataset["prompt"]]
    responses = [(doc + " " + eos_token).strip() for doc in dataset["response"]]
    new_dataset = Dataset.from_dict({"prompt": prompts, "response": responses})
    return new_dataset


class MultiEvalTrainer(Seq2SeqTrainer):
    def __init__(
        self,
        eval_datasets: Optional[Dict[str, Union[Dataset, Dict[str, Dataset]]]] = None,
        dataset2_is_rouge: Optional[Dict[str, bool]] = None,
        **kwargs,
    ):
        self.eval_datasets = eval_datasets
        self.rouge = load("rouge")
        self.eval_is_rouge = dataset2_is_rouge

        super().__init__(
            **kwargs,
        )

    def evaluate(
        self,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        **gen_kwargs,
    ) -> Union[Dict[str, float], Dict]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        """
        eval_scores = {}
        for dataset_name, eval_dataset in self.eval_datasets.items():
            if self.eval_is_rouge and self.eval_is_rouge[dataset_name]:
                self.compute_metrics = self.compute_rouge
            else:
                self.compute_metrics = self.compute_prf

            print(f"Evaluating on {dataset_name}")
            eval_scores.update(
                super(MultiEvalTrainer, self).evaluate(
                    eval_dataset=eval_dataset,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=f"eval_{dataset_name}",
                )
            )

        eval_loss_keys = [key for key in eval_scores if key.endswith("loss")] 
        eval_loss = sum([eval_scores[key] for key in eval_loss_keys])
        eval_scores[f"{metric_key_prefix}_loss"] = eval_loss

        return eval_scores

    def compute_prf(self, eval_pred):
        predictions, labs = eval_pred
        decoded_preds = self.decode_predictions(predictions)
        decoded_labels = self.decode_predictions(labs)

        decoded_preds = [
            (i, p.strip())
            for i, pred in enumerate(decoded_preds)
            for p in pred.split("|")
            if p.strip() != ""
        ]
        decoded_labels = [
            (i, p.strip())
            for i, pred in enumerate(decoded_labels)
            for p in pred.split("|")
            if p.strip() != ""
        ]

        return get_prf(decoded_labels, decoded_preds)

    def compute_rouge(self, eval_pred):
        predictions, labs = eval_pred
        decoded_preds = self.decode_predictions(predictions)
        decoded_labels = self.decode_predictions(labs)

        return self.rouge.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )

    def decode_predictions(self, predictions):
        predictions = np.where(
            predictions != -100, predictions, self.tokenizer.pad_token_id
        )
        decoded_preds = self.tokenizer.batch_decode(
            predictions, skip_special_tokens=True
        )
        return decoded_preds

    def training_step(self, model, inputs) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        try:
            return super().training_step(model, inputs)
        except torch.cuda.OutOfMemoryError:
            print("Out of memory error")
            return torch.tensor(0.0).to(self.args.device)


def update_config_dict(config, kwargs):
    # Update the config dictionary with the kwargs recursively
    for v in config.values():
        if isinstance(v, dict):
            update_config_dict(v, kwargs)
        else:
            for k, val in kwargs.items():
                if k in config:
                    config[k] = type(config[k])(val)


def trainer_seq2seq_multi(
    config_file: Path,
    datasets_dict: Dict[str, Dict[str, Dataset]],
    debug: bool = False,
    is_peft: bool = False,
    check_pt: str = None,
    **kwargs,
):
    """

    :param config_file:
    :param datasets_dict: Dictionary of Dictionaries of datasets. Outer Dict = task, Inner Dict = split
    :param debug: If True, only train on a small subset of the data
    :param kwargs: additional arguments to update config_file dictionary
    :return:
    """
    config = json.load(open(config_file))

    update_config_dict(config, kwargs)

    # print(dataset_names)
    # datasets_dict = {d_name: load_dataset(d_name) for d_name in dataset_names}

    model_name_or_path = config.pop("model_name_or_path")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    # Concatenate the train sets of each dataset

    train_dataset = concatenate_datasets(
        [
            pre_process_eos(dataset["train"], tokenizer.eos_token)
            for dataset in datasets_dict.values()
        ]
    )
   

    eval_datasets = {
        dataset_name: (
            pre_process_eos(dataset["dev"], tokenizer.eos_token)
            if "dev" in dataset.keys()
            else pre_process_eos(dataset["validation"], tokenizer.eos_token)
        )
        for dataset_name, dataset in datasets_dict.items()
    }

    if debug:
        train_dataset = train_dataset.sort("prompt")
        train_dataset = Dataset.from_dict(train_dataset[:500])

        eval_datasets = {k: Dataset.from_dict(v[:100]) for k, v in eval_datasets.items()}
        config["trainer"]["logging_steps"] = 10
        config["trainer"]["warmup_steps"] = 5
        config["trainer"]["eval_steps"] = 50


    def preprocess_data(examples):
        model_inputs = tokenizer(
            examples["prompt"],
            max_length=config["max_input_length"],
            truncation=True,
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["response"],
                truncation=True,
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_tokenized = train_dataset.map(preprocess_data, batched=True)
    evals_tokenized = {
        dataset_name: eval_dataset.map(preprocess_data, batched=True)
        for dataset_name, eval_dataset in eval_datasets.items()
    }

    if is_peft:
        from peft import prepare_model_for_kbit_training
        from peft import LoraConfig, get_peft_model, TaskType

        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path, torch_dtype=torch.bfloat16
        )
        model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=64,  # Increased rank
            lora_alpha=16,  # Scaling factor
            target_modules=["q", "v", "k", "o", "wi", "wo"],  # Expanded target modules
            lora_dropout=0.1,
            # bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path, device_map="balanced", torch_dtype=torch.bfloat16
        )

    training_args = Seq2SeqTrainingArguments(**config["trainer"])

    generation_config = GenerationConfig.from_pretrained(model_name_or_path)
    generation_config.eos_token_id = tokenizer.eos_token_id
    generation_config.update(**config["generation"])

    training_args.generation_config = generation_config
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    t5_trainer = MultiEvalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_datasets=evals_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=None,
        # compute_metrics=compute_metrics, # This now gets set depending on which type of dataset is being evaluated
    )
    if check_pt:
        t5_trainer.train(check_pt)
    else:
        t5_trainer.train()
    t5_trainer.save_model()


@app.command()
def train(
    dataset_names: List[str],
    config_file: str = "config.json",
    is_peft: bool = False,
    debug: bool = False,
    check_pt: str = None,
    kv: str = Option(
        None,
        "--kv",
        help="Key-value pairs, separated by commas, e.g., key1=value1,key2=value2",
    ),
):
    """
    Train datasets of the form:
        {
            "prompt": "SRL for [predicate]: sentence with [predicate]",
            "response": ARG-0: [arg0] | ARG-1: [arg1] | ... | ARG-N: [argn]
        }
    :param config_file:
    :param dataset_names:
    :param is_peft: If True, use PEFT for training
    :param debug: If True, only train on a small subset of the data
    :param kv: override config file parameters with this. e.g., "num_train_epochs=20,per_device_train_batch_size=8"
    :return:
    """
    dataset_names = list(set(dataset_names))
    dataset_dict = {}

    for ds_name in dataset_names:
        dataset_dict[ds_name] = load_dataset(ds_name)

    kv_dict = {}

    if kv:
        try:
            kv_dict = parse_kv(kv)
            echo("Received key-value arguments:")
            for key, value in kv_dict.items():
                echo(f"{key}: {value}")
        except ValueError as e:
            echo(f"Error: {e}")
    else:
        echo("No key-value arguments provided.")

    trainer_seq2seq_multi(config_file, dataset_dict, debug, is_peft, check_pt, **kv_dict)


if __name__ == "__main__":
    app()
