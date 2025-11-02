from datetime import datetime
from typing import Optional
import datasets
import evaluate
import lightning as L
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
import argparse
import os
from lightning.pytorch.loggers import TensorBoardLogger  
from transformers import get_cosine_schedule_with_warmup
from transformers import get_constant_schedule_with_warmup

DEFAULT_CONFIG = {
    'learning_rate': 2e-5,
    'warmup_steps': 0,
    'weight_decay': 0.0,
    'train_batch_size': 32,
    'eval_batch_size': 32,
    'max_seq_length': 128,
    'optimizer_type': 'adamw',
    'adam_beta1': 0.9,
    'adam_beta2': 0.999,
    'lr_scheduler': 'linear',
    'sgd_momentum': 0.9,
    'epochs': 3,
    'seed': 42,
    'fresh_start' : False
}


class GLUEDataModule(L.LightningDataModule):
    """loads the task dataset and creates dataloaders for train and validation sets"""
    task_text_field_map = {
        "cola": ["sentence"],
        "sst2": ["sentence"],
        "mrpc": ["sentence1", "sentence2"],
        "qqp": ["question1", "question2"],
        "stsb": ["sentence1", "sentence2"],
        "mnli": ["premise", "hypothesis"],
        "qnli": ["question", "sentence"],
        "rte": ["sentence1", "sentence2"],
        "wnli": ["sentence1", "sentence2"],
        "ax": ["premise", "hypothesis"],
    }

    glue_task_num_labels = {
        "cola": 2,
        "sst2": 2,
        "mrpc": 2,
        "qqp": 2,
        "stsb": 1,
        "mnli": 3,
        "qnli": 2,
        "rte": 2,
        "wnli": 2,
        "ax": 3,
    }

    loader_columns = [
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "labels",
    ]

    def __init__(
        self,
        model_name_or_path: str,
        task_name: str = "mrpc",
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.text_fields = self.task_text_field_map[self.hparams.task_name]
        self.num_labels = self.glue_task_num_labels[self.hparams.task_name]
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_name_or_path, use_fast=True)

    def setup(self, stage: str):
        self.dataset = datasets.load_dataset("glue", self.hparams.task_name)

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                remove_columns=["label"],
            )
            self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.dataset[split].set_format(type="torch", columns=self.columns)

        self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]

    def prepare_data(self):
        datasets.load_dataset("glue", self.hparams.task_name)
        AutoTokenizer.from_pretrained(self.hparams.model_name_or_path, use_fast=True)

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=self.hparams.train_batch_size, shuffle=True)

    def val_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["validation"], batch_size=self.hparams.eval_batch_size)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.hparams.eval_batch_size) for x in self.eval_splits]

    def test_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["test"], batch_size=self.hparams.eval_batch_size)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.hparams.eval_batch_size) for x in self.eval_splits]

    def convert_to_features(self, example_batch, indices=None):
        # Either encode single sentence or sentence pairs
        if len(self.text_fields) > 1:
            texts_or_text_pairs = list(zip(example_batch[self.text_fields[0]], example_batch[self.text_fields[1]]))
        else:
            texts_or_text_pairs = example_batch[self.text_fields[0]]

        # Tokenize the text/text pairs
        features = self.tokenizer.batch_encode_plus(
            texts_or_text_pairs, max_length=self.hparams.max_seq_length, padding="max_length", truncation=True
        )

        # Rename label to labels to make it easier to pass to model forward
        features["labels"] = example_batch["label"]

        return features
    
class GLUETransformer(L.LightningModule):
    """ implements the model forward pass and training/validation steps
        this is where the LOGGING happenes """
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        task_name: str,
        # Hyperparameters
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)
        self.metric = evaluate.load(
        "glue", self.hparams.task_name, experiment_id=datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        )
        self.validation_step_outputs = []

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]

        if self.hparams.num_labels > 1:
            preds = torch.argmax(logits, axis=1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()

        labels = batch["labels"]
        self.validation_step_outputs.append({"loss": val_loss, "preds": preds, "labels": labels})
        return val_loss

    def on_validation_epoch_end(self):
        if self.hparams.task_name == "mnli":
            for i, output in enumerate(self.validation_step_outputs):
                # matched or mismatched
                split = self.hparams.eval_splits[i].split("_")[-1]
                preds = torch.cat([x["preds"] for x in output]).detach().cpu().numpy()
                labels = torch.cat([x["labels"] for x in output]).detach().cpu().numpy()
                loss = torch.stack([x["loss"] for x in output]).mean()
                self.log(f"val_loss_{split}", loss, prog_bar=True)
                split_metrics = {
                    f"{k}_{split}": v for k, v in self.metric.compute(predictions=preds, references=labels).items()
                }
                self.log_dict(split_metrics, prog_bar=True)
            self.validation_step_outputs.clear()
            return loss

        preds = torch.cat([x["preds"] for x in self.validation_step_outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in self.validation_step_outputs]).detach().cpu().numpy()
        loss = torch.stack([x["loss"] for x in self.validation_step_outputs]).mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log_dict(self.metric.compute(predictions=preds, references=labels), prog_bar=True)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        # optimizer based on type
        if self.hparams.optimizer_type.lower() == 'adam':
            optimizer = torch.optim.Adam(
                optimizer_grouped_parameters,
                lr=self.hparams.learning_rate,
                betas=(self.hparams.adam_beta1, self.hparams.adam_beta2)
            )
        elif self.hparams.optimizer_type.lower() == 'sgd':
            optimizer = torch.optim.SGD(
                optimizer_grouped_parameters,
                lr=self.hparams.learning_rate,
                momentum=self.hparams.sgd_momentum
            )
        elif self.hparams.optimizer_type.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.hparams.learning_rate,
                betas=(self.hparams.adam_beta1, self.hparams.adam_beta2)
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {self.hparams.optimizer_type}")

        # scheduler based on type
        if self.hparams.lr_scheduler == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.warmup_steps,
                num_training_steps=self.trainer.estimated_stepping_batches,
            )
        elif self.hparams.lr_scheduler == 'constant':
            scheduler = get_constant_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.warmup_steps,
            )
        elif self.hparams.lr_scheduler.lower() == 'linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.warmup_steps,
                num_training_steps=self.trainer.estimated_stepping_batches,
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {self.hparams.lr_scheduler}")
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency":1}
        return [optimizer], [scheduler]
 

def get_accelerator():
    """ enable use of the best available backend """
    if torch.cuda.is_available():
        return "gpu", 1, "Using GPU acceleration"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps", 1, "Using Metal Performance Shaders"
    else:
        return "cpu", 1, "Using CPU"
    
def get_next_run_number():
    if not os.path.exists('./checkpoints'):
        return 1
    runs = [d for d in os.listdir('./checkpoints') if d.startswith('run_')]
    if not runs:
        return 1
    return max([int(r.split('_')[1]) for r in runs]) + 1

def get_latest_checkpoint():
    if not os.path.exists('./checkpoints'):
        return None
    all_ckpts = glob.glob('./checkpoints/run_*.ckpt')
    if not all_ckpts:
        return None
    return max(all_ckpts, key=os.path.getmtime)

def run(**kwargs):
    """ run a training experiment with finetuned hyperparameters. If non are given, the run will
    use the best result values found during manual tuning
    """
    params = DEFAULT_CONFIG.copy()
    params.update(kwargs)
    L.seed_everything(params['seed'])

    accelerator, devices, accel_msg = get_accelerator()
    print(accel_msg)

    run_number = get_next_run_number()

    # use hyperparamters for naming convention
    hp_str = "_".join([f"{k}_{v}" for k, v in params.items()])
    run_name = f"run_{run_number:03d}_{hp_str}"
    run_dir = f"./checkpoints/{run_name}"

    os.makedirs(run_dir, exist_ok=True)
    logger = TensorBoardLogger(save_dir="./logs", name="hp_tuning", version=run_name)

    # Setup data module
    dm = GLUEDataModule(
        model_name_or_path="distilbert-base-uncased",
        task_name="mrpc",
        **params
    )
    dm.setup("fit")


    last_checkpoint = f'{run_dir}/last.ckpt'
    if not params['fresh_start'] and os.path.exists(last_checkpoint):
        model = GLUETransformer.load_from_checkpoint(
            last_checkpoint,
            model_name_or_path="distilbert-base-uncased",
            num_labels=dm.num_labels,
            eval_splits=dm.eval_splits,
            task_name=dm.hparams.task_name,
            **params
        )
        print(f"resuming from {last_checkpoint}")
    else:
        model = GLUETransformer(
        model_name_or_path="distilbert-base-uncased",
        num_labels=dm.num_labels,
        eval_splits=dm.eval_splits,
        task_name=dm.hparams.task_name,
        **params
        )
        print("starting with fresh model")
    checkpoint_callback = L.pytorch.callbacks. ModelCheckpoint(
        dirpath = run_dir,
        filename='last',
        every_n_epochs=params['epochs']
    )

    # Setup trainer
    trainer = L.Trainer(
        max_epochs=params['epochs'], # fuction param
        accelerator=accelerator, # ensure proper backend is used
        devices=devices, # ensure proper backend is used
        logger=logger,
        callbacks=[checkpoint_callback],
        enable_checkpointing=True,
        enable_progress_bar=True, 
        enable_model_summary=False,  # Suppress model summary to reduce clutter
    )

    # Train
    trainer.fit(model, datamodule=dm)

    results = {
        "run_number" : run_number,
        "accuracy" : trainer.callback_metrics.get("accuracy", 0.0).item(),
        "val_loss" : trainer.callback_metrics.get("val_loss", float('inf')).item(),
        "run_name" : run_name,
    }

    return results



if __name__ == "__main__":
    parser = argparse.ArgumentParser(usage='%(prog)s [options]')
    for param, default in DEFAULT_CONFIG.items():
        parser.add_argument(
            f'--{param}',
            type=type(default),
            default=None,
            help=str(default),
        )

    args = parser.parse_args()

    kwargs = {k: v for k, v in vars(args).items() if v is not None}
    results = run(**kwargs)
    print(f"Results: {results}")



