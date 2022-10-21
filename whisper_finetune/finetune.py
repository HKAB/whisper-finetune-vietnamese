import os
import argparse
from pathlib import Path
import torch
from config import Config

try:
    import tensorflow  # required in Colab to avoid protobuf compatibility issues
except ImportError:
    pass

from dataset import load_dataset, WhisperDataCollatorWithPadding

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from model import WhisperModelModule

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint_path', type=str, default='', help='path of checkpoint, if not set, use origin pretrained model')
    parser.add_argument('--model_name', type=str, default='base', help='model name, tiny, small, medium, base, large')
    parser.add_argument('--dataset', type=str, default='fluers', help='the dataset for finetuning, includes fluers, vin100h, vlsp2019')
    parser.add_argument('--lang', type=str, default='vi', help='language, vi, en')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--epoch', type=int, default=10, help='number of epoch for finetuning')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')

    args = parser.parse_args()

    # Load default config from config.py and set new config from args
    config = Config()
    config.model_name = args.model_name
    config.lang = args.lang
    config.learning_rate = args.lr
    config.num_train_epochs = args.epoch
    config.batch_size = args.batch_size
    config.checkpoint_path = args.checkpoint_path

    print(f"""Finetuning Whisper with new config:
            checkpoint_path: %s,
            dataset: %s,
            model_name: %s,
            lang: %s,
            learning_rate: %.5f,
            num_finetune_epochs: %d,
            batch_size: %d""" % ("No check point" if config.checkpoint_path == "" else config.checkpoint_path, args.dataset, config.model_name, config.lang, config.learning_rate, config.num_train_epochs, config.batch_size))

    # Load dataset for finetuning
    if config.lang == "vi":
        train_dataset, valid_dataset = load_dataset(args.dataset)
    else:
        raise ValueError("Not support other language dataset, please choose vi for languague!")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_worker,
        collate_fn=WhisperDataCollatorWithPadding(),
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_worker,
        collate_fn=WhisperDataCollatorWithPadding(),
    )


    Path(os.path.join(os.getcwd(), config.log_output_dir)).mkdir(exist_ok=True)
    Path(os.path.join(os.getcwd(), config.check_output_dir)).mkdir(exist_ok=True)

    # Log and checkpoint
    tflogger = TensorBoardLogger(
        save_dir=config.log_output_dir, name=config.train_name, version=config.train_id
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{config.check_output_dir}/checkpoint",
        filename="checkpoint-{epoch:04d}",
        save_top_k=1,  # -1: all model save, 1: best model save
    )

    # callback list
    callback_list = [checkpoint_callback, LearningRateMonitor(logging_interval="epoch")]
    model = WhisperModelModule(config, train_loader, valid_loader)

    # Trainer
    trainer = Trainer(
        precision=16,
        accelerator=DEVICE,
        max_epochs=config.num_train_epochs,
        accumulate_grad_batches=config.gradient_accumulation_steps,
        logger=tflogger,
        callbacks=callback_list,
    )

    trainer.fit(model)