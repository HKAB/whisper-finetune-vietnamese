try:
    import tensorflow  # required in Colab to avoid protobuf compatibility issues
except ImportError:
    pass


import jiwer
import whisper
import torch
import argparse
from tqdm import tqdm
import pandas as pd

from config import Config
from dataset import load_dataset, WhisperDataCollatorWithPadding
from model import WhisperModelModule



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default='', help='path of checkpoint, if not set, use origin pretrained model')
    parser.add_argument('--dataset_name', type=str, default='fluers', help='the dataset for finetuning, includes fluers, vin100h, vlsp2019')
    parser.add_argument('--model_name', type=str, default='tiny', help='model name')

    args = parser.parse_args()
    config = Config()
    config.checkpoint_path = args.checkpoint_path
    config.model_name = args.model_name

    module = WhisperModelModule(config)
    try:
        state_dict = torch.load(config.checkpoint_path)
        state_dict = state_dict["state_dict"]
        module.load_state_dict(state_dict)
        print(f"load checkpoint successfully from {config.checkpoint_path}")
    except Exception as e:
        print(e)
        print(f"load checkpoint failt using origin weigth of {config.model_name} model")
    model = module.model
    model.to(device)

    _, valid_dataset = load_dataset(args.dataset_name, test=True)
    test_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_worker,
        collate_fn=WhisperDataCollatorWithPadding(),
    )

    # decode the audio
    options = whisper.DecodingOptions(
        language="vi", without_timestamps=True, fp16=torch.cuda.is_available()
    )

    hypotheses = []
    references = []
    print(model.device)
    for sample in tqdm(test_loader):
        mels = sample["input_ids"].to(model.device)
        texts = sample["texts"]
        results = model.decode(mels, options)
        hypotheses.extend([result.text for result in results])
        references.extend(texts)

    data = pd.DataFrame(dict(hypothesis=hypotheses, reference=references))
    
    data["hypothesis_clean"] = [
        text.lower() for text in data["hypothesis"]
    ]
    data["reference_clean"] = [
        text.lower() for text in data["reference"]
    ]

    data.to_csv('results.csv')
    for i in range(60):
        print('Reference:', data["reference_clean"][i])
        print('Predict:', data["hypothesis_clean"][i])
        print('\n')
    wer = jiwer.wer(list(data["reference_clean"]), list(data["hypothesis_clean"]))

    print(f"WER: {wer * 100:.2f} %")