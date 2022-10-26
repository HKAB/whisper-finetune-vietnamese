
# Whiper vietnamese finetuning

This notebook contains:
- Notebooks finetuning, inferencing and generating N-gram.
- Demo Whisper and Wav2vec




## Installation

For using Beam search with LM, install Whisper from my Github
```bash
    pip install git+https://github.com/HKAB/whisper.git
```

## Run

For training & inference

  ```shell
    python finetune.py  --model_name base \
                        --dataset vin100h

    python test.py      --checkpoint_path path/to/ckpt \
                        --dataset vin100h \
                        --model_name base

  ```

For generating language model with KenLM, use notebook in notebooks folder.

We share the checkpoint (*base, batch_size 1, gradient accumulation steps 10, epoch 14*)\.
## Demo


![Whisper](images/whisper.png "Whisper")

![Wav2vec](images/wav2vec.png "Wav2vec")
## Contributing

- We modified the ASTGCN model with full three information from last hour, yesterday and previous week.
- We compare ASTGCN with GCN, LSTM, HA in various settings.
## Result

| Methods                   | Fleurs  | Vin100h (Full) | 
|---------------------------|---------|--------------- |
| Whisper (base)            | 50.38%  | 50.33%         |
| Finetune Whisper (base)   | 28.68%  | 33%            |
| Whisper (large) one shot  | -       | 26.87%         |
