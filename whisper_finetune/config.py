class Config:
    learning_rate = 0.0001
    weight_decay = 0.01
    adam_epsilon = 1e-8
    warmup_steps = 2
    batch_size = 16
    num_worker = 2
    num_train_epochs = 10
    gradient_accumulation_steps = 1
    sample_rate = 16000
    log_output_dir = "logs"
    check_output_dir = "artifacts"
    train_name = "whisper"
    train_id = "fluers"
    model_name = "base"
    lang = "vi"
    checkpoint_path = ""  # using origin model if this parh is invaild