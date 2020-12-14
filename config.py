class config():
    fold_num=10
    fold_id=fold_num-1
    dropout = 0.15
    # build word encoder
    word_hidden_size = 128
    word_num_layers = 2
    # build sent encoder
    sent_hidden_size = 256
    sent_num_layers = 2
    # learn
    learning_rate = 2e-4
    decay = .75
    decay_step = 1000
    epochs=1
    clip = 5.0
    epochs = 1
    early_stops = 3
    log_interval = 200
    test_batch_size = 16
    train_batch_size = 16