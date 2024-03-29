
batch_size=32
encode_embed_dim=32
decode_embed_dim=32
rnn_size=128
num_layers=1
learning_rate=0.001
epochs=20


model_save_path='models/'
model_name='tf_seq2seq_upper.ckpt'
train_log_name='log/train.log'


def check_dir():
    import os
    if not os.path.exists("models"):
        os.makedirs("models")
    if not os.path.exists("Log"):
        os.makedirs("Log")
    if not os.path.exists("data"):
        os.makedirs("data")
