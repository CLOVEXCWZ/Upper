from data_precess import *
from seq2seq_tf import Seq2Seq
from Log import Log
from Config import *
import tensorflow as tf


def train(log):
    char_data=CharData()

    seq2seq = Seq2Seq(source_vocab_size=char_data.vocab_size,
                      target_vocab_size=char_data.vocab_size,
                      target_start_flag_index=char_data.start_index,
                      target_end_flag_index=char_data.end_index,
                      batch_size=batch_size,
                      encode_embed_dim=encode_embed_dim,
                      decode_embed_dim=decode_embed_dim,
                      rnn_size=rnn_size,
                      num_layers=num_layers,
                      learning_rate=learning_rate,
                      train_mode=True
                      )

    log.info("create seq2seq model")
    log.info(str(seq2seq.params_set()))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        checkpoint = tf.train.latest_checkpoint(model_save_path)
        if checkpoint:
            seq2seq.load(sess, checkpoint)
            log.info("load model form:" + checkpoint)
        for index in range(epochs):
            for batch_i, (targets_batch, sources_batch, targets_lengths, sources_lengths) in enumerate(
                    char_data.get_batches(batch_size)):
                loss = seq2seq.train(sess, sources_batch, sources_lengths, targets_batch, targets_lengths)
            log.info("epochs:{}/{} loss:{} ".format(index, epochs, loss))
            seq2seq.save(sess, model_save_path+model_name)
            log.info("save model:"+model_save_path+model_name)



if __name__ == '__main__':

    check_dir()

    log=Log(train_log_name)
    train(log)


