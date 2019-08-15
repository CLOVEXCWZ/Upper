"""
测试
"""

from data_precess import *
from seq2seq_tf import Seq2Seq
from Log import Log
from Config import *
import tensorflow as tf


def test(texts, log):
    char_data = CharData()

    # 数据转化成输入的格式
    inputs_pad, lengs, max_input_len=char_data.pred_text_to_ids(test_texts)

    # 初始化模型
    seq2seq = Seq2Seq(source_vocab_size=char_data.vocab_size,
                      target_vocab_size=char_data.vocab_size,
                      target_start_flag_index=char_data.start_index,
                      target_end_flag_index=char_data.end_index,
                      batch_size=len(test_texts),
                      encode_embed_dim=encode_embed_dim,
                      decode_embed_dim=decode_embed_dim,
                      rnn_size=rnn_size,
                      num_layers=num_layers,
                      learning_rate=learning_rate,
                      max_pred_len=max_input_len,
                      train_mode=False
                      )

    # 预测处理
    with tf.Session() as sess:
        checkpoint = tf.train.latest_checkpoint(model_save_path)
        if checkpoint:
            seq2seq.load(sess, checkpoint)
            log.info("test laod model"+checkpoint)
            log.info(str(seq2seq.params_set()))
        else:
            log.info("请先训练模型")
            return

        pred_ids = seq2seq.predict(sess, inputs_pad, lengs)
        pred_texts=char_data.index_to_text(pred_ids)
        print(test_texts)
        print(pred_texts)
        log.info("测试数据:"+str(test_texts))
        log.info("预测数据:"+str(pred_texts))


if __name__ == '__main__':

    #  测试texts
    test_texts = ["askcd", "scdokosjdvoih"]

    log = Log(train_log_name)
    test(texts=test_texts, log=log)


