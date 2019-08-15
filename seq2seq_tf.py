"""
时间： 2019-08-15

seq2seq model

"""
import tensorflow as tf
from tensorflow.python.layers.core import Dense

class Seq2Seq(object):

    def __init__(self,
                 source_vocab_size,  # Int 源数据 covab大小
                 target_vocab_size,  # Int 目标数据 vocab大小
                 target_start_flag_index=0,  # Int 目标数据开始标记
                 target_end_flag_index=1,  # Int 目标数据介绍标记
                 batch_size=32,  # Int batch大小
                 encode_embed_dim=128,  # Int encode_dim 大小
                 decode_embed_dim=128,  # Int decoder_dim 大小
                 max_pred_len=128,  # Int 预测时最大长度(预测时需要)
                 rnn_size=128,  # Int 一层rnn的神经元格式
                 num_layers=2,  # Int 层数
                 learning_rate=0.001,  # float  学习率
                 train_mode=True,  # bool 是否为训练模式
                 ):

        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.target_start_flag_index = target_start_flag_index
        self.target_end_flag_index = target_end_flag_index
        self.batch_size = batch_size
        self.encode_embed_dim = encode_embed_dim
        self.decode_embed_dim = decode_embed_dim
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.train_mode = train_mode
        self.max_pred_len = max_pred_len

        self.build_model()  # 创建模型

    def get_inputs(self):
        """ 创建 placeholder """
        self.inputs = tf.placeholder(tf.int32, (None, None), name='inputs')  # 输入原句 (None, None)
        self.source_sequence_length = tf.placeholder(tf.int32, (None,), name='source_sequence_length')  # 原数据长度-(None,)
        if self.train_mode:
            self.targets = tf.placeholder(tf.int32, (None, None), name='targets')  # 目标句子 (None, None)
            self.target_sequence_length = tf.placeholder(tf.int32, (None,), name='target_sequence_length')  # 目标数据长度 (None, )
            self.max_target_sequence_length = tf.reduce_max(self.target_sequence_length, name='max_target_len')  # 最大目标长度

    def get_encoder_layer(self,
                          input_data,  # 输入tensor （None, None）
                          source_sequence_length):  # 源数据的序列长度
        """
        构建encoder层
        :param input_data: (None, None)
        :param source_sequence_length: (None,)
        :return: encoder_output  encoder_state
        """

        # (?, ?, 128) (batch_size, None, dim)
        encoder_embed_input = tf.contrib.layers.embed_sequence(ids=input_data,
                                                               vocab_size=self.source_vocab_size,
                                                               embed_dim=self.encode_embed_dim)
        def get_lstm_cell(rnn_size):
            return tf.contrib.rnn.LSTMCell(num_units=rnn_size,
                                           initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        cell = tf.contrib.rnn.MultiRNNCell([get_lstm_cell(self.rnn_size) for _ in range(self.num_layers)])
        encoder_output, encoder_state = tf.nn.dynamic_rnn(cell=cell,
                                                          inputs=encoder_embed_input,
                                                          sequence_length=source_sequence_length,
                                                          dtype=tf.float32)

        # encoder_output (?, ?, 128) (batch_size, None, rnn_size)
        # encoder_state Tuple((None, 128), (None, 128))
        return encoder_output, encoder_state

    def process_decoder_input(self, data):
        """
        把最后一个字符移除，前面添加一个 start_flag_index
        例如：  A B C D <EOS>       (<EOS> 为结束标识符)
        --> <GO> A B C D           （<GO> 为开始标识符）
        """
        ''' 补充start_flag，并移除最后一个字符 '''
        ending = tf.strided_slice(data, [0, 0], [self.batch_size, -1], [1, 1])  # cut掉最后一个字符
        decoder_input = tf.concat([tf.fill([self.batch_size, 1], self.target_start_flag_index), ending], 1)
        return decoder_input

    def decoding_layer(self,
                       source_sequence_length,  # 源数据长度
                       encoder_state,    # encode 的状态
                       decoder_input=None,  # decoder端输入
                       target_sequence_length=None,  # target数据序列长度
                       max_target_sequence_length=None, ):  # target数据序列最大长度
        # embedding
        decoder_embeddings = tf.Variable(tf.random_uniform([self.target_vocab_size, self.decode_embed_dim]))

        def get_decoder_cell(rnn_size):
            decoder_cell = tf.contrib.rnn.LSTMCell(num_units=rnn_size,
                                                   initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            return decoder_cell

        cell = tf.contrib.rnn.MultiRNNCell([get_decoder_cell(self.rnn_size) for _ in range(self.num_layers)])
        #  Output全连接层
        output_layer = Dense(units=self.target_vocab_size,
                             kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
        # Training decoder
        with tf.variable_scope("decode"):
            if self.train_mode:
                #  Embedding
                decoder_embed_input = tf.nn.embedding_lookup(params=decoder_embeddings,
                                                             ids=decoder_input)
                # 得到help对象
                training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embed_input,
                                                                    sequence_length=target_sequence_length,
                                                                    time_major=False)
                # 构造decoder decoder_initial_state encoder_state
                training_decoder = tf.contrib.seq2seq.BasicDecoder(cell=cell,
                                                                   helper=training_helper,
                                                                   initial_state=encoder_state,
                                                                   output_layer=output_layer)
                training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=training_decoder,
                                                                                  impute_finished=True,
                                                                                  maximum_iterations=max_target_sequence_length)
                self.training_decoder_output = training_decoder_output
            else:
                # 创建一个常量tensor并复制为batch_size的大小
                start_tokens = tf.tile(tf.constant([self.target_start_flag_index], dtype=tf.int32), [self.batch_size],
                                       name='start_tokens')
                predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=decoder_embeddings,
                                                                             start_tokens=start_tokens,
                                                                             end_token=self.target_end_flag_index)

                # decoder_initial_state  encoder_state
                predicting_decoder = tf.contrib.seq2seq.BasicDecoder(cell,
                                                                     predicting_helper,
                                                                     initial_state=encoder_state
                                                                     , output_layer=output_layer)
                predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=predicting_decoder,
                                                                                    impute_finished=True,
                                                                                    maximum_iterations=self.max_pred_len)
                self.predicting_decoder_output = predicting_decoder_output
                self.predicting_decoder_output = self.predicting_decoder_output.sample_id

    def build_model(self):
        self.get_inputs()
        _, encoder_state = self.get_encoder_layer(input_data=self.inputs,
                                      source_sequence_length=self.source_sequence_length)

        if self.train_mode:
            decoder_input = self.process_decoder_input(self.targets)  # 预处理后的decoder输入
            self.decoding_layer(
                source_sequence_length=self.source_sequence_length,
                encoder_state=encoder_state,
                target_sequence_length=self.target_sequence_length,
                max_target_sequence_length=self.max_target_sequence_length,
                decoder_input=decoder_input, )
            self.masks = tf.sequence_mask(lengths=self.target_sequence_length,
                                          maxlen=self.max_target_sequence_length,
                                          dtype=tf.float32, name='masks')
            self.loss = tf.contrib.seq2seq.sequence_loss(logits=self.training_decoder_output.rnn_output,
                                                         targets=self.targets,
                                                         weights=self.masks,
                                                         )
            self.opt = tf.train.AdamOptimizer(self.learning_rate)
            gradients = self.opt.compute_gradients(self.loss)
            capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
            self.update = self.opt.apply_gradients(capped_gradients)
        else:
            self.decoding_layer(
                source_sequence_length=self.source_sequence_length,
                encoder_state=encoder_state)
        self.saver = tf.train.Saver()

    def train(self,
              sess,  # tensorflow Session
              encoder_inputs,  # (batch_size, None)
              encoder_inputs_length, # (None,)
              decoder_inputs, # (batch_size, None)
              decoder_inputs_length): # (None, )
        input_feed = {
            self.inputs.name: encoder_inputs,
            self.source_sequence_length.name: encoder_inputs_length,
            self.targets.name: decoder_inputs,
            self.target_sequence_length.name: decoder_inputs_length
        }
        output_feed = [
            self.update,
            self.loss]
        _, loss = sess.run(output_feed, input_feed)
        return loss

    def predict(self, sess, encoder_inputs, encoder_inputs_length):
        """
        预测
        :param sess: tensorflow Session
        :param encoder_inputs: (batch_size, None) 二维数组
        :param encoder_inputs_length: (None,)  一维数组

        :return: (batch_size, max_pred_len) 二维数组
        """
        input_feed = {
            self.inputs.name: encoder_inputs,
            self.source_sequence_length.name: encoder_inputs_length
        }
        pred = sess.run(self.predicting_decoder_output, input_feed)
        return pred

    def save(self, sess, save_path):
        """
        保存模型
        :param sess: tensorflow Session
        :param save_path: 保存地址
        :return: None
        """
        self.saver.save(sess, save_path=save_path)

    def load(self, sess, save_path):
        """
        加载模型
        :param sess: tensorflow Session
        :param save_path: 加载地址
        :return:None
        """
        self.saver.restore(sess, save_path)

    def params_set(self):
        """ 返回当前模型的参数 """
        return {
            "source_vocab_size":self.source_vocab_size,
            "target_vocab_size":self.target_vocab_size,
            "target_start_flag_index":self.target_start_flag_index,
            "target_end_flag_index":self.target_end_flag_index,
            "batch_size":self.batch_size,
            "encode_embed_dim":self.encode_embed_dim,
            "decode_embed_dim":self.decode_embed_dim,
            "max_pred_len":self.max_pred_len,
            "rnn_size":self.rnn_size,
            "num_layers":self.num_layers,
            "learning_rate":self.learning_rate,
            "train_mode":self.train_mode
        }


if __name__ == '__main__':
    model = Seq2Seq(source_vocab_size=64, target_vocab_size=64)
    print(model)

