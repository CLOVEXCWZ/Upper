import os
import string
import random
import numpy as np

# 数据存储路径
char_lower_path="data/char_lower.txt"
char_upper_path="data/char_upper.txt"

def create_data(num=15000, max_len=20, min_len=5):
    """
    生成数据
    :param num: 数据数目
    :param max_len: 单条数据最大长度
    :param min_len: 单条数据小长度
    :return:  char_lowers char_uppers
    """
    # 检查数据是否存在如果已经存在就不再生成数据了
    if os.path.exists(char_lower_path) and os.path.exists(char_upper_path):
        with open(char_lower_path, 'r') as f:
            char_lowers=f.read().split("\n")
        with open(char_upper_path, 'r') as f:
            char_uppers = f.read().split("\n")
        return char_lowers, char_uppers

    # 对参数进行限制
    if num < 5000: num=5000
    if max_len < 5: max_len=5
    if min_len < 1: min_len=1

    chars = string.ascii_lowercase  # 获取小写字母(a~z)
    char_lowers = []
    char_uppers = []
    for _ in range(num):
        # 生成随机长度的字符串
        one_chars = [chars[random.randint(0, len(chars) - 1)] for _ in range(random.randint(min_len, max_len))]
        one_chars = "".join(one_chars)
        char_lowers.append(one_chars)
        char_uppers.append(one_chars.upper())

    # 保存数据
    with open(char_lower_path, 'w') as f:
        f.write("\n".join(char_lowers))
    with open(char_upper_path, 'w') as f:
        f.write("\n".join(char_uppers))

    return char_lowers, char_uppers


class CharData(object):
    def __init__(self):

        # 特殊标记flage定义
        self.start_flag = '<GO>'
        self.end_flag = '<EOS>'
        self.pad_flag = '<PAD>'
        self.unk_flag = '<UNK>'

        # 特殊标记对应的index
        self.start_index = 0
        self.end_index = 1
        self.pad_index = 2
        self.unk_index = 3

        self.index_to_char = None  # dict: index to char
        self.char_to_index = None  # dict: char to index
        self.source_indexs = None  # 原数据转化为 index形式的数组
        self.target_indexs = None  # 目标数据转化为 index形式的数组
        self.vocab_size = None  # vocab长度

        self.load_data()
        self.word_to_index()

    def load_data(self):
        char_lowers, char_uppers=create_data(num=15000, max_len=20, min_len=5)
        all_chars = "".join(char_lowers) + "".join(char_uppers) # 获取全部文本数据
        char_list = sorted(list(set(all_chars)))  # 获取字符的数组
        char_list = [self.start_flag, self.end_flag, self.pad_flag, self.unk_flag] + char_list  # 把特殊标记加入到数组中
        self.index_to_char = {idx: char for idx, char in enumerate(char_list)}  # 建立 index_to_char字典
        self.char_to_index = {char: idx for idx, char in enumerate(char_list)}  # 建立char_to_index字典
        self.vocab_size = len(self.index_to_char)

    def word_to_index(self):
        """ 把数据转化成index的形式 """
        char_lowers, char_uppers = create_data()
        def text_to_index(texts, char_to_index):  # 把数据转化为Index的形式
            texts_indexs = []
            for item in texts:
                texts_indexs.append([char_to_index.get(char, self.unk_index) for char in item]+ [self.end_index])
            return texts_indexs

        self.source_indexs = text_to_index(char_lowers, self.char_to_index)  # 原句子转化为Index形式
        self.target_indexs = text_to_index(char_uppers, self.char_to_index)  # 目标句子转化为index形式

    def pad_sentence_batch(self, sentence_batch, pad_int):
        """ 补全数据 """
        max_sentence = max([len(sentence) for sentence in sentence_batch])
        return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]

    def get_batches(self, batch_size=32):
        for batch_i in range(0, len(self.source_indexs) // batch_size):
            start_i = batch_i * batch_size
            sources_batch = self.source_indexs[start_i:start_i + batch_size]
            targets_batch = self.target_indexs[start_i:start_i + batch_size]

            # 补全序列
            pad_sources_batch = np.array(self.pad_sentence_batch(sources_batch, self.pad_index))
            pad_targets_batch = np.array(self.pad_sentence_batch(targets_batch, self.pad_index))

            # 记录每条记录的长度
            targets_lengths = []
            for target in targets_batch:
                targets_lengths.append(len(target))

            source_lengths = []
            for source in sources_batch:
                source_lengths.append(len(source))

            pad_targets_batch = np.array(pad_targets_batch)
            pad_sources_batch = np.array(pad_sources_batch)
            targets_lengths = np.array(targets_lengths)
            source_lengths = np.array(source_lengths)

            yield pad_targets_batch, pad_sources_batch, targets_lengths, source_lengths

    def pred_text_to_ids(self, texts):
        """
        预测text编码
        :param texts:  (batch, None) 二维数组
        :return:  inputs_pad, lengs, max_input_len
        """
        texts_indexs = []
        for item in texts:
            texts_indexs.append([self.char_to_index.get(char, self.unk_index) for char in item])
        max_sentence = max([len(sentence) for sentence in texts_indexs])
        inputs_pad = [sentence + [self.end_index] + [self.pad_index] * (max_sentence - len(sentence)) for sentence in texts_indexs]
        lengs = [len(item) + 1 for item in texts]
        max_input_len = max([len(item) for item in texts]) + 1
        return inputs_pad, lengs, max_input_len

    def index_to_text(self, ids):
        """
        预测数据解码
        :param ids:  (batch_size, None) 二维数组
        :return: texts
        """
        end_index = 1
        texts = []
        for item in ids:
            chars = []
            for index in item:
                if index == self.end_index:
                    break
                chars = chars + [self.index_to_char.get(index, self.unk_flag)]
            texts.append("".join(chars))
        return texts


if __name__ == '__main__':
    dp = CharData()
    pad_targets_batch, pad_sources_batch, targets_lengths, source_lengths=next(dp.get_batches())
    print(pad_targets_batch.shape)
    print(pad_sources_batch.shape)
    print(targets_lengths.shape)
    print(source_lengths.shape)

