import random
import string

import loguru
import pandas as pd
import csv
import retry
from openai import OpenAI

# Configure logger
logger = loguru.logger


class Perturbation:
    """Base class for random perturbations."""

    def __init__(self, q):
        self.q = q
        self.alphabet = string.printable


class RandomSwapPerturbation(Perturbation):
    """Implementation of random swap perturbations.
    See `RandomSwapPerturbation` in lines 1-5 of Algorithm 2."""

    def __init__(self, q):
        """
        初始化随机交换扰动类。

        Args:
            q (int): 替换比例，表示字符串中需要替换的字符百分比。
        """
        super(RandomSwapPerturbation, self).__init__(q)

    def __call__(self, s):
        """
        对输入的字符串进行随机字符替换。

        Args:
            s (str): 需要处理的字符串。

        Returns:
            str: 经过随机字符替换后的字符串。
        """
        # 将字符串转换为列表，方便进行字符替换
        list_s = list(s)

        # 随机选择需要替换的字符索引
        sampled_indices = random.sample(range(len(s)), int(len(s) * self.q / 100))

        # 对选中的索引进行字符替换
        for i in sampled_indices:
            list_s[i] = random.choice(self.alphabet)

        # 将列表重新拼接成字符串并返回
        return ''.join(list_s)


class RandomPatchPerturbation(Perturbation):
    """Implementation of random patch perturbations.
    See `RandomPatchPerturbation` in lines 6-10 of Algorithm 2."""

    def __init__(self, q):
        super(RandomPatchPerturbation, self).__init__(q)

    def __call__(self, s):
        # 将输入的字符串s转换为列表，以便进行字符替换
        list_s = list(s)

        # 计算子字符串的宽度，根据self.q的值来确定百分比
        substring_width = int(len(s) * self.q / 100)

        # 计算替换子字符串的最大起始位置，确保子字符串不会超出原始字符串的范围
        max_start = len(s) - substring_width

        # 随机选择子字符串的起始索引，确保替换操作的位置是随机的
        start_index = random.randint(0, max_start)

        # 生成与子字符串宽度相等的随机字符序列，这些字符来自self.alphabet
        sampled_chars = ''.join([
            random.choice(self.alphabet) for _ in range(substring_width)
        ])

        # 使用生成的随机字符序列替换原始字符串中的相应部分
        list_s[start_index:start_index + substring_width] = sampled_chars

        # 将修改后的字符列表转换回字符串并返回
        return ''.join(list_s)


class RandomInsertPerturbation(Perturbation):
    """Implementation of random insert perturbations.
    See `RandomPatchPerturbation` in lines 11-17 of Algorithm 2."""

    def __init__(self, q):
        super(RandomInsertPerturbation, self).__init__(q)

    def __call__(self, s):
        """
        当实例被当作函数调用时执行的特殊方法。

        该方法接受一个字符串s作为输入，根据预定义的采样概率q，
        随机选择字符串中的某些位置，并在这些位置插入随机选择的字符，
        最终返回修改后的字符串。

        参数:
        - s: 输入的字符串。

        返回值:
        - 修改后的字符串，其中根据采样概率q随机插入了一些字符。
        """
        # 将输入字符串s转换为字符列表，便于后续操作
        list_s = list(s)

        # 根据预定义的采样概率q，计算需要插入字符的索引位置
        # 这里使用random.sample来随机选择一定数量的索引，确保随机性
        sampled_indices = random.sample(range(len(s)), int(len(s) * self.q / 100))

        # 遍历每个选中的索引位置，准备插入随机字符
        for i in sampled_indices:
            # 在选中的索引位置插入一个从预定义字母表中随机选择的字符
            list_s.insert(i, random.choice(self.alphabet))

        # 将修改后的字符列表转换回字符串，并返回
        return ''.join(list_s)


class RandomDeletePerturbation(Perturbation):
    """Implementation of random delete perturbations.
    随机删除扰动类，用于对字符串进行随机字符删除。

    Attributes:
        q (int): 删除比例，表示字符串中需要删除的字符百分比。
    """

    def __init__(self, q):
        """
        初始化随机删除扰动类。

        Args:
            q (int): 删除比例，表示字符串中需要删除的字符百分比。
        """
        super(RandomDeletePerturbation, self).__init__(q)

    def __call__(self, s):
        """
        对输入的字符串进行随机字符删除。

        Args:
            s (str): 需要处理的字符串。

        Returns:
            str: 经过随机字符删除后的字符串。
        """
        # 将字符串转换为列表，方便进行字符删除
        list_s = list(s)

        # 随机选择需要删除的字符索引
        sampled_indices = random.sample(range(len(s)), int(len(s) * self.q / 100))

        # 对选中的索引进行字符删除
        # 注意：从后往前删除，避免索引错位
        for i in sorted(sampled_indices, reverse=True):
            del list_s[i]

        # 将列表重新拼接成字符串并返回
        return ''.join(list_s)


class CombinedPerturbation(Perturbation):
    """Implementation of combined perturbations.
    组合扰动类，用于对字符串同时进行随机交换、随机插入和随机删除操作。

    Attributes:
        q (int): 扰动比例，表示字符串中需要扰动的字符百分比。
    """

    def __init__(self, q):
        """
        初始化组合扰动类。

        Args:
            q (int): 扰动比例，表示字符串中需要扰动的字符百分比。
        """
        super(CombinedPerturbation, self).__init__(q)

    def __call__(self, s):
        """
        对输入的字符串同时进行随机交换、随机插入和随机删除操作。

        Args:
            s (str): 需要处理的字符串。

        Returns:
            str: 经过组合扰动后的字符串。
        """
        # 创建三种扰动类的实例
        swap_perturbation = RandomSwapPerturbation(self.q)
        insert_perturbation = RandomInsertPerturbation(self.q)
        delete_perturbation = RandomDeletePerturbation(self.q)

        # 依次应用三种扰动
        s = swap_perturbation(s)  # 随机交换
        s = insert_perturbation(s)  # 随机插入
        s = delete_perturbation(s)  # 随机删除

        return s

if __name__ == '__main__':

    prompt = "This is a test text."
    # 设置扰动比例
    q = 4
    perturbation = CombinedPerturbation(q)
    print(perturbation(prompt))

