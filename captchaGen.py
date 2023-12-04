import os
from captcha.image import ImageCaptcha
import random
from setting import *

# 验证码中的字符, 就不用汉字了
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']


# 验证码一般都无视大小写；验证码长度4个字符
def random_captcha_text(char_set=None, ):
    if char_set is None:
        char_set = number + alphabet + ALPHABET
    captcha_text = []
    for kkk in range(captcha_len):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text


# 生成字符对应的验证码

def gen_captcha_text_and_image(save_path):
    image = ImageCaptcha()
    captcha_text = random_captcha_text()
    captcha_text = ''.join(captcha_text)
    captcha = image.generate(captcha_text)

    # 使用 with open 保存验证码图片到本地
    file_path1 = save_path + captcha_text + ".png"  # 您可以修改路径和文件名格式
    with open(file_path1, 'wb') as f:
        f.write(captcha.read())


if __name__ == '__main__':

    if not os.path.isdir(train_path):
        os.makedirs(train_path)
    if not os.path.isdir(test_path):
        os.makedirs(test_path)

    # 生成验证码并显示
    for i in range(test_num):
        gen_captcha_text_and_image(test_path)
    for i in range(train_num):
        gen_captcha_text_and_image(train_path)
