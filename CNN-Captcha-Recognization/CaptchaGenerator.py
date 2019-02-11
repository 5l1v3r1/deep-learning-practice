# ecnodig:utf-8
from captcha.image import ImageCaptcha
import random
import shutil
import os
import string
characters = string.digits + string.ascii_uppercase
print(characters)

width, height, n_len, n_class = 170, 80, 4, len(characters)
# width,height用于指定图片宽高
# n_len, n_class 验证码位数和候选字符种类个数

generator = ImageCaptcha(width=width, height=height)


def clear(path="img"):
    shutil.rmtree(path)
    os.mkdir(path)

def gen(num=1):
    for i in range(num):
        random_str = ''.join([random.choice(characters) for j in range(4)])
        img = generator.generate_image(random_str)
        img.save("img/"+random_str+".png")

if __name__ == "__main__":
    gen(10)
    clear()