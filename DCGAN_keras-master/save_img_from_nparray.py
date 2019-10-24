import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from datetime import datetime
import cv2

class SaveImgFromList:
    def __init__(self, imgs, shape, tag=None, comment=None):
        self.imgs = [np.array(img) * 255. for img in imgs]
        for i in self.imgs:
            print(type(i))
            print(type(i[0]))
            print(i)
        self.shape = shape
        if comment is not None:
            comment = "_" + comment
        self.path = os.getcwd() + r"\corrected_img\{}.jpg".format(datetime.now().strftime("%Y%m%d%H%M%S")+comment)

    def __call__(self, *args, **kwargs):
        if self.shape[0] == 1  or self.shape[1] == 1:
            return
        # print(self.img)
        imgs = [np.reshape(img, self.shape) for img in self.imgs]
        # print("img:{}\n{}".format(img.shape, img))
        # print(self.shape)
        pil_imgs = [Image.fromarray(img) for img in imgs]
        for pil_img in pil_imgs:
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
        pil_imgs = vconcat_resize_min(pil_imgs)
        print("saved to -> {}".format(self.path))
        pil_imgs.save(self.path)
        return

def vconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    w_min = max(im.shape[1] for im in im_list)
    im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation)
                      for im in im_list]
    return cv2.vconcat(im_list_resize)
