# encoding: utf-8
"""
@author:  zhoumi
@contact: zhoumi281571814@126.com
"""
class crop_lt(object):
    def __init__(self, crop_h, crop_w):
        self.crop_h = crop_h
        self.crop_w = crop_w

    def __call__(self, x):
        return x.crop((0, 0, self.crop_w, self.crop_h))


class crop_lb(object):
    def __init__(self, crop_h, crop_w):
        self.crop_h = crop_h
        self.crop_w = crop_w

    def __call__(self, x):
        return x.crop((0, x.size[1] - self.crop_h ,self.crop_w, x.size[1]))

class crop_rt(object):
    def __init__(self, crop_h, crop_w):
        self.crop_h = crop_h
        self.crop_w = crop_w

    def __call__(self, x):
        return x.crop((x.size[0] - self.crop_w, 0 , x.size[0], self.crop_h))

class crop_rb(object):
    def __init__(self, crop_h, crop_w):
        self.crop_h = crop_h
        self.crop_w = crop_w

    def __call__(self, x):
        return x.crop((x.size[0] - self.crop_w, x.size[1] - self.crop_h, x.size[0], x.size[1]))

class center_crop(object):
    def __init__(self, crop_h, crop_w):
        self.crop_h = crop_h
        self.crop_w = crop_w

    def __call__(self, x):
        center_h = x.size[1] // 2
        center_w = x.size[0] // 2
        half_crop_h = self.crop_h // 2
        half_crop_w = self.crop_w // 2

        y_min = center_h - half_crop_h
        y_max = center_h + half_crop_h + self.crop_h % 2
        x_min = center_w - half_crop_w
        x_max = center_w + half_crop_w + self.crop_w % 2

        return x.crop((x_min, y_min, x_max, y_max))