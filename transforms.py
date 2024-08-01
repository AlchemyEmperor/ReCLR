from imgaug import augmenters as iaa
import imgaug as ia
import numpy as np
from PIL import Image
from imgaug import parameters as iap

class ImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.Resize((224, 224)),
            iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0))),
            iaa.Fliplr(0.5),
            iaa.Affine(rotate=(-20, 20), mode='symmetric'),
            iaa.Sometimes(0.25,
                          iaa.OneOf([iaa.Dropout(p=(0, 0.1)),
                                     iaa.CoarseDropout(0.1, size_percent=0.5)])),
            iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)
        ])
    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)

class Raw:
    def __init__(self):
        pass

    def __call__(self, img):
        return img


class ChannelShuffle:
    def __init__(self):
        self.aug = iaa.ChannelShuffle(p=1, channels=[0, 1])

    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)

class BlendAlpha:
    def __init__(self):
        self.aug = iaa.BlendAlpha(0.5, iaa.Grayscale(1.0))

    def __call__(self, img):
        img = np.array(img)
        img = Image.fromarray(self.aug.augment_image(img))
        return img


class GammaContrast:
    def __init__(self):
        self.aug = iaa.GammaContrast((0.5, 2.0), per_channel=True)

    def __call__(self, img):
        img = np.array(img)

        img = Image.fromarray(self.aug.augment_image(img))

        return img


class ElasticTransformation:
    def __init__(self):
        self.aug = iaa.ElasticTransformation(alpha=150.0, sigma=15)

    def __call__(self, img):
        img = np.array(img)

        img = Image.fromarray(self.aug.augment_image(img))

        return img


class AveragePooling:
    def __init__(self):
        self.aug = iaa.AveragePooling(5)

    def __call__(self, img):
        img = np.array(img)

        img = Image.fromarray(self.aug.augment_image(img))

        return img


class Superpixels:
    def __init__(self):
        self.aug = iaa.Superpixels(p_replace=1, n_segments=128)

    def __call__(self, img):
        img = np.array(img)

        img = Image.fromarray(self.aug.augment_image(img))

        return img


#用高斯模糊，均值模糊，中值模糊中的一种增强
class GaussianBlur:
    def __init__(self):
        self.aug = iaa.GaussianBlur((0, 3.0))

    def __call__(self, img):
        img = np.array(img)

        img = Image.fromarray(self.aug.augment_image(img))

        return img

class AverageBlur:
    def __init__(self):
        self.aug = iaa.AverageBlur(k=(2, 7))

    def __call__(self, img):
        img = np.array(img)

        img = Image.fromarray(self.aug.augment_image(img))

        return img

class AverageBlur2:
    def __init__(self):
        self.aug = iaa.MedianBlur(k=(3, 11))

    def __call__(self, img):
        img = np.array(img)

        img = Image.fromarray(self.aug.augment_image(img))

        return img

#锐化处理
class Sharpen:
    def __init__(self):
        self.aug = iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),

    def __call__(self, img):
        img = np.array(img)

        img = Image.fromarray(self.aug.augment_image(img))

        return img
#浮雕效果
class Sharpen:
    def __init__(self):
        self.aug = iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),

    def __call__(self, img):
        img = np.array(img)

        img = Image.fromarray(self.aug.augment_image(img))

        return img

#加入高斯噪声
class AdditiveGaussianNoise:
    def __init__(self):
        self.aug = iaa.AdditiveGaussianNoise(scale=(0, 0.2 * 255))

    def __call__(self, img):
        img = np.array(img)
        img = Image.fromarray(self.aug.augment_image(img))
        return img

class BilateralBlur:
    def __init__(self):
        self.aug = iaa.BilateralBlur(d=(3, 10), sigma_color=(10, 250), sigma_space=(10, 50))

    def __call__(self, img):
        img = np.array(img)
        img = Image.fromarray(self.aug.augment_image(img))
        return img



class GuassianBlur:
    def __init__(self):
        self.aug = iaa.GaussianBlur(sigma=(0.0, 3.0))

    def __call__(self, img):
        img = np.array(img)
        img = Image.fromarray(self.aug.augment_image(img))
        return img

class MedianBlur:
    def __init__(self):
        self.aug = iaa.MedianBlur(k=(3, 11))

    def __call__(self, img):
        img = np.array(img)
        img = Image.fromarray(self.aug.augment_image(img))
        return img

class MotionBlur:
    def __init__(self):
        self.aug = iaa.MotionBlur(k=7)

    def __call__(self, img):
        img = np.array(img)
        img = Image.fromarray(self.aug.augment_image(img))
        return img

class GuassianNoise:
    def __init__(self):
        self.aug = iaa.AdditiveGaussianNoise(scale=(0, 0.2 * 255))

    def __call__(self, img):
        img = np.array(img)
        img = Image.fromarray(self.aug.augment_image(img))
        return img


class Emboss:
    def __init__(self):
        self.aug = iaa.Emboss(alpha=(0.0, 1.0), strength=(0.5, 1.5))

    def __call__(self, img):
        img = np.array(img)
        img = Image.fromarray(self.aug.augment_image(img))
        return img

class FilterEdgeEnhance:
    def __init__(self):
        self.aug = iaa.pillike.FilterEdgeEnhance()

    def __call__(self, img):
        img = np.array(img)
        img = Image.fromarray(self.aug.augment_image(img))
        return img


class CoarseDropout:
    def __init__(self):
        self.aug = iaa.OneOf([
                    iaa.Dropout((0.01, 0.1), per_channel=0.5),
                    iaa.CoarseDropout(
                        (0.03, 0.15), size_percent=(0.02, 0.05),
                        per_channel=0.2
                    ),
                ]),

    def __call__(self, img):
        img = np.array(img)
        img = Image.fromarray(self.aug.augment_image(img))
        return img

class Affine:
    def __init__(self):
        self.aug = iaa.Affine(
        rotate=iap.Normal(0.0, 30),
        translate_px=iap.RandomSign(iap.Poisson(3))
    ),

    def __call__(self, img):
        img = np.array(img)
        img = Image.fromarray(self.aug.augment_image(img))
        return img