import torch
import numpy as np
from PIL import Image
import cv2


def scaling(img,scale=1.0):
    h,w=img.shape[:2]
    new_h = int(h * scale)
    new_w = int(w * scale)
    return cv2.resize(img,(new_w,new_h))

def im_convert(tensor):
    """ Display a tensor as an image. """

    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze(0)    # change size to (channel, height, width)

    '''
        tensor (batch, channel, height, width)
        numpy.array (height, width, channel)
        to transform tensor to numpy, tensor.transpose(1,2,0) 
    '''
    image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))   # change into unnormalized image
    image = image.clip(0, 1)    # in the previous steps, we change PIL image(0, 255) into tensor(0.0, 1.0), so convert it

    return image


def get_features(image, model, mode="style"):
    '''
        return a dictionary consists of each layer's name and it's feature maps
    '''
    if mode=="style":
        layers = {
            '0': 'conv1_1',   # default style layer
            '5': 'conv2_1',   # default style layer
            '10': 'conv3_1',# default style layer
            #'12': 'conv3_2',# for attention
            #'14': 'conv3_3',# for attention
            #'16': 'conv3_4',# for attention
            '19': 'conv4_1',  # default style layer
            #'21': 'conv4_2',  # for attention
            #'23': 'conv4_3',  # for attention
            #'25': 'conv4_4',  # for attention
            #'28': 'conv5_1'
        }
    elif mode=="camouflage":
        layers = {
            #'0': 'conv1_1',   # default style layer
            #'5': 'conv2_1',   # default style layer
            '10': 'conv3_1',# default style layer
            '12': 'conv3_2',# for attention
            '14': 'conv3_3',# for attention
            '16': 'conv3_4',# for attention
            '19': 'conv4_1',  # default style layer
            '21': 'conv4_2',  # for attention
            '23': 'conv4_3',  # for attention
            '25': 'conv4_4',  # for attention
        }
    elif mode=="content":
        layers = {
            #'0': 'conv1_1',   # default style layer
            #'5': 'conv2_1',   # default style layer
            #'10': 'conv3_1',# default style layer
            #'12': 'conv3_2',# for attention
            #'14': 'conv3_3',# for attention
            #'16': 'conv3_4',# for attention
            #'19': 'conv4_1',  # default style layer
            '21': 'conv4_2',  # for attention
            #'23': 'conv4_3',  # for attention
            #'25': 'conv4_4',  # for attention
        }
    features = {}
    x = image
    # model._modules is a dictionary holding each module in the model
    for name, layer in model._modules.items():
        x = layer(x)    #  layer(x) is the feature map through the layer when the input is x
        if name in layers:
            features[layers[name]] = x
    
    return features

def attention_map_cv(img):
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    _,ch,h,w=img.shape
    att_cv=np.zeros((h,w))
    for j in range(ch):
        (success, saliencyMap) = saliency.computeSaliency(img[0,j,:,:])
        att_cv += saliencyMap
    return att_cv

def gram_matrix_slice(tensor,idxes):
    b, c, h, w = tensor.size()
    tensor = tensor.view(b * c, h * w)
    tensor = tensor[:,idxes]
    gram_matrix = torch.mm(tensor, tensor.t())
    return gram_matrix
