import torch.fft as fft
import torch
import utils
from tqdm import tqdm as tqdm
import numpy as np

def attention_map(target_feature):
    ch=target_feature.shape[1]
    for k in range(ch):

        f_map=target_feature[0,k]
        fourie=fft.fft(f_map)
        
        log_fourie=torch.log(fourie.abs()+1e-10)
        g_log_fourie=utils.gaussian_blur((log_fourie.abs()).unsqueeze(0).unsqueeze(0),(3,3),(1,1))[0,0,:]

        if k== 0:
            attention = fft.ifft(log_fourie-g_log_fourie).abs()
        else:
            attention += fft.ifft(log_fourie-g_log_fourie).abs()
        """
        log_fourie=torch.log(fourie+1e-10)
        g_log_fourie=utils.gaussian_blur(log_fourie.unsqueeze(0).unsqueeze(0),(3,3),(1,1))[0,0,:]

        if k== 0:
            attention = fft.ifft(log_fourie-g_log_fourie).real
        else:
            attention += fft.ifft(log_fourie-g_log_fourie).real
        """

    return attention
    
def calc_attentionmap(target_features,normalize=True):
    attention_layers=[
        "conv3_1","conv3_2","conv3_3","conv3_4",
        "conv4_1","conv4_2","conv4_3","conv4_4",
    ]

    for u,layer in enumerate(attention_layers):
        target_feature = target_features[layer]  # output image's feature map after layer
        attention=attention_map(target_feature)
        if "conv3" in layer:
            attention=torch.nn.functional.max_pool2d(attention.unsqueeze(0),kernel_size=(2,2))
            attention=attention.squeeze_() / 4
        #print(layer,target_feature.shape,target_gram_matrix.shape,attention.shape)
        #print(attention[0,:15])
        if u== 0:
            all_attention = attention
        else:
            all_attention += attention
    all_attention /= 5
    max_att,min_att = torch.max(all_attention),torch.min(all_attention)
    if normalize:
        all_attention = (all_attention-min_att) / (max_att-min_att)
    return all_attention

def cosine_distance(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


def total_variation_loss(img,normalize=True):
    bs_img, c_img, h_img, w_img = img.size()
    tv_h = torch.pow(img[:,:,1:,:]-img[:,:,:-1,:], 2).sum()
    tv_w = torch.pow(img[:,:,:,1:]-img[:,:,:,:-1], 2).sum()
    if normalize:
        return (tv_h+tv_w)/(bs_img*c_img*h_img*w_img) 
    else:
        return (tv_h+tv_w)

def calc_weightMatrix(mask_dilated,fore,K=7):
    weight_array=[]
    pos_list=[]
    idx_y,idx_x=np.where(mask_dilated>0)
    img_normalize=fore / 255.
    for u in tqdm(range(len(idx_x))):
        x,y=idx_x[u],idx_y[u]
        center=img_normalize[y,x]
        windows=img_normalize[y-K//2:y+K//2+1,x-K//2:x+K//2+1]
        a,b,_=windows.shape
        if a*b == K**2:
            position=[]
            for j in range(y-K//2,y+K//2+1):
                for k in range(x-K//2,x+K//2+1):
                    position.append([j,k])
            pos_list.append(position)
            windows_reshape = windows.reshape(K**2,3)
            variance=np.cov(windows_reshape-center)
            #variance+=(np.trace(variance)*1e-3)*np.identity(K**2)
            variance+=(np.trace(variance)+1e-3)*np.identity(K**2)
            weight=np.dot(np.linalg.inv(variance),np.ones(K**2))
            weight/=np.sum(weight)
            weight_array.append(1-weight)

    h,w,c=fore.shape
    weight_matrix=np.zeros((h,w))

    for weight,pos in zip(weight_array,pos_list):
        for w,p in zip(weight,pos):
            y,x=p
            weight_matrix[y,x]+=w

    weight_matrix_torch=torch.from_numpy(weight_matrix)
    weight_matrix_torch=weight_matrix_torch.unsqueeze(0).unsqueeze(0)
    return weight_matrix_torch