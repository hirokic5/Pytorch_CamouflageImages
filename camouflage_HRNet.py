import os
from PIL import Image
import cv2
import numpy as np
import argparse
from importlib import import_module
from albumentations import Normalize,Compose
from albumentations.pytorch import ToTensorV2
from torchvision import transforms, models
import torch
import torch.optim as optim
import torch.nn as nn
import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold.locally_linear import barycenter_kneighbors_graph


import HRNet
import utils
from hidden_recommend import recommend
from loss import calc_attentionmap,cosine_distance,total_variation_loss,calc_weightMatrix


def attention_map_cv(img):
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    _,ch,h,w=img.shape
    att_cv=np.zeros((h,w))
    for j in range(ch):
        (success, saliencyMap) = saliency.computeSaliency(img[0,j,:,:])
        att_cv += saliencyMap
    return att_cv

def main(args):
    i_path=args.input_path
    m_path=args.mask_path
    bg_path=args.bg_path
    

    camouflage_dir=args.output_dir
    os.makedirs(camouflage_dir,exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    VGG = models.vgg19(pretrained=True).features
    VGG.to(device)

    for parameter in VGG.parameters():
        parameter.requires_grad_(False)

    style_net = HRNet.HRNet()
    style_net.to(device)

    transform = Compose([
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
    ])   

    # try to give fore con_layers more weight so that can get more detail in output iamge
    style_weights = args.style_weight_dic
            
    mask=cv2.imread(m_path,0)
    mask=utils.scaling(mask,scale=args.mask_scale)
    
    
    idx_y,idx_x=np.where(mask>0)
    x1_m,y1_m,x2_m,y2_m=np.min(idx_x),np.min(idx_y),np.max(idx_x),np.max(idx_y)
    x1_m =8*(x1_m //8)
    x2_m =8*(x2_m //8)
    y1_m =8*(y1_m //8)
    y2_m =8*(y2_m //8)
    
    fore_origin=cv2.cvtColor(cv2.imread(i_path),cv2.COLOR_BGR2RGB)
    fore_origin=utils.scaling(fore_origin,scale=args.mask_scale)
    fore=fore_origin[y1_m:y2_m,x1_m:x2_m]
   
    
    mask_crop=mask[y1_m:y2_m,x1_m:x2_m]
    mask_crop=np.where(mask_crop>0,255,0).astype(np.uint8)
    kernel = np.ones((15,15),np.uint8)
    mask_dilated=cv2.dilate(mask_crop,kernel,iterations = 1)
    mat_dilated=fore*np.expand_dims(mask_dilated/255,axis=-1)


    origin=cv2.cvtColor(cv2.imread(bg_path),cv2.COLOR_BGR2RGB)
    h_origin,w_origin,_ = origin.shape
    h,w=mask_dilated.shape
    assert h < h_origin, "mask height must be smaller than bg height, and lower mask_scale parameter!!"
    assert w < w_origin, "mask width must be smaller than bg width, and lower mask_scale parameter!!"
    
    print("mask size,height:{},width:{}".format(h,w))
    if args.hidden_selected is None:
        y_start,x_start=recommend(origin,fore,mask_dilated)
    else:
        y_start,x_start=args.hidden_selected
        
    x1,y1=x_start+x1_m,y_start+y1_m
    x2,y2=x1+w,y1+h
    if y2 > h_origin:
        y1 -= (y2-h_origin)
        y2 = h_origin
    if x2 > w_origin:
        x1 -= (x2-w_origin)
        x2 = w_origin
        
    print("hidden region...,height-{}:{},width-{}:{}".format(y1,y2,x1,x2))
    bg=origin.copy()
    bg[y1:y2,x1:x2] = fore*np.expand_dims(mask_crop/255,axis=-1) + origin[y1:y2,x1:x2]*np.expand_dims(1-mask_crop/255,axis=-1)
    
    content_image = transform(image=mat_dilated)["image"].unsqueeze(0)
    style_image = transform(image=origin[y1:y2,x1:x2])["image"].unsqueeze(0)
    #style_image = transform(image=cv2.resize(origin,(x2-x1,y2-y1)))["image"].unsqueeze(0)
    content_image = content_image.to(device)
    style_image = style_image.to(device)

    style_features   = utils.get_features(style_image, VGG,mode="style")
    style_gram_matrixs = {layer: utils.get_gram_matrix(style_features[layer]) for layer in style_features}

    target = content_image.clone().requires_grad_(True).to(device)

    foreground_features=utils.get_features(content_image, VGG,mode="camouflage")
    target_features = foreground_features.copy()
    attention_layers=[
        "conv3_1","conv3_2","conv3_3","conv3_4",
        "conv4_1","conv4_2","conv4_3","conv4_4",
    ]

    for u,layer in enumerate(attention_layers):
        target_feature = target_features[layer].detach().cpu().numpy()  # output image's feature map after layer
        attention=attention_map_cv(target_feature)
        print(attention.shape)
        h,w=attention.shape
        if "conv3" in layer:
            attention=cv2.resize(attention,(w//2,h//2))
        if u== 0:
            all_attention = attention
        else:
            all_attention += attention
    all_attention /= 5
    max_att,min_att = np.max(all_attention),np.min(all_attention)
    all_attention = (all_attention-min_att) / (max_att-min_att)
    foreground_attention= torch.from_numpy(all_attention.astype(np.float32)).clone().to(device)
    b,ch,h,w=foreground_features["conv4_1"].shape

    foreground_cosine=np.zeros([b,ch,h,h])
    for k in range(ch):
        cos_matrix=cosine_similarity((foreground_attention*foreground_features["conv4_1"][0,k,:]).detach().cpu().numpy())
        cos_matrix /= (np.sum(cos_matrix)+1e-10)
        foreground_cosine[0,k,:]=1-cos_matrix

    foreground_cosine = torch.from_numpy(foreground_cosine.astype(np.float32)).clone().to(device)

    background_features=utils.get_features(style_image, VGG,mode="camouflage")
    
    idxes=np.where(mask_dilated>0)
    n_neighbors,n_jobs,reg=7,None,1e-3
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, n_jobs=n_jobs)
    X_origin=origin[y1:y2,x1:x2][idxes] / 255
    nbrs.fit(X_origin)
    X = nbrs._fit_X
    Weight_Matrix = barycenter_kneighbors_graph(
                nbrs, n_neighbors=n_neighbors, reg=reg, n_jobs=n_jobs)
    
    
    mask_norm=mask_crop/255.
    

    content_loss_epoch = []
    style_loss_epoch = []
    total_loss_epoch = []
    time_start=datetime.datetime.now()
    epoch=0
    show_every = args.show_every
    optimizer = optim.Adam(style_net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.9)
    steps = args.epoch
    mse = nn.MSELoss()
    while epoch <= steps:
        target = style_net(content_image).to(device)
        target.requires_grad_(True)


        target_features = utils.get_features(target, VGG)  # extract output image's all feature maps
        
        #############################
        ### content loss    #########
        #############################
        target_features_content = utils.get_features(target, VGG,mode="content") 
        content_loss = torch.sum((target_features_content['conv4_2'] - foreground_features['conv4_2']) ** 2) / 2


        #############################
        ### style loss      #########
        #############################
        style_loss = 0

        # compute each layer's style loss and add them
        for layer in style_weights:
            target_feature = target_features[layer]  # output image's feature map after layer
            target_gram_matrix = utils.get_gram_matrix(target_feature)
            style_gram_matrix = style_gram_matrixs[layer]
            b, c, h, w = target_feature.shape
            layer_style_loss = style_weights[layer] * torch.sum((target_gram_matrix - style_gram_matrix) ** 2) / ((2*c*w*h)**2)
            #layer_style_loss = style_weights[layer] * torch.mean((target_gram_matrix - style_gram_matrix) ** 2) 
            style_loss += layer_style_loss


        
        #############################
        ### camouflage loss #########
        #############################

        target_cosine=np.zeros([b,ch,h,h])
        for k in range(ch):
            cos_matrix=cosine_similarity((foreground_attention*target_features["conv4_1"][0,k,:]).detach().cpu().numpy())
            cos_matrix /= (np.sum(cos_matrix)+1e-10)
            target_cosine[0,k,:]=1-cos_matrix

        target_cosine = torch.from_numpy(target_cosine.astype(np.float32)).clone().to(device)

        leave_loss = (torch.mean(torch.abs(target_cosine-foreground_cosine))/2).to(device)
        remove_matrix=torch.empty([b,ch,h,w])
        for k in range(ch):
            remove_matrix[0,k,:] = (1.0-foreground_attention)*(target_features["conv4_1"][0,k,:]-background_features["conv4_1"][0,k,:])
        remove_loss = (torch.mean(remove_matrix**2)/2).to(device)

        camouflage_loss = leave_loss + args.mu*remove_loss
        
        #############################
        ### regularization loss #####
        #############################
        
        target_renormalize = target.detach().cpu().numpy()[0,:].transpose(1,2,0)
        target_renormalize = target_renormalize * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))  
        target_renormalize = target_renormalize.clip(0,1)[idxes]
        target_reconst = torch.from_numpy((Weight_Matrix*target_renormalize).astype(np.float32))
        target_renormalize= torch.from_numpy(target_renormalize.astype(np.float32))
        reg_loss = mse(target_renormalize,target_reconst).to(device)
        
        
        #############################
        ### total variation loss ####
        #############################
        
        tv_loss=total_variation_loss(target,True)
        

        
        total_loss = args.lambda_weights["content"]*content_loss + args.lambda_weights["style"]*style_loss + args.lambda_weights["cam"]*camouflage_loss + args.lambda_weights["reg"]*reg_loss + args.lambda_weights["tv"]*tv_loss
        total_loss_epoch.append(total_loss)

        style_loss_epoch.append(style_loss)
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % show_every == 0:
            print("After %d criterions:" % epoch)
            print('Total loss: ', total_loss.item())
            print('camouflage loss: ', camouflage_loss.item())
            print('regularization loss: ', reg_loss.item())
            print('total variation loss: ', tv_loss.item())
            print('Style loss: ', style_loss.item())
            print('content loss: ', content_loss.item())
            print("elapsed time:{}".format(datetime.datetime.now()-time_start))
            canvas=origin.copy()
            fore_gen=utils.im_convert(target) * 255.
            canvas[y1:y2,x1:x2]=fore_gen*np.expand_dims(mask_norm,axis=-1) + origin[y1:y2,x1:x2]*np.expand_dims(1.0-mask_norm,axis=-1)
            new_path=os.path.join(camouflage_dir,"{}_epoch{}.png".format(args.name,epoch))
            canvas=canvas.astype(np.uint8)
            cv2.imwrite(new_path,cv2.cvtColor(canvas,cv2.COLOR_RGB2BGR))
            cv2.rectangle(canvas,(x1,y1),(x2,y2),(255,0,0),10)
            cv2.rectangle(canvas,(x1-x1_m,y1-y1_m),(x2,y2),(255,255,0),10)
            canvas=np.vstack([canvas,bg])
            canvas=canvas.astype(np.uint8)
            canvas=cv2.cvtColor(canvas,cv2.COLOR_RGB2BGR)
            h_show,w_show,c=canvas.shape
            cv2.imshow("now camouflage...",cv2.resize(canvas,(w_show//args.show_comp,h_show//args.show_comp)))

            
        epoch+=1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    time_end=datetime.datetime.now()
    print('totally cost:{}'.format(time_end - time_start))
    new_path=os.path.join(camouflage_dir,"{}.png".format(args.name))
    canvas=origin.copy()
    fore_gen=utils.im_convert(target) * 255.
    canvas[y1:y2,x1:x2]=fore_gen*np.expand_dims(mask_norm,axis=-1) + origin[y1:y2,x1:x2]*np.expand_dims(1.0-mask_norm,axis=-1)
    canvas=canvas.astype(np.uint8)
    canvas=cv2.cvtColor(canvas,cv2.COLOR_RGB2BGR)
    cv2.imwrite(new_path,canvas)
        
        

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', type=str, required=True, default="params")
    args = parser.parse_args()
    params = import_module(args.params)
    main(params.CFG)