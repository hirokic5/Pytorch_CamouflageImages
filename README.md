# Pytorch_CamouflageImages
Generate Camouflage Images by Pytorch
![Sample](https://user-images.githubusercontent.com/19792127/119169642-38c50580-ba9d-11eb-9ead-6e6a56356d8b.png)
this implementation is mainly based on **[Deep Camouflage Images]**(http://zhangqing-home.net/files/papers/2020/aaai2020.pdf) 

## Usage
```python camouflage_HRNet.py --params <path to parameter python file>```

For quick test, you can generate camouflage images by ```python camouflage_HRNet.py --params params_Cliff```

### Dependencies
- PyTorch (>= 1.7)
- torchvision
- opencv-contrib
- pillow
- scikit-learn
- scikit-image
- tqdm
- albumentations

### Preparaion
For custom images, you should prepare 
- background image (ex. samples/inputs/cliff.jpg)
- foreground image (ex. samples/inputs/kuma.png)
- foreground mask image (ex.samples/inputs/kuma_mask/png)
    - the size of foreground mask image must be **same as that of foreground image** !
- .py file with parameters corresponding for your data (ex. params_Cliff.py)

## Parameters
**initial setting**
```
- input_path : path to foreground image
- mask_path : path to foreground image mask
- bg_path : path to background image
- name : prefix name for generated image
- seed : pytorch seed
```

**mask setting**
```
- mask_scale : scale ratio for foreground image mask
- crop : crop foreground image mask to bounding box
- hidden_selected : If None, use hidden recommendation. If you wouldn't like to use hidden recommendation, you should give [y1_start,x1_start] for this parameter
```

**train setting**
```
- epoch : iteration for training
- lr : learning rate for adam
- step_size : step size for LR Scheduler
```

**loss setting**
```
- erode_border : If True, erode attention map
- style_weight_dic : dictionary of weights for style loss
- style_all : If True, use all background image for style loss. If False, use corresponding background image for style loss.
- alpha1 : scale parameter for leave loss
- alpha2 : scale parameter for remove loss
- mu : ratio of remove loss for camouflage loss
- lambda_weights : dictionary of weights for all loss
```

**log setting**
```
- show_every : interval for displaying intermediate result
- save_process : If True, save intermediate result by every `show_every`
- show_comp : compressino ratio for display
```

## Influence by loss function
According to **[Deep Camouflage Images]**(http://zhangqing-home.net/files/papers/2020/aaai2020.pdf), losses has following impact for generated image:

- style loss : control similarity between generated image and background image
- camouflage loss : control diffuculty for detection of camouflage objects in generated image
    - leave loss : leave foreground features in generated image
    - remove loss : remove foreground features in generated image
- reguralization loss : control consistency for generated image
- total variation loss : smooth generated image


## Gallary
With fix of seed, generated camouflage images could be duplicated **for some extent (not comletely** ...)


