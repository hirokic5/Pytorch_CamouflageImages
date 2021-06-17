class CFG:
    input_path="samples/inputs/kuma.png "
    mask_path="samples/inputs/kuma_mask.png"
    bg_path="samples/inputs/cliff.jpg"
    output_dir="samples/outputs/HRNet/Cliff_kuma"
    name="camouflage"
    mask_scale=0.25
    
    epoch=1000
    lr=5e-3
    step_size=100
    
    show_comp=4
    crop=True
    erode_border=True
    hidden_selected=None
    
    style_weight_dic={
        'conv1_1': 1.5,
        'conv2_1': 1.5,
        'conv3_1': 1.5,
        'conv4_1': 1.5,
    }
    style_all = False
    mu = 0.5
    alpha1 = 1
    alpha2 = 1
    beta = 1.5
    lambda_weights={
        "content":0,
        "style":1.0,
        "cam":1e0,
        "reg":1e0,
        "tv":1e-1
    }
    
    show_every = 100
    save_process = False
    