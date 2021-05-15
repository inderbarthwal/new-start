import os.path as osp
import glob
import cv2, os
import numpy as np
import torch 
import ResNet_arch as arch

def test1():
    model_path = 'models/image-super-resolution.pth' 
    device = torch.device('cpu')
    test_img_folder = 'LR/*'
    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)
    idx = 0
    fileName = '' 
    for path in glob.glob(test_img_folder):
        idx += 1
        base = osp.splitext(osp.basename(path))[0]
        img = cv2.imread(path, cv2.IMREAD_COLOR)     
        img = img * 1.0 / 255
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = img.unsqueeze(0)
        img_LR = img_LR.to(device)

        with torch.no_grad():
            output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round()
        cv2.imwrite('Static/{:s}.png'.format(base), output)

        fileName = base
    return fileName



    

