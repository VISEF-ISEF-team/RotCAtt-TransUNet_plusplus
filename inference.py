import numpy as np
import SimpleITK as sitk
from glob import glob
from time import time
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import pickle
from utils import parse_args, write_csv, save_vol, resize_vol

import torch
import torch.nn.functional as F
from skimage.transform import resize as skires
from metrics import Dice, IOU, HD


class Inference:
    def __init__(self, config, id):
        self.config = config
        self.network = config.network
        self.name = config.name
        self.num_classes = config.num_classes
        self.dataset = config.dataset
        if self.dataset == 'VHSCDD': self.dataset += f'_{config.img_size}'
        self.batch_size = config.batch_size
        self.num_heads = 4
        self.id = id
        self.image_path = f'data/{self.dataset}/test_images/{id:04d}.nii.gz'
        self.label_path = f'data/{self.dataset}/test_labels/{id:04d}.nii.gz'
        self.pred_path = f'data/{self.dataset}/test_pred/{id:04d}_{self.network}.nii.gz'
        self.infer_log_file = f'outputs/{self.network}/{config.name}/inference_log.csv'
        self.dice_class_file = f'outputs/{self.network}/{config.name}/infer_dice_class.csv'
        self.iou_class_file = f'outputs/{self.network}/{config.name}/infer_iou_class.csv'
        self.vis_path = f'outputs/{self.network}/{self.name}/test/{id}/'
        self.index_slice = [30, 37, 70, 82, 110, 145]
        self.head_1, self.head_2, self.head_3, self.head_4 = [], [], [], []
        self.inference_image = False
        
        self.a_weights = {
            'size_1': {
                'head_1': [],
                'head_2': [],
                'head_3': [],
                'head_4': [],
            },
            'size_2': {
                'head_1': [],
                'head_2': [],
                'head_3': [],
                'head_4': [],
            },
            'size_3': {
                'head_1': [],
                'head_2': [],
                'head_3': [],
                'head_4': [],
            }
        }
        
        self.c_weights = {
            'size_1': [],
            'size_2': [],
            'size_3': [],
        }
        
        
    def volume_convert(self):
        image_paths = glob(f'data/{self.dataset}/images/{self.id:04d}_*.npy')
        label_paths = glob(f'data/{self.dataset}/labels/{self.id:04d}_*.npy')
        image_vol = []
        label_vol = []

        for image_path, label_path in zip(image_paths, label_paths):
            image = np.load(image_path)
            label = np.load(label_path)
            image_vol.append(image)
            label_vol.append(label)

        image_vol = np.array(image_vol)
        label_vol = np.array(label_vol)
        
        save_vol(image_vol, 'images', self.image_path)
        save_vol(label_vol, 'labels', self.label_path)
        
    def save_vis(self, a_weights, r_weights, c_weights):
        for s in range(len(a_weights)):
            save_vol(r_weights[s].detach().cpu().numpy(), 'images', self.vis_path + f'r_size_{s+1}')
            for l in range(len(a_weights[s])):
                self.c_weights[f'size_{s+1}'].append(c_weights[s][l][2].detach().cpu().numpy())
                for h in range(self.num_heads):    
                    self.a_weights[f'size_{s+1}'][f'head_{h+1}'].append(a_weights[s][l][2,h,:,:].detach().cpu().numpy())                
        
        
    def prediction(self):
        model_path = f'outputs/{self.config.network}/{self.config.name}/model.pth'
        model = torch.load(model_path)
        image = sitk.GetArrayFromImage(sitk.ReadImage(self.image_path, sitk.sitkFloat32))
        image = torch.from_numpy(image).to(torch.float32).cuda()
        
        # pipeline
        segment_size = self.batch_size
        residual = image.shape[0] % segment_size
        if 1 <= residual and residual <= 3: segment_size -= 1
        steps = int(image.shape[0] / segment_size)
        steps = steps + 1 if residual != 0 else steps
        
        # prediction
        model.eval()
        results = []
        start_t = time()
        
        start = 0
        end = steps    # end = steps
        pbar = tqdm(total=start-end)
        
        # if not os.path.exists(self.vis_path): os.mkdir(self.vis_path)
        with torch.no_grad():
            for i in range(segment_size*start, segment_size*end, segment_size):
                input = image[i:i+segment_size, :, :].unsqueeze(1).cuda()
                if self.network == 'RotCAtt_TrasnUNet_plusplus': logits, a_weights, r_weights, c_weights = model(input)  
                else: logits = model(input)
                logits = F.softmax(logits, dim=1)

                if self.network =='RotCAtt_TransUNet_plusplus': self.save_vis(a_weights, r_weights, c_weights)
                _, pred = torch.max(logits, dim=1) 
                results.append(pred)
                pbar.update(1)
            
            if self.network == 'RotCAtt_TrasnUNet_plusplus':
                for s in range(len(a_weights)):
                    instance = self.c_weights[f'size_{s+1}']
                    save_vol(np.array(instance), 'images', self.vis_path + f'c_size_{s+1}.nii.gz')
                    for h in range(self.num_heads):
                        instance = self.a_weights[f'size_{s+1}'][f'head_{h+1}']
                        save_vol(np.array(instance), 'images', self.vis_path + f'a_size_{s+1}_head_{h+1}.nii.gz')    
            
            
        end_t = time()
        print(f'Prediction time: {end_t-start_t}')
        
        final_output = torch.cat(results, dim=0).detach().cpu().numpy()
        save_vol(final_output, 'labels', self.pred_path)
        
    def encoding(self, vol):
        encoded_vol = np.zeros((vol.shape[0], self.num_classes, vol.shape[1], vol.shape[2]))
        for i in range(self.num_classes):
            encoded_vol[:, i, :, :] = (vol == i).astype(np.float32)
        return encoded_vol
    
    def visualize(self, image, path, cmap='gray'):
        plt.imshow(image, cmap=cmap)
        plt.axis('off')  
        plt.tight_layout() 
        plt.savefig(path, bbox_inches="tight", pad_inches=0.0)
        plt.close() 
    
    def metrics(self):
        image = sitk.GetArrayFromImage(sitk.ReadImage(self.image_path, sitk.sitkFloat32))
        label = sitk.GetArrayFromImage(sitk.ReadImage(self.label_path))
        pred = sitk.GetArrayFromImage(sitk.ReadImage(self.pred_path))
        
        if self.inference_images:
            for index in self.index_slice:
                if index > image.shape[0]: continue
                self.visualize(image[index], f'{self.viz_file}/images/{self.id:04d}_{index+1:04d}.png')
                self.visualize(label[index], f'{self.viz_file}/labels/{self.id:04d}_{index+1:04d}.png', cmap='magma')
                self.visualize(pred[index],  f'{self.viz_file}/preds/{self.id:04d}_{index+1:04d}.png', cmap='magma')
            
        encoded_label = torch.tensor(self.encoding(label))
        encoded_pred = torch.tensor(self.encoding(pred))
        
        dice = Dice(num_classes=self.num_classes, ignore_index=[0], softmax=False)
        iou = IOU(num_classes=self.num_classes, ignore_index=[0])
        hd = HD()
        
        dice_score, _, cls_dice_score, _ = dice(encoded_pred, encoded_label)
        iou_score, cls_iou = iou(encoded_pred, encoded_label)
        hausdorff = hd(encoded_pred, encoded_label)
        
        fieldnames = ['Dice Score', 'IoU Score', 'HausDorff Distance']
        if not os.path.exists(self.infer_log_file): write_csv(self.infer_log_file, fieldnames)
        
        write_csv(self.infer_log_file, [dice_score.item(), iou_score.item(), hausdorff])
        write_csv(self.dice_class_file, cls_dice_score)
        write_csv(self.iou_class_file, cls_iou)

    
if __name__ == '__main__':
    config = parse_args()
    infer = Inference(config, 6)
    # infer.volume_convert()
    infer.prediction()
    infer.metrics()
    