import numpy as np
import torch
import SimpleITK as sitk
import nibabel as nib

def save_vol(vol, path):
    vol = np.transpose(vol, (2, 1, 0))
    affine = np.eye(4)
    nifti_file = nib.Nifti1Image(vol.astype(np.int8), affine)
    nib.save(nifti_file, path)    

def reconstruction_nii(image_path, model_path):
    model = torch.load(model_path)
    image = sitk.GetArrayFromImage(sitk.ReadImage(image_path, sitk.sitkFloat32))
    model.eval()
    segment_size = 12
    results = []
    with torch.no_grad():
        for index in range(image.shape[0]):
            input = image[index:index + segment_size,:,:].unsqueeze(1).cuda()
            output = model(input)
            pred = torch.argmax(output, dim=1)
            results.append(pred)   
    final_output = torch.cat(results, dim=0).detach().cpu().numpy()
    save_vol(final_output, 'temp.nii.gz')
        
    
    
    