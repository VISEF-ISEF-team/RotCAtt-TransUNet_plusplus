import torch
from networks.RotCAtt_TransUNet_plusplus_gradcam import RotCAtt_TransUNet_plusplus_GradCam
import nibabel as nib
import numpy as np
import cv2
import matplotlib.pyplot as plt


def normalize_image_intensity_range(img):
    HOUNSFIELD_MAX = np.max(img)
    HOUNSFIELD_MIN = np.min(img)
    HOUNSFIELD_RANGE = HOUNSFIELD_MAX - HOUNSFIELD_MIN

    img[img < HOUNSFIELD_MIN] = HOUNSFIELD_MIN
    img[img > HOUNSFIELD_MAX] = HOUNSFIELD_MAX

    return (img - HOUNSFIELD_MIN) / HOUNSFIELD_RANGE


def seg_grad_cam_to_volumetric(grad_model, volume_image_path, batch_size, class_instance=0):
    volume_image = nib.load(volume_image_path).get_fdata()
    n_samples = volume_image.shape[-1]

    residual = n_samples % batch_size
    if 1 <= residual and residual < 3:
        batch_size -= 1

    # grad_cam output
    grad_cam_output = torch.ones((256, 256, n_samples, 3))

    # convert to 2d slices
    for i in range(0, n_samples, batch_size):

        if i + batch_size >= n_samples:
            num_slice = n_samples - batch_size - 1
        else:
            num_slice = batch_size

        new_output = np.ones((256, 256, num_slice), dtype=np.float32)
        slice_list = volume_image[:, :, i:i + num_slice]

        for j in range(num_slice):
            slice = slice_list[:, :, j]
            slice = cv2.resize(slice, (256, 256))

            slice = normalize_image_intensity_range(slice)

            new_output[:, :, i] = slice

        # process volume image
        new_output = torch.from_numpy(new_output)
        new_output = new_output.permute(2, 0, 1)
        new_output = torch.unsqueeze(new_output, dim=1)
        new_output = new_output.to(device="cuda")

        # prediction
        y_pred = grad_model(new_output)
        activations = grad_model.get_activations(new_output).detach()

        # calculate score
        for x in range(num_slice):
            class_output = y_pred[x, class_instance]
            class_score_sum = class_output.sum()
            class_score_sum.backward(retain_graph=True)

            gradients = grad_model.get_activations_gradient()
            pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

            print(f"Pooled Gradient: {pooled_gradients.shape}")

            instance_activation = activations[x]

            for channel in range(64):
                instance_activation[channel, :, :] *= pooled_gradients[channel]

            print(f"Activations: {instance_activation.shape}")

            heatmap = torch.mean(instance_activation, dim=0)
            print(f"Heatmap shape: {heatmap.shape}")

            heatmap = np.maximum(heatmap, 0)
            heatmap /= torch.max(heatmap)
            heatmap = heatmap.numpy()

            heatmap = cv2.resize(heatmap, (256, 256))
            heatmap = np.uint8(255 * heatmap)

            grad_cam_output[:, :, i + x]

    nifti_image = nib.nifti1.Nifti1Image(
        grad_cam_output, affine=np.eye(4), dtype=np.int32)
    nib.save(
        nifti_image, f"./outputs/GradCam/Imagechd_RotCAtt_TransUNet_plusplus_256/grad_cam_instance-{class_instance}.nii")


def seg_grad_cam_to_volumetric_visualize(grad_model, volume_image_path, start_index, num_slice, class_instance=0):
    volume_image = nib.load(volume_image_path).get_fdata()

    # convert to 2d slices
    new_output = np.ones((256, 256, num_slice), dtype=np.float32)
    slice_list = volume_image[:, :, start_index:start_index + num_slice]

    fig, axes = plt.subplots(nrows=2, ncols=num_slice, figsize=(16, 16))

    for j in range(num_slice):
        slice = slice_list[:, :, j]
        slice = cv2.resize(slice, (256, 256))

        slice = normalize_image_intensity_range(slice)

        new_output[:, :, j] = slice

    # process volume image
    new_output = torch.from_numpy(new_output)
    new_output = new_output.permute(2, 0, 1)
    new_output = torch.unsqueeze(new_output, dim=1)
    new_output = new_output.to(device="cuda")

    # prediction
    y_pred = grad_model(new_output)
    activations = grad_model.get_activations(new_output).detach()

    # calculate score
    for x in range(num_slice):
        class_output = y_pred[x, class_instance]
        class_score_sum = class_output.sum()
        class_score_sum.backward(retain_graph=True)

        gradients = grad_model.get_activations_gradient()
        print(f"Length gradients: {len(gradients)}")
        gradients = gradients[-1]
        grad_model.clear_activations_gradient()
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        print(f"Pooled Gradient: {pooled_gradients.shape}")

        instance_activation = activations[x]

        for channel in range(64):
            instance_activation[channel, :, :] *= pooled_gradients[channel]

        print(f"Activations: {instance_activation.shape}")

        heatmap = torch.mean(instance_activation, dim=0)
        print(f"Heatmap shape: {heatmap.shape}")

        heatmap /= torch.max(heatmap)
        heatmap = heatmap.cpu().numpy()
        heatmap = np.maximum(heatmap, 0)

        heatmap = cv2.resize(heatmap, (256, 256))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        image0 = torch.squeeze(new_output[x, :, :, :])
        image0 = image0.cpu().numpy()
        image0 = cv2.cvtColor(image0, cv2.COLOR_GRAY2BGR)

        superimposed_img = (heatmap / 255.0) * 0.6 + image0

        heatmap_plot = axes[0][x].imshow(superimposed_img, cmap='RdBu')
        axes[0][x].set_xlabel(f"Slice: {start_index + x}")
        fig.colorbar(heatmap_plot, ax=axes[0][x], fraction=0.046)

        superimposed_plot = axes[1][x].imshow(heatmap, cmap='RdBu')
        axes[1][x].set_xlabel(f"Heatmap of slice: {start_index + x}")
        fig.colorbar(superimposed_plot,
                     ax=axes[1][x], fraction=0.046)

    plt.suptitle(
        f"Model attention on slices {start_index} - {start_index + num_slice} with class instance {class_instance}", fontsize=16)
    plt.subplots_adjust(wspace=0.5)
    plt.show()


if __name__ == "__main__":
    volume_image_path = "D:\\Hy\\images\\image_0001.nii.gz"
    model = torch.load("C:\\Users\\Acer\\Downloads\\model.pth")
    grad_model = RotCAtt_TransUNet_plusplus_GradCam(model=model)

    start_index = 135
    num_slice = 3
    class_instance = 2

    seg_grad_cam_to_volumetric_visualize(grad_model, volume_image_path=volume_image_path,
                                         start_index=start_index, num_slice=num_slice, class_instance=class_instance)
