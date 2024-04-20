import cv2
import torch as tc

def pad_predict(model, images, divis):
    preds = []
    for idx in range(images.shape[0]):
        h,w = images[idx].shape[1:]
        pad_b = (divis -  h % divis) % divis
        pad_r = (divis -  w % divis) % divis
        padded_image = cv2.copyMakeBorder(images[idx].squeeze().cpu().numpy(), top=0, left=0, bottom=pad_b, right=pad_r, borderType=cv2.BORDER_CONSTANT, value=0)
        padded_image = tc.from_numpy(padded_image)[(None,)*2]
        with tc.no_grad():
            pred = model(padded_image.cuda())
        
        pred = pred[..., 0 : h, 0 : w]
        preds.append(pred)
        
    preds = tc.cat(preds, dim=0)
    return preds

def sliding_window(model, patch_size, stride, image, threshold, device):
    h, w = image.shape[1:]
    image_shape = (h,w)
    p_h, p_w = patch_size[0], patch_size[1]
    k_h = next(k for k in range(0, 100000) if (k >= (h - p_h)/stride))
    k_w = next(k for k in range(0, 100000) if (k >= (w - p_w)/stride))
    pad_b = k_h * stride + p_h - h
    pad_r = k_w * stride + p_w - w
    
    info = {'image height' : h, 'image width' : w, 'patch height' : p_h, 'patch width' : p_w, 'right padding' : pad_r, 'bottom padding' : pad_b}
    padded_image = cv2.copyMakeBorder(image.squeeze().cpu().numpy(), top = 0, left = 0, bottom = pad_b, right = pad_r, borderType = cv2.BORDER_CONSTANT, value = 0)
    h, w = padded_image.shape
    padded_shape = (h,w)
    patches = []
    for height in range(0, h - p_h  + 1, stride):
        for width in range(0, w - p_w + 1, stride):
            patch = tc.from_numpy(padded_image[height : height + p_h, width : width + p_w]).unsqueeze(0).to(device)
            with tc.no_grad():
                patch_pred = model(patch.unsqueeze(0))
                patch_pred = tc.sigmoid(patch_pred)
                patch_pred = (patch_pred>threshold).float()
            patches.append((patch_pred, height, width))
    info['padded height'], info['padded width'], info['patches lenth'] = h, w, len(patches)
    return patches, info, image_shape, padded_shape

def reconstruct_img(image_shape, padded_shape, patches, patch_size):
    h, w = padded_shape[0], padded_shape[1]
    reconstruct_img = tc.zeros(1,h,w)
    for patch in patches:
        reconstruct_img[:, patch[1] : patch[1] + patch_size[0], patch[2] : patch[2] + patch_size[1]] = patch[0].squeeze()
    reconstruct_img = reconstruct_img[:, 0 : image_shape[0], 0 : image_shape[1]]
    return reconstruct_img