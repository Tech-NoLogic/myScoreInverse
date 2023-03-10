import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

image = None

# source: https://pytorch.org/vision/stable/auto_examples/plot_transforms.html#sphx-glr-auto-examples-plot-transforms-py
def plot(imgs, with_orig=False, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(figsize=(200,200), nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        row = [image] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]

            # # modified to show img
            # arr = np.asarray(img)
            # if img.shape[1] == 1:
            #     arr = np.squeeze(arr, (0, 1))

            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()
    

def extract(vec, t, x_shape):
    batch_size = t.shape[0] 
    # t: [b,]
    out = vec.gather(-1, t.cpu()) 
    # vec: [timesteps,] -> out: [b,] (use t as the index)
    
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
    # out: [b,] -> [b, *((1,)*(4-1))] -> [b, *(1, 1, 1)] -> [b, 1, 1, 1] -> [b, 1, 1, 1].to(cuda)
    
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)