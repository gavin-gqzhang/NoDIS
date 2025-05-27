import json
import random
import colorsys
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import patches, lines
import numpy as np
import torch
import cv2
from maskrcnn_benchmark.data.datasets.visual_genome import load_graphs, load_image_filenames, load_info
BOX_SCALE = 1024  # Scale at which we have the boxes

# based on https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/visualize.py
def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def display_instances(image, boxes,box_labels,
                      figsize=(16, 16)):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    """
    # Number of instances
    N = len(boxes)
    if not N:
        print("\n*** No instances to display *** \n")

    _, ax = plt.subplots(1, figsize=figsize)

    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    
    for i in range(N):
        color = colors[i]

        # Bounding box
        x1, y1, x2, y2 = boxes[i][:4]
        w, h = x2 - x1, y2 - y1
        p = patches.Rectangle((x1, y1), w, h, linewidth=2,
                            alpha=0.7, linestyle="dashed",
                            edgecolor=color, facecolor='none')
        ax.add_patch(p)
        cv2.imwrite(f"{box_labels[i]}.png",image[int(y1):int(y2),int(x1):int(x2),...])

        # label = box_labels[i]
        # ax.text(x1, y1 + 8, label, color='w', size=15, backgroundcolor="none")

    ax.imshow(image)
    
    plt.savefig('vis_box.png', bbox_inches='tight',dpi=300)

if __name__=='__main__':
    """
    n_samples,n_classes = 100,51

    weights = torch.rand(n_samples, n_classes)
    weights = weights / weights.sum(dim=1, keepdim=True)

    # 转换成numpy数组
    weights_np = weights.numpy()
    
    plt.figure(figsize=(10, 6))
    plt.imshow(weights_np, cmap='plasma', aspect='auto')
    plt.xticks([])
    plt.yticks([])
    plt.savefig('rep_attn_cen.png',bbox_inches='tight',dpi=300)
    
    weights = torch.rand(n_classes,n_samples )
    weights = weights / weights.sum(dim=1, keepdim=True)

    # 转换成numpy数组
    weights_np = weights.numpy()
    
    plt.figure(figsize=(10, 6))
    plt.imshow(weights_np, cmap='plasma', aspect='auto')
    plt.xticks([])
    plt.yticks([])
    plt.savefig('cen_attn_rep.png',bbox_inches='tight',dpi=300)
    
    # print(weights.argmax(1))
    # labels=torch.zeros((n_samples,n_classes))
    # labels[torch.arange(n_samples),weights.argmax(1)]=1
    # plt.figure(figsize=(10, 6))
    # plt.imshow(labels, cmap='binary', aspect='auto')
    # plt.xticks([])
    # plt.yticks([])
    # plt.savefig('onehot.png',bbox_inches='tight',dpi=300)
    
    raise
    """
    dict_files,img_dir,image_file,roidb_file,split,num_im,num_val_im="/data/sdc/SGG_data/VG/VG-SGG-dicts-with-attri.json","/data/sdc/SGG_data/VG/VG_100K","/data/sdc/SGG_data/VG/image_data.json","/data/sdc/SGG_data/VG/VG-SGG-with-attri.h5",'train',-1,5000
    ind_to_classes, ind_to_predicates, ind_to_attributes = load_info(dict_files) # contiguous 151, 51 containing __background__
    categories = {i : ind_to_classes[i] for i in range(len(ind_to_classes))}

    split_mask, gt_boxes, gt_classes, gt_attributes, relationships = load_graphs(
            roidb_file, split, num_im, num_val_im=num_val_im,
            filter_empty_rels=True,
            filter_non_overlap=True,
        )

    filenames, img_info = load_image_filenames(img_dir, image_file) # length equals to split_mask
    filenames = [filenames[i] for i in np.where(split_mask)[0]]
    img_info = [img_info[i] for i in np.where(split_mask)[0]]

    img_file='/data/sdc/SGG_data/VG/VG_100K/2369103.jpg'
    img_idx=filenames.index(img_file)
    # img_idx=random.randint(0,len(filenames))
    # img_file=filenames[img_idx]
    # print(filenames[img_idx])
    info=img_info[img_idx]
    
    relation=relationships[img_idx]
    box_labels=[f'{ind_to_classes[cls_id]}_{i}' for i,cls_id in enumerate(gt_classes[img_idx])]
    for (o0, o1, r) in relation:
        print(f'{box_labels[o0]}-{ind_to_predicates[r]}-{box_labels[o1]}')
    
    w, h = info['width'], info['height']
    box = gt_boxes[img_idx] / BOX_SCALE * max(w, h)
    box = np.reshape(box,(-1, 4))  # guard against no boxes

    image = Image.open(img_file).convert("RGB")
    image = np.asarray(image, np.uint8)
    display_instances(image, box,box_labels)
