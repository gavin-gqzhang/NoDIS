import os.path
import random
import torch
import json
import h5py
import numpy as np
from matplotlib.pyplot import imshow
from PIL import Image, ImageDraw


def get_info_by_idx(img_idx, det_input, thres=0.5):
    groundtruth = det_input['groundtruths'][img_idx]
    prediction = det_input['predictions'][img_idx]
    # image path
    img_path = prediction.get_field('file_name')
    # boxes
    boxes = groundtruth.bbox
    # object labels
    idx2label = vocab_file['idx_to_label']
    
    labels,pred_labels=[],[]
    for idx, i in enumerate(groundtruth.get_field('labels').tolist()):
        labels.append(f'{idx}-{idx2label[str(i)]}')
    for idx, i in enumerate(prediction.get_field('pred_labels').tolist()):
        pred_labels.append(f'{idx}-{idx2label[str(int(i))]}')
    
    # groundtruth relation triplet
    idx2pred = vocab_file['idx_to_predicate']
    gt_rels = groundtruth.get_field('relation_tuple').tolist()
    gt_rels = [(labels[i[0]], idx2pred[str(i[2])], labels[i[1]]) for i in gt_rels]
    # prediction relation triplet
    pred_rel_pair = prediction.get_field('rel_pair_idxs').tolist()
    pred_rel_label = prediction.get_field('pred_rel_scores')
    pred_rel_label[:,0] = 0
    pred_rel_score, pred_rel_label = pred_rel_label.max(-1)
    #mask = pred_rel_score > thres
    #pred_rel_score = pred_rel_score[mask]
    #pred_rel_label = pred_rel_label[mask]
    pred_rels = [(pred_labels[int(i[0])], idx2pred[str(j)], pred_labels[int(i[1])]) for i, j in zip(pred_rel_pair, pred_rel_label.tolist())]
    return img_idx,img_path, boxes, labels, gt_rels, pred_rels, pred_rel_score, pred_rel_label

def draw_single_box(pic, box, color='red', draw_info=None):
    draw = ImageDraw.Draw(pic)
    x1,y1,x2,y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    draw.rectangle(((x1, y1), (x2, y2)), outline=color)
    if draw_info:
        draw.rectangle(((x1, y1), (x1+50, y1+10)), fill=color)
        info = draw_info
        draw.text((x1, y1), info)

def print_list(name, input_list):
    for i, item in enumerate(input_list):
        print(name + ' ' + str(i) + ': ' + str(item))
        
def write_log(idx,name,input_list):
    if not os.path.exists('visualize_relation'):
        os.makedirs('visualize_relation',exist_ok=True)
    with open(f'visualize_relation/{idx}.log','a') as log_f:
        log_f.write('*'*50)
        for i, item in enumerate(input_list):
            log_f.write('\n')
            log_f.write(name + ' ' + str(i) + ': ' + str(item))
        log_f.write('\n')
        
def draw_image(idx,img_path, boxes, labels, gt_rels, pred_rels, pred_rel_score, pred_rel_label, print_img=True):
    pic = Image.open(img_path)
    num_obj = boxes.shape[0]
    for i in range(num_obj):
        info = labels[i]
        draw_single_box(pic, boxes[i], draw_info=info)
    if os.path.exists(f'visualize_relation/{idx}.log'):
        os.remove(f'visualize_relation/{idx}.log')
    write_log(idx,'gt_boxes', labels)
    write_log(idx,'gt_rels', gt_rels)
    write_log(idx,'pred_rels', pred_rels)
    
    return pic

def show_selected(idx_list):
    for select_idx in idx_list:
        print(select_idx)
        pic=draw_image(*get_info_by_idx(select_idx, detected_origin_result))
        
        if not os.path.exists('visualize_relation'):
            os.makedirs('visualize_relation',exist_ok=True)
        pic.save(f'visualize_relation/{select_idx}.png')
        
def show_all(start_idx, length):
    for cand_idx in range(start_idx, start_idx+length):
        pic=draw_image(*get_info_by_idx(cand_idx, detected_origin_result))
        
        if not os.path.exists('visualize_relation'):
            os.makedirs('visualize_relation',exist_ok=True)
        pic.save(f'visualize_relation/{cand_idx}.png')

image_file = json.load(open('/data/sdc/SGG_data/VG/image_data.json'))
vocab_file = json.load(open('/data/sdc/SGG_data/VG/VG-SGG-dicts.json'))
data_file = h5py.File('/data/sdc/SGG_data/VG/VG-SGG-with-attri.h5', 'r')
# remove invalid image
corrupted_ims = [1592, 1722, 4616, 4617]
tmp = []
for item in image_file:
    if int(item['image_id']) not in corrupted_ims:
        tmp.append(item)
image_file = tmp

# load detected results
detected_origin_path = '/data/sdc/checkpoints/SGG_Benchmark/VG/Multi_step_Denoise/PENetPredictor_sgcls_wo_bias_step1/inference/VG_stanford_filtered_with_attribute_test/'
detected_origin_result = torch.load(detected_origin_path + 'eval_results.pytorch')
detected_info = json.load(open(detected_origin_path + 'visual_info.json'))

show_all(start_idx=random.choice(range(len(detected_origin_result['groundtruths']))), length=5)