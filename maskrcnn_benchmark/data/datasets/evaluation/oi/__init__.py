import copy
import os

import numpy as np

from .oi_evaluation import eval_rel_results, eval_entites_detection
from ..vg.vg_eval import save_output,evaluate_relation_of_one_image
from ..vg.sgg_eval import *


def oi_evaluation(
        cfg,
        dataset,
        predictions,
        output_folder,
        logger,
        iou_types,
        **kwargs
):
    if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
        if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            mode = 'predcls'
        else:
            mode = 'sgcls'
    else:
        mode = 'sgdet'

    result_str = '\n' + '=' * 100 + '\n'

    result_dict_list_to_log = []

    predicate_cls_list = dataset.ind_to_predicates

    groundtruths = dict()
    # resize predition to same scale with the images
    for image_id, prediction in predictions.items():
        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        # recover original size which is before transform
        predictions[image_id] = prediction.resize((image_width, image_height))
        gt = dataset.get_groundtruth(image_id, evaluation=True)
        groundtruths[image_id]=gt

    save_output(output_folder, groundtruths, predictions, dataset)

    # eval detection by coco style eval
    if "bbox" in iou_types:
        (mAp,result_dict_list_to_log,result_str) = eval_entites_detection(mode, groundtruths, dataset, predictions,
                                              result_dict_list_to_log, result_str, logger)

        if not cfg.MODEL.RELATION_ON:
            logger.info(result_str)
            return mAp, result_dict_list_to_log

    # result_str_tmp = ''
    # result_str_tmp, \
    # result_dict_list_to_log = eval_classic_recall(mode, groundtruths, predictions, predicate_cls_list,
    #                                               logger, result_str_tmp, result_dict_list_to_log)
    # result_str += result_str_tmp
    # logger.info(result_str_tmp)
    if "relations" in iou_types:
        result_str+=routine_sgg_eval(cfg=cfg,mode=mode,dataset=dataset,groundtruths=groundtruths,predictions=predictions,logger=logger)
        result_str+='=' * 100 + '\n'

    # transform the initial prediction into oi predition format
    packed_results = adapt_results(groundtruths, predictions)

    result_str, result_dict = eval_rel_results(
        packed_results, predicate_cls_list, result_str, logger,
    )
    result_dict_list_to_log.append(result_dict)

    result_str += '=' * 100 + '\n'
    logger.info(result_str)

    # if output_folder:
    #     with open(os.path.join(output_folder, "evaluation_res.txt"), 'w') as f:
    #         f.write(result_str)

    return float(result_dict['w_final_score'])


def routine_sgg_eval(cfg,mode, dataset,groundtruths,predictions,logger):
    # ******************************** Init Parameters ********************************
    # get zeroshot triplet
    if cfg.SOLVER.ZEROSHOT_MODE:
        if cfg.SOLVER.ZEROSHOT_MODE=='Seen':
            zeroshot_load_path='maskrcnn_benchmark/data/datasets/evaluation/vg/zeroshot_triplet_seen.pytorch'
        else:
            zeroshot_load_path='maskrcnn_benchmark/data/datasets/evaluation/vg/zeroshot_triplet_unseen.pytorch'
    else:
        zeroshot_load_path='maskrcnn_benchmark/data/datasets/evaluation/vg/zeroshot_triplet.pytorch'
        
    zeroshot_triplet = torch.load(zeroshot_load_path, map_location=torch.device("cpu")).long().numpy()
    logger.info(f'Load zeroshot triplet from: {zeroshot_load_path}')
    
    attribute_on = cfg.MODEL.ATTRIBUTE_ON
    num_attributes = cfg.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
    
    num_rel_category = cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
    multiple_preds = cfg.TEST.RELATION.MULTIPLE_PREDS
    iou_thres = cfg.TEST.RELATION.IOU_THRESHOLD
    
    result_str = ''
    
    # ******************************** Calculate Evaluation Metrics ********************************
    
    result_dict = {}
    evaluator = {}
    # tradictional Recall@K
    eval_recall = SGRecall(result_dict)
    eval_recall.register_container(mode)
    evaluator['eval_recall'] = eval_recall

    # no graphical constraint
    eval_nog_recall = SGNoGraphConstraintRecall(result_dict)
    eval_nog_recall.register_container(mode)
    evaluator['eval_nog_recall'] = eval_nog_recall

    # test on different distribution
    eval_zeroshot_recall = SGZeroShotRecall(result_dict)
    eval_zeroshot_recall.register_container(mode)
    evaluator['eval_zeroshot_recall'] = eval_zeroshot_recall

    # test on no graph constraint zero-shot recall
    eval_ng_zeroshot_recall = SGNGZeroShotRecall(result_dict)
    eval_ng_zeroshot_recall.register_container(mode)
    evaluator['eval_ng_zeroshot_recall'] = eval_ng_zeroshot_recall
    
    # used by https://github.com/NVIDIA/ContrastiveLosses4VRD for sgcls and predcls
    eval_pair_accuracy = SGPairAccuracy(result_dict)
    eval_pair_accuracy.register_container(mode)
    evaluator['eval_pair_accuracy'] = eval_pair_accuracy

    # used for meanRecall@K
    eval_mean_recall = SGMeanRecall(result_dict, num_rel_category, dataset.ind_to_predicates, print_detail=True)
    eval_mean_recall.register_container(mode)
    evaluator['eval_mean_recall'] = eval_mean_recall

    # used for no graph constraint mean Recall@K
    eval_ng_mean_recall = SGNGMeanRecall(result_dict, num_rel_category, dataset.ind_to_predicates, print_detail=True)
    eval_ng_mean_recall.register_container(mode)
    evaluator['eval_ng_mean_recall'] = eval_ng_mean_recall

    # prepare all inputs
    global_container = {}
    global_container['zeroshot_triplet'] = zeroshot_triplet
    global_container['result_dict'] = result_dict
    global_container['mode'] = mode
    global_container['multiple_preds'] = multiple_preds
    global_container['num_rel_category'] = num_rel_category
    global_container['iou_thres'] = iou_thres
    global_container['attribute_on'] = attribute_on
    global_container['num_attributes'] = num_attributes
    
    for idx,(groundtruth, prediction) in enumerate(zip(groundtruths.values(), predictions.values())):
        evaluate_relation_of_one_image(groundtruth, prediction, global_container, evaluator)
        
    # calculate mean recall
    eval_mean_recall.calculate_mean_recall(mode)
    eval_ng_mean_recall.calculate_mean_recall(mode)
    
    # print result
    result_str += eval_recall.generate_print_string(mode)
    result_str += eval_nog_recall.generate_print_string(mode)
    result_str += eval_zeroshot_recall.generate_print_string(mode)
    result_str += eval_ng_zeroshot_recall.generate_print_string(mode)
    result_str += eval_mean_recall.generate_print_string(mode)
    result_str += eval_ng_mean_recall.generate_print_string(mode)
    
    if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
        result_str += eval_pair_accuracy.generate_print_string(mode)
    
    result_str += 'SGG eval: '
    for k in [20,50,100]:
        recall = float(np.mean(result_dict[mode + '_recall'][k]))
        mrecall = float(result_dict[mode + '_mean_recall'][k])
        result_str += '    F @ %d: %.4f; ' % (k, 2/(1/recall+1/mrecall))
        
    result_str += ' for mode=%s, type=Harmonic average of R@K and mR@K.' % (mode)
    result_str += '\n'
    
    return result_str

def adapt_results(
        groudtruths, predictions,
):
    packed_results = []
    for gt, pred in zip(groudtruths.values(), predictions.values()):
        gt = copy.deepcopy(gt)
        pred = copy.deepcopy(pred)

        pred_boxlist = pred.convert('xyxy').to("cpu")
        pred_ent_scores = pred_boxlist.get_field('pred_scores').detach().cpu()
        pred_ent_labels = pred_boxlist.get_field('pred_labels').long().detach().cpu()
        pred_ent_labels = pred_ent_labels - 1  # remove the background class

        pred_rel_pairs = pred_boxlist.get_field('rel_pair_idxs').long().detach().cpu()  # N * R * 2
        pred_rel_scores = pred_boxlist.get_field('pred_rel_scores').detach().cpu()  # N * C

        sbj_boxes = pred_boxlist.bbox[pred_rel_pairs[:, 0], :].numpy()
        sbj_labels = pred_ent_labels[pred_rel_pairs[:, 0]].numpy()
        sbj_scores = pred_ent_scores[pred_rel_pairs[:, 0]].numpy()

        obj_boxes = pred_boxlist.bbox[pred_rel_pairs[:, 1], :].numpy()
        obj_labels = pred_ent_labels[pred_rel_pairs[:, 1]].numpy()
        obj_scores = pred_ent_scores[pred_rel_pairs[:, 1]].numpy()

        prd_scores = pred_rel_scores

        gt_boxlist = gt.convert('xyxy').to("cpu")
        gt_ent_labels = gt_boxlist.get_field('labels')
        gt_ent_labels = gt_ent_labels - 1

        gt_rel_tuple = gt_boxlist.get_field('relation_tuple').long().detach().cpu()
        sbj_gt_boxes = gt_boxlist.bbox[gt_rel_tuple[:, 0], :].detach().cpu().numpy()
        obj_gt_boxes = gt_boxlist.bbox[gt_rel_tuple[:, 1], :].detach().cpu().numpy()
        sbj_gt_classes = gt_ent_labels[gt_rel_tuple[:, 0]].long().detach().cpu().numpy()
        obj_gt_classes = gt_ent_labels[gt_rel_tuple[:, 1]].long().detach().cpu().numpy()
        prd_gt_classes = gt_rel_tuple[:, -1].long().detach().cpu().numpy()
        prd_gt_classes = prd_gt_classes - 1

        return_dict = dict(sbj_boxes=sbj_boxes,
                           sbj_labels=sbj_labels.astype(np.int32, copy=False),
                           sbj_scores=sbj_scores,
                           obj_boxes=obj_boxes,
                           obj_labels=obj_labels.astype(np.int32, copy=False),
                           obj_scores=obj_scores,
                           prd_scores=prd_scores,
                           # prd_scores_bias=prd_scores,
                           # prd_scores_spt=prd_scores,
                           # prd_ttl_scores=prd_scores,
                           gt_sbj_boxes=sbj_gt_boxes,
                           gt_obj_boxes=obj_gt_boxes,
                           gt_sbj_labels=sbj_gt_classes.astype(np.int32, copy=False),
                           gt_obj_labels=obj_gt_classes.astype(np.int32, copy=False),
                           gt_prd_labels=prd_gt_classes.astype(np.int32, copy=False))

        packed_results.append(return_dict)

    return packed_results