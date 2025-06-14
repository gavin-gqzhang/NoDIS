from .coco_eval import do_coco_evaluation


def coco_evaluation(
    cfg,
    dataset,
    predictions,
    output_folder,
    logger,
    box_only,
    iou_types,
    expected_results,
    expected_results_sigma_tol,
    **kwargs
):
    return do_coco_evaluation(
        dataset=dataset,
        predictions=predictions,
        box_only=box_only,
        output_folder=output_folder,
        logger=logger,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
        **kwargs
    )
