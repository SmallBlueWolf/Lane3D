import os
import torch
import tqdm
import torch.nn.functional as F
from torch.nn import DataParallel as MMDataParallel
import json
import mmengine
from data_tools.vis_openlane import LaneVis


dp_factory = {'cuda': MMDataParallel, 'cpu': MMDataParallel}

def build_dp(model, device='cuda', dim=0, *args, **kwargs):
    """build DataParallel module by device type.

    if device is cuda, return a MMDataParallel module; if device is mlu,
    return a MLUDataParallel module.

    Args:
        model (:class:`nn.Module`): module to be parallelized.
        device (str): device type, cuda, cpu or mlu. Defaults to cuda.
        dim (int): Dimension used to scatter the data. Defaults to 0.

    Returns:
        :class:`nn.Module`: parallelized module.
    """
    if device == 'cuda':
        model = model.cuda()

    return dp_factory[device](model, dim=dim, *args, **kwargs)

def postprocess(output, anchor_len=10):
    proposals = output[0]
    logits = F.softmax(proposals[:, 5 + 3 * anchor_len:], dim=1)
    score = 1 - logits[:, 0]  # [N]
    proposals[:, 5 + 3 * anchor_len:] = logits  # [N, 2]
    proposals[:, 1] = score
    results = {'proposals_list': proposals.cpu().numpy()}
    return results   # [1, 7, 16]

def inference_openlane(
    model,
    data_loader,
    out_dir
):
    model.eval()
    results = []
    dataset = data_loader.dataset
    loader_indices = data_loader.batch_sampler
    pred_file = os.path.join(out_dir, 'lane3d_prediction.json')
    for batch_indices, data in tqdm.tqdm(zip(loader_indices, data_loader)):
        with torch.no_grad():
            outputs = model(return_loss=False, **data)
            for output in outputs['proposals_list']:
                result = postprocess(output, anchor_len=dataset.anchor_len)
                results.append(result)
    dataset.format_results(results, pred_file)
    
def evaluation(data_loader, out_dir, prob_th, eval_file):
    dataset = data_loader.dataset
    pred_file = os.path.join(out_dir, 'lane3d_prediction.json')
    # print("evaluating results...")
    # test_result = dataset.eval(pred_file, eval_file,  prob_th=prob_th)
    # print("===> Evaluation on validation set: \n"   
    #         "laneline F-measure {:.4} \n"
    #         "laneline Recall  {:.4} \n"
    #         "laneline Precision  {:.4} \n"
    #         "laneline Category Accuracy  {:.4} \n"
    #         "laneline x error (close)  {:.4} m\n"
    #         "laneline x error (far)  {:.4} m\n"
    #         "laneline z error (close)  {:.4} m\n"
    #         "laneline z error (far)  {:.4} m\n".format(test_result['F_score'], test_result['recall'],
    #                                                         test_result['precision'], test_result['cate_acc'],
    #                                                         test_result['x_error_close'], test_result['x_error_far'],
    #                                                         test_result['z_error_close'], test_result['z_error_far']))
    # print("save test result to", os.path.join(out_dir, 'evaluation_result.json'))
    # with open(os.path.join(out_dir, 'evaluation_result.json'), 'w') as f:
    #     json.dump(test_result, f)
        
    # visualizing
    save_dir = os.path.join(out_dir, 'vis')
    mmengine.mkdir_or_exist(save_dir)
    print("visualizing results at", save_dir)
    visualizer = LaneVis(dataset)
    visualizer.visualize(pred_file, gt_file = eval_file, img_dir = dataset.data_root, test_file=dataset.test_list, 
                        save_dir = save_dir, prob_th=prob_th)