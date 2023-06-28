import copy

# import ensemble_boxes
import numpy as np
import torch
import torchvision
from PIL import Image
import cv2



def normalize_boxes(boxes, img_size, invert=False):
    if invert:
        boxes[..., 0] = boxes[..., 0] * img_size[0]
        boxes[..., 1] = boxes[..., 1] * img_size[1]
        boxes[..., 2] = boxes[..., 2] * img_size[0]
        boxes[..., 3] = boxes[..., 3] * img_size[1]
    else:
        boxes[..., 0] = boxes[..., 0] / img_size[0]
        boxes[..., 1] = boxes[..., 1] / img_size[1]
        boxes[..., 2] = boxes[..., 2] / img_size[0]
        boxes[..., 3] = boxes[..., 3] / img_size[1]
    return boxes



def bounding_boxes(v_boxes, v_labels, v_scores, log_width, log_height, class_id_to_label, score_threshold):
    all_boxes = []
    # plot each bounding box for this image
    for b_i, box in enumerate(v_boxes):

        if v_scores[b_i] < score_threshold or int(v_labels[b_i]) == -1:  # stop when below this threshold, scores in descending order
            break

        caption = "%s" % (class_id_to_label[int(v_labels[b_i])])
        if v_scores[b_i] <= 1:
            caption = "%s (%.3f)" % (class_id_to_label[int(v_labels[b_i])], v_scores[b_i])
        # from xyxy
        box_data = {"position" : {
            "minX" : int(box[0]),
            "minY" : int(box[1]),
            "maxX" : int(box[2]),
            "maxY" : int(box[3])},
            "class_id" : int(v_labels[b_i]),
            # optionally caption each box with its class and score
            "box_caption" : caption,
            "domain" : "pixel",
            "scores" : { "score" : int(v_scores[b_i]*100) }}

        all_boxes.append(box_data)
    return all_boxes


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.
    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor.clamp(-1.0, 1.0).cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def visualize_detections(dataset, detections, target, wandb, args, split='val', score_threshold=0.5, img_tensor=None):
    class_id_to_label = { int(i) : str(i) for i in range(1, args.num_classes + 1)}
    class_id_to_label.update({1: "person", 2: "bicycle", 3: "car"})
    detections = detections.detach().cpu().numpy()
    img_indices = target['img_idx'].cpu().numpy()
    bboxes = target['bbox'].cpu().numpy()
    clses = target['cls'].cpu().numpy()
    scores = target['scores'].cpu().numpy() if 'scores' in target else np.zeros_like(clses) + 1000
    img_scales = target['img_scale'].cpu().numpy()
    for i, (img_idx, img_dets, bbox, cls, img_scale, score) in enumerate(zip(img_indices, detections, bboxes, clses, img_scales, scores)):
        img_id = dataset.parser.img_ids[img_idx]
        img_info = dataset.parser.img_infos[img_idx]
        # yxyx to xyxy
        if img_tensor is None:
            bbox[:, 0:4] = bbox[:, [1, 0, 3, 2]] * img_scale
            filename = dataset.thermal_data_dir/img_info['file_name']
            raw_image = Image.open(filename).convert('RGB')
        else:
            bbox[:, 0:4] = bbox[:, [1, 0, 3, 2]]
            img_dets[:, 0:4] = img_dets[:, 0:4] / img_scale
            raw_image = tensor2im(img_tensor[i])
        predicted_boxes = bounding_boxes(
            v_boxes=img_dets[:, 0:4],
            v_labels=img_dets[:, 5],
            v_scores=img_dets[:, 4],
            log_width=img_info['width'],
            log_height=img_info['height'],
            class_id_to_label=class_id_to_label,
            score_threshold=score_threshold)
        gt_boxes = bounding_boxes(
            v_boxes=bbox,
            v_labels=cls,
            v_scores=score,
            log_width=img_info['width'],
            log_height=img_info['height'],
            class_id_to_label=class_id_to_label,
            score_threshold=0)
        # log to wandb: raw image, predictions, and dictionary of class labels for each class id
        box_image = wandb.Image(raw_image, boxes = {"predictions": {"box_data": predicted_boxes, "class_labels" : class_id_to_label},
                                                    "gts": {"box_data": gt_boxes, "class_labels" : class_id_to_label}})
        wandb.log({split: box_image})


def visualize_target(dataset, target, wandb, args, split='val', img_tensor=None):
    class_id_to_label = { int(i) : str(i) for i in range(1, args.num_classes + 1)}
    class_id_to_label.update({1: "person", 2: "bicycle", 3: "car"})
    img_indices = target['img_idx'].cpu().numpy()
    bboxes = target['bbox'].cpu().numpy()
    clses = target['cls'].cpu().numpy()
    scores = target['scores'].cpu().numpy() if 'scores' in target else np.zeros_like(clses) + 1000
    img_scales = target['img_scale'].cpu().numpy()
    for i, (img_idx, bbox, cls, img_scale, score) in enumerate(zip(img_indices, bboxes, clses, img_scales, scores)):
        img_id = dataset.parser.img_ids[img_idx]
        img_info = dataset.parser.img_infos[img_idx]
        # yxyx to xyxy
        if img_tensor is None:
            bbox[:, 0:4] = bbox[:, [1, 0, 3, 2]] * img_scale
            filename = dataset.thermal_data_dir/img_info['file_name']
            raw_image = Image.open(filename).convert('RGB')
        else:
            bbox[:, 0:4] = bbox[:, [1, 0, 3, 2]]
            raw_image = tensor2im(img_tensor[i])
        gt_boxes = bounding_boxes(
            v_boxes=bbox,
            v_labels=cls,
            v_scores=score,
            log_width=img_info['width'],
            log_height=img_info['height'],
            class_id_to_label=class_id_to_label,
            score_threshold=0)
        # log to wandb: raw image, predictions, and dictionary of class labels for each class id
        box_image = wandb.Image(raw_image, boxes = {'gts': {"box_data": gt_boxes, "class_labels" : class_id_to_label}})
        wandb.log({split: box_image})
        
def load_checkpoint_selective(net, snapshot, scene=None):
    """
    Restore weights and optimizer (if needed ) for resuming job.
    """
    checkpoint = torch.load(snapshot, map_location=torch.device('cpu'))

    if 'state_dict' in checkpoint:
        net = state_restore_selective(net, checkpoint['state_dict'], scene)
    else:
        net = state_restore_selective(net, checkpoint, scene)

    return net


def state_restore_selective(net, loaded_dict, scene=None):
    """
    Handle partial loading when some tensors don't match up in size.
    Because we want to use models that were trained off a different
    number of classes.
    """
    net_state_dict = net.state_dict()
    new_loaded_dict = {}
    for k in loaded_dict:
        if 'cbam' in k:
            new_loaded_dict[k.replace('fusion', 'fusion'+str(scene))] = loaded_dict[k]
            print('successfully loaded ', k.replace('fusion', 'fusion'+str(scene)))
        if 'classifier' in k:
            new_loaded_dict[k] = loaded_dict[k]
            print('successfully loaded ', k)
    net_state_dict.update(new_loaded_dict)
    net.load_state_dict(net_state_dict)
    return net


def visualize_detections(dataset, detections, target, wandb, args, split='val', score_threshold=0.5):
    class_id_to_label = { int(i) : str(i) for i in range(args.num_classes)}
    #class_id_to_label.update({1: "person", 2: "bicycle", 3: "car"})
    #class_id_to_label.update({1: "people", 2: "car", 3: "motorcycle", 4: "bus", 5: "truck", 6: "lamp"})
    class_id_to_label.update({0: "DontCare", 1:"truck", 2: "pedestrian", 3: "car", 4: "cyclist"})
    detections = detections.detach().cpu().numpy()
    img_indices = target['img_idx'].cpu().numpy()
    bboxes = target['bbox'].cpu().numpy()
    clses = target['cls'].cpu().numpy()
    scores = target['scores'].cpu().numpy() if 'scores' in target else np.zeros_like(clses) + 1000
    img_scales = target['img_scale'].cpu().numpy()
    for img_idx, img_dets, bbox, cls, img_scale, score in zip(img_indices, detections, bboxes, clses, img_scales, scores):
        img_id = dataset.parser.img_ids[img_idx]
        img_info = dataset.parser.img_infos[img_idx]
        # yxyx to xyxy
        bbox[:, 0:4] = bbox[:, [1, 0, 3, 2]] * img_scale
        predicted_boxes = bounding_boxes(
            v_boxes=img_dets[:, 0:4],
            v_labels=img_dets[:, 5],
            v_scores=img_dets[:, 4],
            log_width=img_info['width'],
            log_height=img_info['height'],
            class_id_to_label=class_id_to_label,
            score_threshold=score_threshold)
        gt_boxes = bounding_boxes(
            v_boxes=bbox,
            v_labels=cls,
            v_scores=score,
            log_width=img_info['width'],
            log_height=img_info['height'],
            class_id_to_label=class_id_to_label,
            score_threshold=0)
        filename = dataset.gated_data_dir/img_info['file_name']
        filename_rgb = dataset.rgb_data_dir/img_info['file_name']
        raw_image = Image.open(filename).convert('RGB')
        gated_img = cv2.imread(str(filename), -1) / (2**10 - 1) * 255
        if len(gated_img.shape) < 3:
            gated_img = np.stack([gated_img]*3, axis=2)
        else:
            gated_img = gated_img[:,:,::-1]
        # raw_image_rgb = Image.open(filename_rgb).convert('RGB')
        rgb_img = cv2.imread(str(filename_rgb), -1) / (2**12 - 1) * 255
        if len(rgb_img.shape) < 3:
            rgb_img = np.stack([rgb_img]*3, axis=2)
        else:
            rgb_img = rgb_img[:,:,::-1]
        raw_image = Image.fromarray(gated_img.astype(np.uint8))
        raw_image_rgb = Image.fromarray(rgb_img.astype(np.uint8))
        # log to wandb: raw image, predictions, and dictionary of class labels for each class id
        # print(rgb[0].cpu().numpy())
        # print(rgb[0].cpu().numpy().shape)
        box_image = wandb.Image(raw_image, boxes = {"predictions": {"box_data": predicted_boxes, "class_labels" : class_id_to_label},
                                                    "gts": {"box_data": gt_boxes, "class_labels" : class_id_to_label}},
                                                    caption=str(filename))
        box_image_rgb = wandb.Image(raw_image_rgb, boxes = {"predictions": {"box_data": predicted_boxes, "class_labels" : class_id_to_label},
                                                    "gts": {"box_data": gt_boxes, "class_labels" : class_id_to_label}},
                                                   caption=str(filename_rgb))
        wandb.log({'thermal': box_image})
        wandb.log({'rgb': box_image_rgb})


class FasterRCNNBoxScoreTarget:
    """ For every original detected bounding box specified in "bounding boxes",
        assign a score on how the current bounding boxes match it,
            1. In IOU
            2. In the classification score.
        If there is not a large enough overlap, or the category changed,
        assign a score of 0.

        The total score is the sum of all the box scores.
    """
    def __init__(self, labels, bounding_boxes, iou_threshold=0.5):
        self.labels = labels
        self.bounding_boxes = bounding_boxes
        self.iou_threshold = iou_threshold

    def __call__(self, model_outputs):
        output = torch.Tensor([0])
        if torch.cuda.is_available():
            output = output.cuda()

        if len(model_outputs['bbox']) == 0:
            return output

        for box, label in zip(self.bounding_boxes, self.labels):
            box = torch.Tensor(box[None, :])
            if torch.cuda.is_available():
                box = box.cuda()

            ious = torchvision.ops.box_iou(box, model_outputs['bbox'])
            index = ious.argmax()
            if ious[0, index] > self.iou_threshold and model_outputs['cls'][index] == label:
                score = ious[0, index] + model_outputs['scores'][index]
                output = output + score
        return output

def fasterrcnn_reshape_transform(x):
    # print(x.shape)
    # print(x)
    target_size = x.size()[-2 : ]
    activations = []
    for key, value in x.items():
        activations.append(torch.nn.functional.interpolate(torch.abs(value), target_size, mode='bilinear'))
    activations = torch.cat(activations, axis=1)
    return activations