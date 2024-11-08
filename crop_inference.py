import os
import os.path as osp
import json
from argparse import ArgumentParser
from glob import glob

import torch
import cv2
from torch import cuda
from model import EAST
from tqdm import tqdm
from detect import detect
from shapely.geometry import Polygon
import numpy as np


CHECKPOINT_EXTENSIONS = ['.pth', '.ckpt']
LANGUAGE_LIST = ['chinese', 'japanese', 'thai', 'vietnamese']


def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', default=os.environ.get('SM_CHANNEL_EVAL', 'data'))
    parser.add_argument('--model_dir', default=os.environ.get('SM_CHANNEL_MODEL', 'trained_models'))
    parser.add_argument('--output_dir', default=os.environ.get('SM_OUTPUT_DATA_DIR', 'predictions'))
    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--image_size', type=int, default=1536)
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=5)
    
    # New arguments for crop size and movement count
    parser.add_argument('--crop_ratio', type=float, default=2/3, help="Ratio for image crop size during sliding window inference.")
    parser.add_argument('--num_crops', type=int, default=2, help="Number of movements in sliding window (e.g., 2 means a 2x2 sliding window).")

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args


def calculate_iou(box1, box2):
    """
    Calculate IoU between two boxes.
    Format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    """
    poly1 = Polygon(box1)
    poly2 = Polygon(box2)
    
    if not poly1.is_valid or not poly2.is_valid:
        return 0.0
        
    try:
        intersection = poly1.intersection(poly2).area
        union = poly1.union(poly2).area
        return intersection / union if union > 0 else 0.0
    except:
        return 0.0


def hard_voting(detections, iou_threshold=0.4, min_vote_count=2):
    """
    Apply hard voting to determine final boxes.
    Parameters:
        detections: list of boxes [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        iou_threshold: IOU threshold for overlapping boxes
        min_vote_count: minimum votes to keep a box
    """
    final_boxes = []
    while detections:
        current_box = detections.pop(0)
        overlapping_boxes = [current_box]
        
        i = 0
        while i < len(detections):
            iou = calculate_iou(current_box, detections[i])
            if iou >= iou_threshold:
                overlapping_boxes.append(detections.pop(i))
            else:
                i += 1

        if len(overlapping_boxes) >= min_vote_count:
            averaged_box = np.mean([np.array(box) for box in overlapping_boxes], axis=0)
            final_boxes.append(averaged_box)
    
    return final_boxes


def do_inference_with_voting(model, ckpt_fpath, data_dir, input_size, image_size, batch_size, margin=10, split='test', crop_ratio=2/3, num_crops=2):
    model.load_state_dict(torch.load(ckpt_fpath, map_location='cpu'))
    model.eval()

    image_fnames, by_sample_bboxes = [], []
    images = []

    image_paths = sum([glob(osp.join(data_dir, f'{lang}_receipt/img/{split}/*')) for lang in LANGUAGE_LIST], [])
    
    for image_fpath in tqdm(image_paths):
        image_fnames.append(osp.basename(image_fpath))
        original_image = cv2.imread(image_fpath)[:, :, ::-1]
        images.append(original_image)

        if len(images) == batch_size:
            batch_bboxes = []
            for image in images:
                h, w, _ = image.shape
                resize_ratio_input_h = h / input_size
                resize_ratio_input_w = w / input_size
                
                resized_image = cv2.resize(image, (input_size, input_size))
                detections = detect(model, [resized_image], input_size)[0]
                detections = [
                    [[pt[0] * resize_ratio_input_w, pt[1] * resize_ratio_input_h] for pt in bbox] 
                    for bbox in detections
                ]

                resized_image = cv2.resize(image, (image_size, image_size))
                resize_ratio_input_h = h / image_size
                resize_ratio_input_w = w / image_size
                
                crops = [
                    (resized_image[i * image_size // num_crops:(i + crop_ratio) * image_size // num_crops, 
                                   j * image_size // num_crops:(j + crop_ratio) * image_size // num_crops], 
                     i * image_size // num_crops, j * image_size // num_crops)
                    for i in range(num_crops) for j in range(num_crops)
                ]

                for crop, offset_y, offset_x in crops:
                    crop_detections = detect(model, [crop], input_size)[0]
                    transformed_detections = [
                        [[(pt[0] + offset_x) * resize_ratio_input_w, (pt[1] + offset_y) * resize_ratio_input_h] for pt in bbox]
                        for bbox in crop_detections
                    ]
                    transformed_detections = [
                        bbox for bbox in transformed_detections
                        if all(margin < pt[0] < (w - margin) and margin < pt[1] < (h - margin) for pt in bbox)
                    ]
                    detections.extend(transformed_detections)

                final_bboxes = hard_voting(detections)
                batch_bboxes.append(final_bboxes)

            by_sample_bboxes.extend(batch_bboxes)
            images = []

    ufo_result = dict(images=dict())
    for image_fname, bboxes in zip(image_fnames, by_sample_bboxes):
        words_info = {idx: dict(points=bbox.tolist()) for idx, bbox in enumerate(bboxes)}
        ufo_result['images'][image_fname] = dict(words=words_info)

    return ufo_result


def main(args):
    model = EAST(pretrained=False).to(args.device)
    ckpt_fpath = osp.join(args.model_dir, 'epoch_140.pth')

    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print('Inference in progress')

    ufo_result = dict(images=dict())
    split_result = do_inference_with_voting(model, ckpt_fpath, args.data_dir, args.input_size, args.image_size,
                                args.batch_size, split='test', crop_ratio=args.crop_ratio, num_crops=args.num_crops)
    ufo_result['images'].update(split_result['images'])

    output_fname = '1.5esemble_1536_1024_modify_0.4_2.csv'
    with open(osp.join(args.output_dir, output_fname), 'w') as f:
        json.dump(ufo_result, f, indent=4)


if __name__ == '__main__':
    args = parse_args()
    main(args)
