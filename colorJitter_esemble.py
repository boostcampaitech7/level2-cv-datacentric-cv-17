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

from torchvision.transforms import ColorJitter
CHECKPOINT_EXTENSIONS = ['.pth', '.ckpt']
LANGUAGE_LIST = ['chinese', 'japanese', 'thai', 'vietnamese']

def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', default=os.environ.get('SM_CHANNEL_EVAL', 'data'))
    parser.add_argument('--model_dir', default=os.environ.get('SM_CHANNEL_MODEL', 'trained_models'))
    parser.add_argument('--output_dir', default=os.environ.get('SM_OUTPUT_DATA_DIR', 'predictions'))

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--input_size', type=int, default=2048)
    parser.add_argument('--batch_size', type=int, default=5)

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args


def apply_color_jitter(images):
    color_jitter = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2)
    jittered_images = [color_jitter(image) for image in images]
    return jittered_images

from shapely.geometry import Polygon
import numpy as np

def calculate_iou(box1, box2):
    """
    두 박스 간의 IOU를 계산합니다.
    박스 형식: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
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

def hard_voting(detections, iou_threshold=0.5, min_vote_count=1):
    """
    하드 보팅을 사용하여 최종 박스를 결정합니다.
    
    매개변수:
        detections: 추론된 박스 리스트 (각각은 [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] 형식)
        iou_threshold: 박스 중첩을 판단할 IOU 임계값
        min_vote_count: 유지하기 위한 최소 투표 수
    
    반환값:
        최종 선택된 박스들의 리스트
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

        # 중첩된 박스들이 min_vote_count 이상일 경우 유지
        if len(overlapping_boxes) >= min_vote_count:
            # 점들의 평균 좌표를 계산하여 최종 박스를 생성
            averaged_box = np.mean([np.array(box) for box in overlapping_boxes], axis=0).tolist()
            final_boxes.append(averaged_box)
    
    return final_boxes


def do_inference(model, ckpt_fpath, data_dir, input_size, batch_size, split='test'):
    model.load_state_dict(torch.load(ckpt_fpath, map_location='cpu'))
    model.eval()

    image_fnames, final_bboxes = [], []
    images = []

    for image_fpath in tqdm(sum([glob(osp.join(data_dir, f'{lang}_receipt/img/{split}/*')) for lang in LANGUAGE_LIST], [])):
        image_fnames.append(osp.basename(image_fpath))
        original_image = cv2.imread(image_fpath)[:, :, ::-1]
        images.append(original_image)

        if len(images) == batch_size:
            # 원본 이미지에서 감지 실행
            original_bboxes = detect(model, images, input_size)
            
            # 컬러 지터링된 이미지에서 감지 실행
            jittered_images = apply_color_jitter(images)
            jittered_bboxes = detect(model, jittered_images, input_size)
            
            # 두 추론 결과를 결합
            for orig, jittered in zip(original_bboxes, jittered_bboxes):
                combined_detections = orig + jittered
                # 하드 보팅을 통해 최종 박스 생성
                final_bboxes.append(hard_voting(combined_detections))

            images = []

    if len(images):
        # 남은 이미지들에 대해 감지 실행
        original_bboxes = detect(model, images, input_size)
        jittered_images = apply_color_jitter(images)
        jittered_bboxes = detect(model, jittered_images, input_size)
        
        # 두 추론 결과를 결합
        for orig, jittered in zip(original_bboxes, jittered_bboxes):
            combined_detections = orig + jittered
            # 하드 보팅을 통해 최종 박스 생성
            final_bboxes.append(hard_voting(combined_detections))

    # ufo 형식으로 변환하여 결과 저장
    ufo_result = dict(images=dict())
    for image_fname, bboxes in zip(image_fnames, final_bboxes):
        words_info = {idx: dict(points=bbox.tolist()) for idx, bbox in enumerate(bboxes)}
        ufo_result['images'][image_fname] = dict(words=words_info)

    return ufo_result



def main(args):
    # Initialize model
    model = EAST(pretrained=False).to(args.device)

    # Get paths to checkpoint files
    ckpt_fpath = osp.join(args.model_dir, 'epoch_100.pth')

    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print('Inference in progress')

    ufo_result = dict(images=dict())
    split_result = do_inference(model, ckpt_fpath, args.data_dir, args.input_size,
                                args.batch_size, split='test')
    ufo_result['images'].update(split_result['images'])

    output_fname = 'colorJitter.csv'
    with open(osp.join(args.output_dir, output_fname), 'w') as f:
        json.dump(ufo_result, f, indent=4)


if __name__ == '__main__':
    args = parse_args()
    main(args)
