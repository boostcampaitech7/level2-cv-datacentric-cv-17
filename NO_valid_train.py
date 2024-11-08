#원본 + augmantation적용이미지 를 훈련시키는 코드
#validation을 지정하지 않고, Rotate하지 않고 추론한다.

import os
import os.path as osp
import time
from datetime import timedelta
from argparse import ArgumentParser

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from east_dataset import EASTDataset
from NO_valid_dataset import SceneTextDataset
from model import EAST

root_dir='/data/ephemeral/home/MCG/Out'

def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', 'data'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR',
                                                                        'trained_models'))

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--image_size', type=int, default=2048)
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=50)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--early_stopping_patience', type=int, default=60,
                        help="Number of epochs to wait for validation loss improvement before stopping early.")

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args
import os
import cv2
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch.utils.data import ConcatDataset
from torch.cuda.amp import GradScaler, autocast
import time
from torch.optim.lr_scheduler import CosineAnnealingLR

def do_training(data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval, early_stopping_patience):
    # Train 데이터셋과 Validation 데이터셋 초기화
    train_dataset = SceneTextDataset(
        root_dir=root_dir,
        split='train',
        image_size=image_size,
        crop_size=input_size,
    )

    # 증강을 강하게 적용한 데이터셋 추가
    aug_train_dataset = SceneTextDataset(
        root_dir=root_dir,
        split='aug_train',
        image_size=image_size,
        crop_size=input_size,
    )
    # 원본과 증강된 데이터셋 결합
    combined_train_dataset = ConcatDataset([train_dataset, aug_train_dataset])

    # DataLoader 설정
    train_loader = DataLoader(
        EASTDataset(combined_train_dataset),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2
    )


    # 모델, 최적화 설정 및 훈련 시작
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[25,40,47,], gamma=0.3)
    
    model.train()
    for epoch in range(max_epoch):
        epoch_loss=0
        cls_loss_sum, angle_loss_sum, iou_loss_sum = 0, 0, 0
        start_time = time.time()

        for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
            loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            cls_loss_sum += extra_info['cls_loss']
            angle_loss_sum += extra_info['angle_loss']
            iou_loss_sum += extra_info['iou_loss']

        scheduler.step()

        train_time = time.time() - start_time
        mean_train_loss = epoch_loss / len(train_loader)
        mean_cls_loss = cls_loss_sum / len(train_loader)
        mean_angle_loss = angle_loss_sum / len(train_loader)
        mean_iou_loss = iou_loss_sum / len(train_loader)
        
        print(f"Epoch [{epoch + 1}/{max_epoch}] - Train Loss: {mean_train_loss:.4f} "
              f"(Cls: {mean_cls_loss:.4f}, Angle: {mean_angle_loss:.4f}, IoU: {mean_iou_loss:.4f}), "
              f"Train Time: {train_time:.2f}s")

        # 지정된 간격으로 모델 체크포인트 저장
        if (epoch + 1) % save_interval == 0:
            if not osp.exists(model_dir):
                os.makedirs(model_dir)

            ckpt_fpath = osp.join(model_dir, f'epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), ckpt_fpath)
            print(f'Checkpoint saved at epoch {epoch + 1}')


def main(args):
    do_training(**args.__dict__)

if __name__ == '__main__':
    args = parse_args()
    main(args)
