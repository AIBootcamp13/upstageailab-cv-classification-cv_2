import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pandas as pd
from sklearn.model_selection import train_test_split

from src.datasets.basic import BasicDataset
from src.datasets.augmented import AugmentedDataset
from src.datasets.augmented_v2 import Augmented_V2_Dataset
from src.datasets.test_dataset import TestDataset
from src.transforms.basic import BasicTransform
from src.transforms.augmented import AugmentedTransform
from PIL import Image
import os
def create_data_loaders(config):
    """설정에 따라 데이터 로더 생성"""
    
    # 변환 생성
    transform_config = config['transform']
    transform_name = transform_config['name']
    
    if transform_name == 'basic':
        transform = BasicTransform(
            image_size=transform_config.get('image_size', 224)
        )
    elif (transform_name == 'augmented') or (transform_name == 'augmented_v2'):
        transform = AugmentedTransform(
            image_size=transform_config.get('image_size', 224)
        )
    else:
        raise ValueError(f"지원하지 않는 변환: {transform_name}")
    
    # 데이터 로드 및 분할
    dataset_config = config['dataset']
    dataset_name = dataset_config['name']
    
    # 전체 데이터 로드
    full_data = pd.read_csv(dataset_config['train_csv'])
    # train/val 분할
    # train_data, val_data = train_test_split(
    #     full_data, 
    #     test_size=0.2, 
    #     random_state=config.get('seed', 42),
    #     stratify=full_data.iloc[:, 1] if len(full_data.columns) > 1 else None
    # )

    # if dataset_name == 'augmented' or (transform_name == 'augmented_v2'):
    #     train_data = pd.read_csv(dataset_config['train_csv'])
    #     train_data['path'] = dataset_config['train_img_dir']
    #     val_data = pd.read_csv(dataset_config['val_csv'])
    #     val_data['path'] = dataset_config['val_img_dir']
    # else:
    #     pass
    train_data, val_data = train_test_split(
        full_data,
        test_size=0.2,  # 5:5 비율로 설정
        random_state=config.get('seed', 42),
        # 열의 위치 대신 '이름'을 사용하여 stratify 지정 (더 안정적인 방법)
        shuffle=True,
        stratify=full_data['target'] if 'target' in full_data.columns else None
    )

    oversample_labels = [3,4,7,14,17]
    mask = train_data['target'].isin(oversample_labels)
    df_dup1 = pd.concat([train_data[mask]] * 5, ignore_index=True)
    df_dup2 = pd.concat([train_data[~mask]] * 4, ignore_index=True)

    df_aug = pd.concat([df_dup1, df_dup2], ignore_index=True)
    train_data = df_aug.sample(frac=1, random_state=config.get('seed', 42)).reset_index(drop=True)
    
    # 임시 CSV 파일 생성
    train_csv = 'temp_train.csv'
    val_csv = 'temp_val.csv'
    train_data.to_csv(train_csv, index=False)
    val_data.to_csv(val_csv, index=False)
    
    # 데이터셋 생성
    if dataset_name == 'basic':
        train_dataset = BasicDataset(
            csv_file=train_csv,
            img_dir=dataset_config['train_img_dir'],
            transform=transform
        )
        val_dataset = BasicDataset(
            csv_file=val_csv,
            img_dir=dataset_config['val_img_dir'],
            transform=transform
        )
    elif dataset_name == 'augmented':
        train_dataset = AugmentedDataset(
            csv_file=train_csv,
            img_dir=dataset_config['train_img_dir'],
            transform=transform
        )
        val_dataset = AugmentedDataset(
            csv_file=val_csv,
            img_dir=dataset_config['val_img_dir'],
            transform=transform
        )
    elif dataset_name == 'augmented_v2':
        train_dataset = Augmented_V2_Dataset(
            csv_file=train_csv,
            img_dir=dataset_config['train_img_dir'],
            transform=transform
        )
        val_dataset = Augmented_V2_Dataset(
            csv_file=val_csv,
            img_dir=dataset_config['val_img_dir'],
            transform=transform
        )
    else:
        raise ValueError(f"지원하지 않는 데이터셋: {dataset_name}")
    import torch.multiprocessing as mp

    # train_ctx = mp.get_context('spawn')
    # 데이터 로더 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True,
        # multiprocessing_context=train_ctx,
        # persistent_workers=False,
        drop_last=True
    )
    # val_ctx = mp.get_context('spawn')
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True,
        # multiprocessing_context=val_ctx,
        # persistent_workers=False,
        drop_last=True
    )
    
    return train_loader, val_loader 