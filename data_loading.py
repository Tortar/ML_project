
import os
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.utils import save_image
from data_transformation import image_transform_grayscale, image_transform_RGB, image_transform_RGB_2
from torchvision import transforms
from sklearn.model_selection import KFold


def augment_data():

    #data = datasets.ImageFolder(root="./data/train", 
    #                            transform=image_transform_RGB)

    augmented_data_1 = datasets.ImageFolder(root="./data/train", 
                                          transform=image_transform_RGB_2)

    augmented_data_2 = datasets.ImageFolder(root="./data/train", 
                                          transform=image_transform_RGB_2)

    augmented_data_3 = datasets.ImageFolder(root="./data/train", 
                                          transform=image_transform_RGB_2)

    augmented_data_4 = datasets.ImageFolder(root="./data/train", 
                                          transform=image_transform_RGB_2)

    new_data = [augmented_data_1, augmented_data_2, augmented_data_3, augmented_data_4]


    try: os.mkdir("./data/train_augmented")
    except FileExistsError: pass

    try: os.mkdir("./data/train_augmented/muffin")
    except FileExistsError: pass

    try: os.mkdir("./data/train_augmented/chihuahua")
    except FileExistsError: pass

    v = 0
    for ds in new_data:
        for i in ds:
            v += 1
            if i[1] == 0:
                save_image(i[0], f"./data/train_augmented/chihuahua/{v}.png")
            else:
                save_image(i[0], f"./data/train_augmented/muffin/{v}.png")

#augment_data()

muffins_filenames = [f"./data/{s}/muffin/" + im for s in ["train", "test"] 
                     for im in os.listdir(f"./data/{s}/muffin")]

chihuahuas_filenames = [f"./data/{s}/chihuahua/" + im for s in ["train", "test"] 
                        for im in os.listdir(f"./data/{s}/chihuahua")]


train_data = datasets.ImageFolder(root="./data/train/",
                                  transform=image_transform_RGB,
                                  target_transform=None)

test_data = datasets.ImageFolder(root="./data/test/", 
                                 transform=image_transform_RGB)


kf = KFold(n_splits=5, shuffle=True, random_state=42)

folds = []

for fold, (train_idx, validation_idx) in enumerate(kf.split(train_data)):

    fold_train = Subset(train_data, train_idx)
    fold_validation = Subset(train_data, validation_idx)

    folds.append([fold_train, fold_validation])

def KFoldDataLoader(k=5):

    assert k == len(folds)

    for fold in folds:

        train_dataloader = DataLoader(
            dataset=fold[0],
            batch_size=32,
            pin_memory=True,
            shuffle=True,
            num_workers=8,
        )
        validation_dataloader = DataLoader(
            dataset=fold[1],
            batch_size=1,
            pin_memory=True,
            shuffle=False,
            num_workers=8,
        )
        yield train_dataloader, validation_dataloader

final_train_dataloader = DataLoader(
    dataset=train_data, 
    batch_size=32, 
    pin_memory=True,
    num_workers=8, 
    shuffle=True
)

final_test_dataloader = DataLoader(
    dataset=test_data, 
    batch_size=1, 
    pin_memory=True,
    num_workers=8, 
    shuffle=False
)