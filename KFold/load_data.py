from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np

CATEGORIES = {
    'CloseUp': 0,
    'MediumCloseUp': 1,
    'MediumShot': 2,
    'MediumLongShot': 3,
    'LongShot': 4
}


class ShotDataset(Dataset):
    def __init__(self, examples, transform):
        self.examples = examples
        self.transform = transform

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        img_path, y = self.examples[index]
        x = self.transform(Image.open(img_path).convert('RGB'))
        return x, y


def read_lines(data_path):
    examples = {}
    with open('data.txt') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip().split()[0].split('/')
        category_name = line[1]
        category_idx = CATEGORIES[category_name]
        image_name = line[2]
        image_path = f'{data_path}/{category_name}/{image_name}'
        if category_idx not in examples.keys():
            examples[category_idx] = [image_path]
        else:
            examples[category_idx].append(image_path)
    return examples


def build_splits():

    examples = read_lines('Data')
    total_examples = 600

    # Build splits
    test_split_length = total_examples * 1/6  # 1/6 of the training split used for validation

    data = []
    X = []
    Y = []
    test_examples = []

    for category_idx, examples_list in examples.items():
        split_idx = 1/5 * test_split_length
        for i, example in enumerate(examples_list):
            if i >= split_idx:
                data.append([example, category_idx]) # each pair is [path_to_img, class_label]
                X.append(example)
                Y.append(category_idx)
            else:
                test_examples.append([example, category_idx]) # each pair is [path_to_img, class_label]
    
    return data, X, Y, test_examples
