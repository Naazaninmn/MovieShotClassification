from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

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
    train_split_length = total_examples * 5/6  # 5/6 of the training split used for validation
    test_split_length = total_examples * 1/6  # 1/6 of the training split used for validation
    #val_split_length = train_split_length * 0.2  # 20% of the training split used for validation

    train_examples_x = []
    train_examples_y = []
    #val_examples = []
    test_loader = []

    #train_examples_dict = {}

    for category_idx, examples_list in examples.items():
        split_idx = 1/5 * test_split_length
        for i, example in enumerate(examples_list):
            if i > split_idx:
                # if category_idx not in train_examples_dict.keys():
                #     train_examples_dict[category_idx] = [example]
                # else:
                #     train_examples_dict[category_idx].append(example)
                train_examples_x.append(example)  
                train_examples_y.append(category_idx) 
            else:
                test_loader.append([example, category_idx]) # each pair is [path_to_img, class_label]
    

    # for category_idx, examples_list in train_examples_dict.items():
    #     split_idx = 1/5 * val_split_length
    #     for i, example in enumerate(examples_list):
    #         if i > split_idx:
    #             train_examples.append([example, category_idx])  # each pair is [path_to_img, class_label]
    #         else:
    #             val_examples.append([example, category_idx])  # each pair is [path_to_img, class_label]
    

    # Transforms
    normalize = T.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # VGG-16 - ImageNet Normalization

    train_transform = T.Compose([
        T.Resize(256),
        T.ColorJitter(),
        T.RandomHorizontalFlip(p=1.0),
        T.ToTensor(),
        normalize
    ])

    eval_transform = T.Compose([
        T.Resize(256),
        T.ToTensor(),
        normalize
    ])

    # Dataloaders
    #train_loader = DataLoader(ShotDataset(train_examples, train_transform), shuffle=True)
    #val_loader = DataLoader(ShotDataset(val_examples, eval_transform), shuffle=False)
    #test_loader = DataLoader(ShotDataset(test_examples, eval_transform), shuffle=False)

    return train_examples_x, train_examples_y, test_loader
