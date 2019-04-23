import json, os, torch
from torchvision import transforms
from PIL import Image, ImageDraw
import numpy as np

COCO_PATH = "/home/dllab/coco_subset/"


class DataReaderSegmentation:
    def __init__(self, images_folder, annotations_file, transform=None):
        print("loading dataset...")
        self.annotations = json.load(open(annotations_file, 'r'))
        self.transform = transform
        self.root = images_folder

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img = Image.open(os.path.join(self.root, ann["file_name"])).convert('RGB')
        
        w, h = img.size
        
        if self.transform is not None:
            img = self.transform(img)
        
        mask = Image.new('L', (w, h), 0)
        segmentation = [int(e) for e in ann["segmentation"][0]]
        ImageDraw.Draw(mask).polygon(segmentation, outline=1, fill=1)
        mask = torch.tensor(np.array(mask)).type(torch.FloatTensor)
        
        return img, mask

    def __len__(self):
        return len(self.annotations)
 

def get_data_loader(batch_size=1, is_train=False):
    transform_list = []

    if is_train:
        transform_list.append(transforms.ColorJitter(hue=.05, saturation=.05))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    transform = transforms.Compose(transform_list)

    if is_train:
        image_folder, annotation_file = os.path.join(COCO_PATH, 'train_256'),\
                                        os.path.join(COCO_PATH, 'annotations', 'keypoints_256_train.json')
    else:
        image_folder, annotation_file = os.path.join(COCO_PATH, 'val_256'),\
                                        os.path.join(COCO_PATH, 'annotations', 'keypoints_256_val.json'),

    reader = DataReaderSegmentation(image_folder,
                                    annotation_file,
                                    transform=transform)
    
    data_loader = torch.utils.data.DataLoader(reader,
                                              batch_size=batch_size,
                                              shuffle=is_train,
                                              num_workers=4 if is_train else 1)
    return data_loader
