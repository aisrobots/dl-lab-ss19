import json, os, torch
from torchvision import transforms
from PIL import Image

COCO_PATH = "/home/dllab/coco_subset/"


class DataReader:
    def __init__(self, images_folder, annotations_file, transform=None, single_sample=False):
        print("loading dataset...")
        self.annotations = json.load(open(annotations_file, 'r'))
        self.transform = transform
        self.root = images_folder
        self.single_sample = single_sample

    def __getitem__(self, idx):
        if self.single_sample:
            idx = 8

        ann = self.annotations[idx]
        img = Image.open(os.path.join(self.root, ann["file_name"])).convert('RGB')
        
        w, h = img.size
        
        if self.transform is not None:
            img = self.transform(img)
        
        keypoints = []
        weights = []
        for i in range(17):
            kx = ann['keypoints'][i*3]#/float(w)
            ky = ann['keypoints'][i*3+1]#/float(h)
            v = ann['keypoints'][i*3+2] > 0
            keypoints += [kx, ky]
            weights.append(v)

        keypoints = torch.tensor(keypoints)
        weights = torch.tensor(weights)

        return img, keypoints, weights

    def __len__(self):
        return len(self.annotations)


def get_data_loader(batch_size=1,
                    is_train=False, single_sample=False):
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

    reader = DataReader(image_folder,
                        annotation_file,
                        transform=transform,
                        single_sample=single_sample)
    
    data_loader = torch.utils.data.DataLoader(reader,
                                              batch_size=batch_size,
                                              shuffle=is_train,
                                              num_workers=4 if is_train else 1)
    return data_loader


