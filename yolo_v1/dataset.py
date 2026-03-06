import torch
from PIL import Image
import pandas as pd
import os


class dataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, label_dir, img_dir, s, b, c, transform=None):
        super(dataset, self).__init__()
        self.s = s
        self.c = c
        self.b = b
        self.transform = transform
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        img = Image.open(img_path)

        with open(label_path) as f:
            for label in f.readlines():
                c, x, y, w, h = [
                    float(x) if float(x) != int(float(x)) else int(x) for x in label.replace("\n", "").split()
                ]
                boxes.append([c, x, y, w, h])
        if self.transform:
            img, boxes = self.transform(img, boxes)
        boxes = torch.tensor(boxes)
        label_matrix = torch.zeros(size=(self.s, self.s, self.c + 5 * self.b))
        for box in boxes:
            c, x, y, w, h = box
            c = int(c)
            i, j = int(self.s * y), int(self.s * x)
            x_cell, y_cell = self.s * x - j, self.s * y - i

            w_cell, h_cell = w * self.s, h * self.s
            if label_matrix[i, j, 20] == 0:
                label_matrix[i, j, 20] = 0
                box_coord = torch.tensor([x_cell, y_cell, w_cell, h_cell])
                label_matrix[i, j, 21:25] = box_coord
                label_matrix[i, j, c] = 1
        return img, label_matrix
class compose(object):
    def __init__(self,transforms):
        self.transforms=transforms
    def __call__(self,img,bb):
        for t in self.transforms:
            img,bb=t(img),bb
        return img,bb