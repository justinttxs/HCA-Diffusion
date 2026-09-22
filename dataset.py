from torch.utils.data import Dataset
import os
from PIL import Image


class base_dataset(Dataset):
    def __init__(self, dir, transform=None) -> None:
        super().__init__()

        self.dir = dir
        src_dir = os.path.join(dir, 'src')
        tgt_dir = os.path.join(dir, 'target')

        valid_ext = ('.png', '.jpg', '.jpeg')

        self.img_names = sorted([
            name for name in os.listdir(src_dir)
            if os.path.isfile(os.path.join(src_dir, name))
            and name.lower().endswith(valid_ext)
        ])

        self.tgt_img_names = sorted([
            name for name in os.listdir(tgt_dir)
            if os.path.isfile(os.path.join(tgt_dir, name))
            and name.lower().endswith(valid_ext)
        ])
        # print(self.img_names)

        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        img_path = os.path.join(self.dir, 'src', self.img_names[index])
        print(f"Accessing index {index}, total images: {len(self.img_names)}")  # 调试输出
        if index >= len(self.img_names) or index < 0:
            raise IndexError(f"Index {index} is out of range for image list with length {len(self.img_names)}")

        img = Image.open(img_path).convert('RGB')

        tgt_index = index % len(self.tgt_img_names)
        tgt_img_path = os.path.join(self.dir, 'target', self.tgt_img_names[tgt_index])
        tgt_img = Image.open(tgt_img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)
            tgt_img = self.transform(tgt_img)

        return img, tgt_img
