import glob
import os


from PIL import Image
from torch.utils.data import Dataset


class Occluded_REID(Dataset):

    dataset_dir = ""

    def __init__(self, root="", data_folder="", transform=None, relabel=False, is_query=True):
        super(Occluded_REID, self).__init__()

        self.transform = transform

        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.data_dir = os.path.join(self.dataset_dir, data_folder)

        self.dataset = self._process_dir(self.data_dir, relabel=relabel, is_query=is_query)
        self.num_pids, self.num_imgs, self.num_cams = self._get_imagedata_info(
            self.dataset
        )

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = self._read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, img_path

    def __len__(self):
        return len(self.dataset)

    def _process_dir(self, dir_path, relabel=False, is_query=True):
        img_paths = glob.glob(os.path.join(dir_path, "*", "*.tif"))

        if is_query:
            camid = 0
        else:
            camid = 1
        pid_container = set()
        for img_path in img_paths:
            img_name = img_path.split("/")[-1]
            pid = int(img_name.split("_")[0])
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            img_name = img_path.split("/")[-1]
            pid = int(img_name.split("_")[0])
            if relabel:
                pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset

    def _read_image(self, img_path):
        """Keep reading image until succeed.
        This can avoid IOError incurred by heavy IO process."""
        got_img = False
        if not os.path.exists(img_path):
            raise IOError("{} does not exist".format(img_path))
        while not got_img:
            try:
                img = Image.open(img_path).convert("RGB")
                got_img = True
            except IOError:
                print(
                    "IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(
                        img_path
                    )
                )
                pass
        return img

    def _get_imagedata_info(self, data):
        pids, cams = [], []
        for _, pid, camid in data:
            pids += [pid]
            cams += [camid]
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        return num_pids, num_imgs, num_cams
