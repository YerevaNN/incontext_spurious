import os
import numpy as np
from PIL import Image

import logging
from src.utils import download_from_url
from wilds.datasets.waterbirds_dataset import WaterbirdsDataset

log = logging.getLogger(__name__)

class CustomizedWaterbirdsDataset(WaterbirdsDataset):
    def __init__(self, root_dir, background, *args, **kwargs):
        super().__init__(root_dir=root_dir, *args, **kwargs)

        self._root_dir = root_dir
        self._background = background

        if background['change_bg']:
            self._backgrounds = None
            self._masks_path = None
            self._places_path = None

            self._backgrounds = self.prepare_backgrounds()

    def download_backgrounds_and_masks(self):
        urls = {
            "waterbirds_masks": "https://drive.google.com/u/0/uc?id=1SMEOBwtEKFr_hnfny3kYiMiM7rdzb57w&export=download&confirm=t&uuid=8169d5b8-05f9-4284-9a2f-ea7d72dfe153&at=AB6BwCDkCw15rDLoUzBvn76AE0M5:1696711362152",
            "waterbirds_places": "https://drive.google.com/u/0/uc?id=1XhZgj03U_mG7KcZhpBhlxHMddG7kuv0L&export=download&confirm=t&uuid=a84fd8ae-6d75-4866-af3b-43032cfcf262&at=AB6BwCD6x6i-_L7VQuyKuVzmoZbR:1697109397161"
        }

        for dirname, url in urls.items():
            path_to_save = os.path.join(self._root_dir, dirname)
            if not os.path.exists(path_to_save):
                log.info(f"Downloading the {dirname} from url")
                download_from_url(path_to_save=self._root_dir, url=url, mode="zip")

    def prepare_backgrounds(self):
        self.download_backgrounds_and_masks()

        self._masks_path = os.path.join(self._root_dir, "waterbirds_masks")
        self._places_path = os.path.join(self._root_dir, "waterbirds_places", "train")

        backgrounds = []
        backgrounds_type = ['land', 'water']
        
        for bg_type in backgrounds_type:
            background_paths = os.path.join(self._places_path, bg_type)

            paths = [os.path.join(background_paths, file) for file in os.listdir(background_paths)]

            random_backgrounds = np.random.choice(paths, self._background[bg_type], replace=False).tolist()

            background_type = [(path, backgrounds_type.index(bg_type)) for path in random_backgrounds]

            backgrounds += background_type

        return backgrounds

    def crop_and_resize(self, source_img, target_img):
        source_width = source_img.size[0]
        source_height = source_img.size[1]

        target_width = target_img.size[0]
        target_height = target_img.size[1]

        # Check if source does not completely cover target
        if (source_width < target_width) or (source_height < target_height):
            # Try matching width
            width_resize = (target_width, int((target_width / source_width) * source_height))
            if (width_resize[0] >= target_width) and (width_resize[1] >= target_height):
                source_resized = source_img.resize(width_resize, Image.ANTIALIAS)
            else:
                height_resize = (int((target_height / source_height) * source_width), target_height)
                assert (height_resize[0] >= target_width) and (height_resize[1] >= target_height)
                source_resized = source_img.resize(height_resize, Image.ANTIALIAS)
            # Rerun the cropping
            return self.crop_and_resize(source_resized, target_img)

        source_aspect = source_width / source_height
        target_aspect = target_width / target_height

        if source_aspect > target_aspect:
            # Crop left/right
            new_source_width = int(target_aspect * source_height)
            offset = (source_width - new_source_width) // 2
            resize = (offset, 0, source_width - offset, source_height)
        else:
            # Crop top/bottom
            new_source_height = int(source_width / target_aspect)
            offset = (source_height - new_source_height) // 2
            resize = (0, offset, source_width, source_height - offset)

        source_resized = source_img.crop(resize).resize((target_width, target_height), Image.ANTIALIAS)
        return source_resized

    def combine_and_mask(self, img_new, mask, img_black): # function inspired by groupDRO (can be changed)
        # Warp new img to match black img
        img_resized = self.crop_and_resize(img_new, img_black)
        img_resized_np = np.asarray(img_resized)

        # Mask new img
        img_masked_np = np.around(img_resized_np * (1 - mask)).astype(np.uint8)

        # Combine
        img_combined_np = np.asarray(img_black) + img_masked_np
        img_combined = Image.fromarray(img_combined_np)

        return img_combined
    
    def __getitem__(self, idx):
        x, y, metadata = super().__getitem__(idx)
        x, y, c = x, y, metadata[0]

        if self._background['change_bg'] and self._split_array[idx] == 0: # change backgrounds only train dataset
            mask_path = os.path.join(self._masks_path, f"{idx}.npy")
            mask = np.load(mask_path) / 255
            x = Image.fromarray((x * mask).astype(np.uint8))

            random_id = np.random.choice(len(self._backgrounds))

            place_path, c = self._backgrounds[random_id]

            place = Image.open(place_path)

            x = self.combine_and_mask(place, mask, x)
            

        return x, y, c
    


