"""
assume cut-and-paste is run already
"""

import json
import os
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
from xml.etree.ElementTree import Element, SubElement, tostring
import xml.etree.ElementTree as ET

cut_paste_label_txt = Path("./cut-and-paste/selected.txt")
kitchen_path = Path("./GMU-Kitchen")
scenes = ["gmu_scene_002", "gmu_scene_004", "gmu_scene_005"]

def convert_VOC_to_COCO(kitchen_path: Path, scenes: list, label2id: dict):
    def get_image_info(root: Element, scene: str):
        filename: str = root.findtext("filename")
        image_id = f"{root.findtext('folder')}_{filename}"
        size = root.find('size')
        return {
            "file_name": os.path.join(".", scene, "Images", filename),
            "height": int(size.findtext("height")),
            "width": int(size.findtext("width")),
            "id": image_id
        }
    def get_object_info(obj):
        label = obj.findtext('name')
        assert label in label2id, f"Error: {label} is not in label2id !"
        category_id = label2id[label]
        bndbox = obj.find('bndbox')
        xmin = int(float(bndbox.findtext('xmin'))) - 1
        ymin = int(float(bndbox.findtext('ymin'))) - 1
        xmax = int(float(bndbox.findtext('xmax')))
        ymax = int(float(bndbox.findtext('ymax')))
        assert xmax > xmin and ymax > ymin, f"Box size error !: (xmin, ymin, xmax, ymax): {xmin, ymin, xmax, ymax}"
        o_width = xmax - xmin
        o_height = ymax - ymin
        area = o_width * o_height
        # cut and paste paper page 7 
        # "we consider boxes of size at least 50 × 30 pixels in the images for evaluation."
        if area <= 50 * 30:
            return
        return {
            'area': area,
            'iscrowd': 0,
            'bbox': [xmin, ymin, o_width, o_height],
            'category_id': category_id,
            'ignore': 0,
            'segmentation': []  # This script is not for segmentation
        }
    
    output_json_dict = {
        "images": [],
        "type": "instances",
        "annotations": [],
        "categories": [
            {'supercategory': 'none', 'id': label_id, 'name': label}
            for label, label_id in label2id.items()
        ]
    }
    bnd_id = 1
    for scene in scenes:
        anno_path = kitchen_path / scene / "Annotations"
        for anno_file in tqdm(os.listdir(anno_path)):
            root = ET.parse(anno_path / anno_file).getroot()
            image_info = get_image_info(root, scene)
            output_json_dict["images"].append(image_info)
            for obj in root.findall("object"):
                ann = get_object_info(obj)
                if ann is None: continue
                ann.update({'image_id': image_info['id'], 'id': bnd_id})
                output_json_dict['annotations'].append(ann)
                bnd_id += 1
    with open("./Kitchen_fold1_COCO.json", "w") as f:
        f.write(json.dumps(output_json_dict))

if __name__ == "__main__":
    with open(cut_paste_label_txt) as f:
        selected_labels = [x.strip() for x in f.readlines()]
    label2id = {
        k : i for i, k in enumerate(selected_labels, 1)
    }

    convert_VOC_to_COCO(kitchen_path, scenes, label2id)
