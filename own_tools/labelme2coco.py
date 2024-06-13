# -*- coding: utf-8 -*-
# come from https://www.guyuehome.com/37048
import os
import json
import random

from PIL import Image,ImageDraw
import numpy as np

from utils import img_b64_to_arr


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


class labelme2coco(object):
    def __init__(self, labelme_json=[], save_json_path="./tran.json"):
        self.labelme_json = labelme_json
        self.save_json_path = save_json_path
        self.images = []
        self.categories = []
        self.annotations = []
        # self.data_coco = {}
        self.label = []
        self.annID = 1
        self.height = 0
        self.width = 0

        self.save_json()

    def data_transfer(self):
        for num, json_file in enumerate(self.labelme_json):
            print(f"{num} : {json_file}")
            with open(json_file, "r", encoding="utf8", errors="ignore") as fp:
                data = json.load(fp)  # 加载json文件
                self.images.append(self.image(data, num))
                for shapes in data["shapes"]:
                    label = shapes["label"]
                    if label not in self.label:
                        self.categories.append(self.categorie(label))
                        self.label.append(label)
                    points = shapes[
                        "points"
                    ]  # 这里的point是用rectangle标注得到的，只有两个点，需要转成四个点
                    points.append([points[0][0], points[1][1]])
                    points.append([points[1][0], points[0][1]])
                    self.annotations.append(self.annotation(points, label, num))
                    self.annID += 1

    def image(self, data, num):
        image = {}
        img = img_b64_to_arr(data["imageData"])  # 解析原图片数据
        # img=io.imread(data['imagePath']) # 通过图片路径打开图片
        # img = cv2.imread(data['imagePath'], 0)
        height, width = img.shape[:2]
        img = None
        image["height"] = height
        image["width"] = width
        image["id"] = num + 1
        image["file_name"] = data["imagePath"].split("/")[-1]

        self.height = height
        self.width = width

        return image

    def categorie(self, label):
        categorie = {}
        categorie["supercategory"] = "Cancer"
        categorie["id"] = len(self.label) + 1  # 0 默认为背景
        categorie["name"] = label
        return categorie

    def annotation(self, points, label, num):
        annotation = {}
        annotation["segmentation"] = [list(np.asarray(points).flatten())]
        annotation["iscrowd"] = 0
        annotation["image_id"] = num + 1
        # annotation['bbox'] = str(self.getbbox(points)) # 使用list保存json文件时报错（不知道为什么）
        # list(map(int,a[1:-1].split(','))) a=annotation['bbox'] 使用该方式转成list
        annotation["bbox"] = list(map(float, self.getbbox(points)))
        annotation["area"] = annotation["bbox"][2] * annotation["bbox"][3]
        # annotation['category_id'] = self.getcatid(label)
        annotation["category_id"] = self.getcatid(label)  # 注意，源代码默认为1
        annotation["id"] = self.annID
        return annotation

    def getcatid(self, label):
        for categorie in self.categories:
            if label == categorie["name"]:
                return categorie["id"]
        return 1

    def getbbox(self, points):
        # img = np.zeros([self.height,self.width],np.uint8)
        # cv2.polylines(img, [np.asarray(points)], True, 1, lineType=cv2.LINE_AA) # 画边界线
        # cv2.fillPoly(img, [np.asarray(points)], 1) # 画多边形 内部像素值为1
        polygons = points

        mask = self.polygons_to_mask([self.height, self.width], polygons)
        return self.mask2box(mask)

    def mask2box(self, mask):
        """从mask反算出其边框 mask：[h,w] 0、1组成的图片 1对应对象，只需计算1对应的行列号（左上角行列号，右下角行列号，就可以算出其边框）"""
        # np.where(mask==1)
        index = np.argwhere(mask == 1)
        rows = index[:, 0]
        clos = index[:, 1]
        # 解析左上角行列号
        left_top_r = np.min(rows)  # y
        left_top_c = np.min(clos)  # x

        # 解析右下角行列号
        right_bottom_r = np.max(rows)
        right_bottom_c = np.max(clos)

        # return [(left_top_r,left_top_c),(right_bottom_r,right_bottom_c)]
        # return [(left_top_c, left_top_r), (right_bottom_c, right_bottom_r)]
        # return [left_top_c, left_top_r, right_bottom_c, right_bottom_r] # [x1,y1,x2,y2]
        return [
            left_top_c,
            left_top_r,
            right_bottom_c - left_top_c,
            right_bottom_r - left_top_r,
        ]  # [x1,y1,w,h] 对应COCO的bbox格式

    def polygons_to_mask(self, img_shape, polygons):
        mask = np.zeros(img_shape, dtype=np.uint8)
        mask = Image.fromarray(mask)
        xy = list(map(tuple, polygons))
        ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
        mask = np.array(mask, dtype=bool)
        return mask

    def data2coco(self):
        data_coco = {}
        data_coco["images"] = self.images
        data_coco["categories"] = self.categories
        data_coco["annotations"] = self.annotations
        return data_coco

    def save_json(self):
        self.data_transfer()
        self.data_coco = self.data2coco()
        # 保存json文件
        json.dump(
            self.data_coco,
            open(self.save_json_path, "w"),
            ensure_ascii=False,
            indent=4,
            cls=MyEncoder,
        )  # indent=4 更加美观显示
        print(f"Json file save in {self.save_json_path}.")


def get_all_file_path(file_dir: str, filter_=(".jpg")) -> list:
    # 遍历文件夹下所有的file
    return [
        os.path.join(maindir, filename)
        for maindir, _, file_name_list in os.walk(file_dir)
        for filename in file_name_list
        if os.path.splitext(filename)[1] in filter_
    ]


def main():
    # 读取目录
    jsons_path = "/data/tmp/can_rm/mask2former_test/data"
    labelme_json = get_all_file_path(jsons_path,filter_=(".json"))
    json_size = len(labelme_json)

    # 划分训练集和验证集
    random.seed(10)
    # train_dataset = random.sample(labelme_json, int(float(json_size * 0.5)))
    # labelme_json_remain = list(set(labelme_json) - set(train_dataset))  # 取补集
    # val_dataset = random.sample(labelme_json_remain, int(float(json_size * 0.25)))
    # test_dataset = list(set(labelme_json_remain) - set(val_dataset))
    # print(f"训练集数量：{len(train_dataset)}")
    # print(f"验证集数量：{len(val_dataset)}")
    # print(f"测试集数量：{len(test_dataset)}")

    # 格式转换
    labelme2coco(
        labelme_json,
        "/data/tmp/can_rm/mask2former_test/coco_format/instances_val2017.json",
    )
    labelme2coco(
        labelme_json,
        "/data/tmp/can_rm/mask2former_test/coco_format/instances_test2017.json",
    )
    labelme2coco(
        labelme_json,
        "/data/tmp/can_rm/mask2former_test/coco_format/instances_train2017.json",
    )


if __name__ == "__main__":
    main()
