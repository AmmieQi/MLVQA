import os
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
# def showImg(path):
#     img_name = path.split('/')[5]
#     img_name = img_name[:-4]
#     print(img_name)
#
#     img_path = "/data/kf/majie/codehub/mlvqa/vqadata/train2014/"
#     path = img_path + img_name
#
#     img = Image.open(path)
#     # cv2.rectangle(img, (bbox[0:0], bbox[0:1], bbox[0:2], bbox[0:3]), (0,255,0), 2)
#     img = cv2.imread(path)
#     plt.imshow(img)  # 显示图片
#     cv2.rectangle(img, (93, 258), (635, 426), (0, 255, 0), 4)
#     plt.axis('off')  # 不显示坐标轴
#     plt.show()
#
#
# if __name__ == "__main__":
#     path = "./data/vqa/feats/train2014/COCO_train2014_000000260040.jpg.npz"
#     showImg(path)


def showImg(path):
    frcn_feat = np.load(path)
    bbox = frcn_feat['bbox']

    img_path = "/data/kf/majie/codehub/mlvqa/vqadata/val2014/COCO_val2014_000000393225.jpg"
    img = cv2.imread(img_path)

    for i in range(14):
        img = cv2.rectangle(img, (int(bbox[i][0]), int(bbox[i][1])), (int(bbox[i][2]), int(bbox[i][3])), (0, 255, 0), 1)

        if i == 6:
            img = cv2.putText(img, str(i), (int(bbox[i][0]), int(bbox[i][1])), cv2.FONT_HERSHEY_PLAIN, 2,
                              (0, 0, 255), 2)
        else:
            img = cv2.putText(img, str(i), (int(bbox[i][0]), int(bbox[i][1]) + 20), cv2.FONT_HERSHEY_PLAIN, 2,
                          (0, 0, 255), 2)

    cv2.imshow('noodle.png', img)
    save_path = "/data/kf/majie/codehub/mlvqa/vqadata/"
    cv2.imwrite(save_path + "noodle.png", img, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == "__main__":
    path = "/data/kf/majie/codehub/mlvqa/data/vqa/feats/val2014/COCO_val2014_000000393225.jpg.npz"
    showImg(path)
