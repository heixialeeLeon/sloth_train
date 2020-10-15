import os
import cv2
from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, show_result_pyplot


def main():
    parser = ArgumentParser()
    parser.add_argument('img_root', help='Image file root')
    args = parser.parse_args()

    config = "work_dirs/mask_rcnn_r50_fpn/mask_rcnn_r50_fpn.py"
    checkpoint = "work_dirs/mask_rcnn_r50_fpn/latest.pth"
    score_thr = 0.3
    
    # build the model from a config file and a checkpoint file
    model = init_detector(config, checkpoint)

    for img_name in os.listdir(args.img_root):
        img_path = os.path.join(args.img_root, img_name)
        # test a single image
        result = inference_detector(model, img_path)

        import pdb; pdb.set_trace()
        # show the results
        img = model.show_result(img_path, result, score_thr=score_thr, show=False)
        cv2.imwrite(os.path.join("result", img_name), img)


if __name__ == '__main__':
    main()
