import os


def train(data, output_dir, img_size=640, epochs=50, weights="yolov5s.pt", batch=16):
    os.system(f"python3 yolov5/train.py --data {data} --img {img_size} --epochs {epochs} --weights {weights} --batch {batch}")
    if os.exists("runs/train/exp/weights/best.pt"):
        os.system(f"cp runs/train/exp/weights/best.pt {output_dir}")
        os.removedirs("runs/train/exp")
    elif os.exists("runs/train/exp/weights/last.pt"):
        os.system(f"cp runs/train/exp/weights/last.pt {output_dir}")
        os.removedirs("runs/train/exp")
    else:
        print("Smt went wrong")
