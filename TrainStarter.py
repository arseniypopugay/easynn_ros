import os
import time


def train(data, output_dir, img_size=640, epochs=50, weights="yolov5s.pt", batch=16):
    images_path, values_path = create_txt_images_from_data(data)
    data_file = create_yaml_data_file(images_path, values_path, len(data["class_labels"]), data["class_labels"])

    os.system(f"python3 yolov5/train.py --data {data_file} --img {img_size} --epochs {epochs} --weights {weights} --batch {batch} ")
    if os.exists("runs/train/exp/weights/best.pt"):
        os.system(f"mv runs/train/exp/weights/best.pt {output_dir}/{time.time()}.pt")
        os.removedirs("runs/train/exp")
    elif os.exists("runs/train/exp/weights/last.pt"):
        os.system(f"mv runs/train/exp/weights/last.pt {output_dir}/{time.time()}.pt")
        os.removedirs("runs/train/exp")
    else:
        print("Smt went wrong")
    os.system("rm -r easynn_current_data")


def create_txt_images_from_data(data):
    os.system("mkdir easynn_current_data")
    os.system("mkdir easynn_current_data/values")
    name = data["image_files"][0].split('/')[-1]
    file = data["image_files"][0].split(name)[0]
    os.system(f"cp -r {file} easynn_current_data/images/")


    for file in data["image_files"]:
        file_name = file.split("/")[-1]
        txt_file = file_name[::-1].split('.', 1)[1][::-1]+".txt"
        with open(f"easynn_current_data/values/{txt_file}", 'w') as f:
            for mark in data["marks"]:
                if file == mark["image"]:
                    class_number = data["class_labels"].index(mark["label"])
                    x_centre = (mark["corners"][0][0] + mark["corners"][1][0])/2
                    y_centre =  (mark["corners"][0][1] + mark["corners"][1][1])/2
                    width = (-mark["corners"][0][0] + mark["corners"][1][0])
                    height = (-mark["corners"][0][1] + mark["corners"][1][1])
                    f.write(f"{class_number} {x_centre} {y_centre} {width} {height}\n")

    return "easynn_current_data/images", "easynn_current_data/values"


def create_yaml_data_file(train_dir, val_dir, number_of_classes, classes):
    write_to_yaml = [
        f"train: {train_dir}\n"
        f"val: {val_dir}\n"
        f"\n"
        f"nc: {number_of_classes}\n"
        f"\n",
        f"names: {classes}\n",
    ]
    with open("cur_data.yaml", 'w') as f:
        for line in write_to_yaml:
            f.write(line)
    return "cur_data.yaml"



# data = {
#     'class_labels': ['asd'],
#     'image_files': [
#         '/home/popugayman/Desktop/nn_smehnov/easynn_marker/images/How-to-Train-Your-Dog-to-Force-Fetch.jpg',
#         '/home/popugayman/Desktop/nn_smehnov/easynn_marker/images/dog-zoomies.jpg',
#         '/home/popugayman/Desktop/nn_smehnov/easynn_marker/images/honey_575p.jpg',
#         '/home/popugayman/Desktop/nn_smehnov/easynn_marker/images/cat.jpeg'],
#     'marks': [
#         {'image': '/home/popugayman/Desktop/nn_smehnov/easynn_marker/images/How-to-Train-Your-Dog-to-Force-Fetch.jpg',
#          'label': 'asd', 'corners': [[0.1675, 0.30451127819548873], [0.8825, 0.5488721804511278]]},
#         {'image': '/home/popugayman/Desktop/nn_smehnov/easynn_marker/images/dog-zoomies.jpg', 'label': 'asd',
#          'corners': [[0.355, 0.07518796992481203], [0.8525, 0.6842105263157895]]},
#         {'image': '/home/popugayman/Desktop/nn_smehnov/easynn_marker/images/dog-zoomies.jpg', 'label': 'asd',
#          'corners': [[0.4625, 0.2631578947368421], [0.7475, 0.6992481203007519]]},
#         {'image': '/home/popugayman/Desktop/nn_smehnov/easynn_marker/images/honey_575p.jpg', 'label': 'asd',
#          'corners': [[0.065, 0.18352059925093633], [0.875, 0.9700374531835206]]},
#         {'image': '/home/popugayman/Desktop/nn_smehnov/easynn_marker/images/cat.jpeg', 'label': 'asd',
#          'corners': [[0.26755852842809363, 0.13], [0.8494983277591973, 0.52]]},
#         {'image': '/home/popugayman/Desktop/nn_smehnov/easynn_marker/images/cat.jpeg', 'label': 'asd',
#          'corners': [[0.3311036789297659, 0.2925], [0.5852842809364549, 0.5625]]},
#         {'image': '/home/popugayman/Desktop/nn_smehnov/easynn_marker/images/cat.jpeg', 'label': 'asd',
#          'corners': [[0.17391304347826086, 0.2925], [0.5652173913043478, 0.4925]]},
#         {'image': '/home/popugayman/Desktop/nn_smehnov/easynn_marker/images/cat.jpeg', 'label': 'asd',
#          'corners': [[0.4816053511705686, 0.5225], [0.7892976588628763, 0.6475]]},
#         {'image': '/home/popugayman/Desktop/nn_smehnov/easynn_marker/images/honey_575p.jpg', 'label': 'asd',
#          'corners': [[0.63, 0.7228464419475655], [0.87, 0.8689138576779026]]}]}
