import cv2
import json
import numpy as np
import torch
import os

from ultralytics import YOLO
from types import SimpleNamespace
from utils import get_oriented_annotations, lies_between, mid_projection
from tqdm import tqdm
from pathlib import Path


def process(model, video_path, markup_path, vid_stride=0, tracker="bytetrack.yaml"):
    # open markup
    with open(markup_path) as f:
        markup = json.load(f)
    areas_list = get_oriented_annotations(markup)

    # tracking: [{id: [class, conf, first_frame, last_frame, valid]}]
    objects_list = [{} for _ in range(len(areas_list))]

    cap = cv2.VideoCapture(video_path)
    TOTAL_N_FRAMES = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # cast area to int
    for i in range(len(areas_list)):
        areas_list[i] = (np.array(areas_list[i]) * np.array([w, h])).astype(np.int32)
    areas_list = areas_list.astype(np.int32)

    # Loop through the video frames
    pbar = tqdm(total=TOTAL_N_FRAMES, desc=video_path)
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        # frame number
        frame_n = cap.get(1)

        if success:
            if frame_n % (vid_stride + 1):
                pbar.update(1)
                continue

            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            result = model.track(frame, persist=True, tracker=tracker, iou=0.4, conf=0.4, verbose=False)

            if result[0].boxes.id is None:
                continue

            # Get the boxes and track IDs
            boxes = result[0].boxes.xywh.cpu()
            box_centers = torch.round(boxes[:, :2] + boxes[:, 2:] * torch.tensor([0, 0.2]))
            box_centers = box_centers.to(torch.int32).numpy()
            classes = result[0].boxes.cls.cpu()
            confs = result[0].boxes.conf.tolist()
            track_ids = result[0].boxes.id.int().cpu().tolist()

            # check if objects inside areas
            for id, cls, conf, center in zip(track_ids, classes, confs, box_centers):
                for i, area in enumerate(areas_list):
                    objects = objects_list[i]
                    seg1 = area[[0, 1]]
                    seg2 = area[[2, 3]]
                    par1 = area[[1, 2]]
                    par2 = area[[0, 3]]
                    mid_par = np.array([(area[0] + area[1]) // 2, (area[2] + area[3]) // 2])

                    # check for new ids
                    if id in objects:
                        # update class if required
                        if conf >= objects[id].conf:
                            objects[id].cls = cls
                            objects[id].conf = conf
                    else:
                        # create new object
                        objects[id] = SimpleNamespace(cls=cls, conf=conf, valid=False,
                                                      start_value=None, start_frame=None,
                                                      end_value=None, end_frame=None)

                    # check if point inside area
                    if lies_between(center, seg1, seg2):
                        # validate (check if car in zone)
                        if not objects[id].valid and lies_between(center, par1, par2):
                            objects[id].valid = True

                        # first enter
                        if objects[id].start_value is None:
                            objects[id].start_value = mid_projection(center, mid_par)
                            objects[id].start_frame = frame_n
                        # not first enter
                        elif objects[id].end_value is not None:
                            objects[id].end_value = None
                            objects[id].end_frame = None
                    # exit
                    elif objects[id].start_value is not None and objects[id].end_value is None:
                        objects[id].end_value = mid_projection(center, mid_par)
                        objects[id].end_frame = frame_n
            pbar.update(1)
        else:
            # Break the loop if the end of the video is reached
            break
    # counters
    # sort ids which present in many zones
    # {id: cls, dist, frames}
    objects_res = {}
    for objects in objects_list:
        for id, obj in objects.items():
            if obj.start_value is None or obj.end_value is None or not obj.valid:
                continue

            dist = abs(obj.end_value - obj.start_value)
            frames = obj.end_frame - obj.start_frame
            if id in objects_res:
                if objects_res[id].dist <= dist:
                    objects_res[id] = SimpleNamespace(cls=obj.cls, dist=dist, frames=frames)
            else:
                objects_res[id] = SimpleNamespace(cls=obj.cls, dist=dist, frames=frames)

    cars = [obj for obj in objects_res.values() if obj.cls == 1]
    car_count = len(cars)
    if car_count:
        kmpf = 0.02 * sum([x.dist / x.frames for x in cars]) / car_count
        # kmpf = 0.02 * sum([car.dist for car in cars]) / sum([car.frames for car in cars])
        car_speed = kmpf * fps * 3600
    else:
        car_speed = float('nan')

    buses = [obj for obj in objects_res.values() if obj.cls == 0]
    bus_count = len(buses)
    if bus_count:
        kmpf = 0.02 * sum([x.dist / x.frames for x in buses]) / bus_count

        # kmpf = 0.02 * sum([bus.dist for bus in buses]) / sum([bus.frames for bus in buses])
        bus_speed = kmpf * fps * 3600
    else:
        bus_speed = float('nan')

    vans = [obj for obj in objects_res.values() if obj.cls == 3]
    van_count = len(vans)
    if van_count:
        kmpf = 0.02 * sum([x.dist / x.frames for x in vans]) / van_count

        # kmpf = 0.02 * sum([van.dist for van in vans]) / sum([van.frames for van in vans])
        van_speed = kmpf * fps * 3600
    else:
        van_speed = float('nan')

    # Release the video capture object and close the display window
    cap.release()

    return SimpleNamespace(car_count=car_count, car_speed=car_speed,
                           van_count=van_count, van_speed=van_speed,
                           bus_count=bus_count, bus_speed=bus_speed)


def process_dir(model, vid_dir: str, markup_dir: str, csv_path: str):
    vid_dir = Path(vid_dir)
    markup_dir = Path(markup_dir)

    # create csv
    if not os.path.exists(csv_path):
        with open(csv_path, 'w') as f:
            f.write('file_name,car,quantity_car,average_speed_car,van,quantity_van,average_speed_van,bus,quantity_bus,average_speed_bus\n')

    # read existing lines
    existing = set()
    with open(csv_path, 'r') as f:
        for line in f.read().split('\n')[1:-1]:
            items = line.split(',')
            existing.add(items[0])

    for file in os.listdir(vid_dir):
        # check if already exists
        fname = file.split('.')[0]
        if fname in existing:
            continue

        # get paths
        vid_path = vid_dir / (fname + '.mp4')
        markup_path = markup_dir / (fname + '.json')

        # process
        res = process(model, str(vid_path), str(markup_path), vid_stride=2)
        f_res = [fname,
                 'car', str(res.car_count), str(round(res.car_speed, 2)),
                 'van', str(res.van_count), str(round(res.van_speed, 2)),
                 'bus', str(res.bus_count), str(round(res.bus_speed, 2))]

        # write line to csv
        with open(csv_path, 'a') as f:
            f.write(','.join(f_res) + '\n')


model = YOLO('models/yolov8s_2.pt')
# fname = 'KRA-2-7-2023-08-23-evening'
# res = process(model, f'videos/raw/CRF18/{fname}.mp4', f'videos/markup/{fname}.json', vid_stride=2)

process_dir(model, 'videos/raw/CRF18', 'videos/markup', 'videos/train_res.csv')
