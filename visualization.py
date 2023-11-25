import cv2
import json
import numpy as np
import torch

from ultralytics import YOLO
from types import SimpleNamespace
from utils import get_oriented_annotations, lies_between, mid_projection


def visualize(model, video_path, markup_path, vid_stride=0, tracker="custom.yaml", factor=0.9):
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    fontColor = (0, 255, 0)
    thickness = 2
    lineType = 2

    # open markup
    with open(markup_path) as f:
        markup = json.load(f)
    areas_list = get_oriented_annotations(markup)

    # tracking: [{id: [class, conf, first_frame, last_frame, valid]}]
    objects_list = [{} for _ in range(len(areas_list))]

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # w, h = 800, 800
    # cast area to int
    for i in range(len(areas_list)):
        areas_list[i] = (np.array(areas_list[i]) * np.array([w, h])).astype(np.int32)
    areas_list = areas_list.astype(np.int32)

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        # frame = cv2.resize(frame, (w, h))
        # frame number
        frame_n = cap.get(1)

        if frame_n % (vid_stride + 1):
            continue

        if success:
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            result = model.track(frame, persist=True, tracker=tracker, iou=0.4)

            if result[0].boxes.id is None:
                continue

            # Get the boxes and track IDs
            boxes = result[0].boxes.xywh.cpu()
            box_centers = torch.round(boxes[:, :2] + boxes[:, 2:] * torch.tensor([0, 0.2]))
            box_centers = box_centers.to(torch.int32).numpy()
            classes = result[0].boxes.cls.cpu()
            confs = result[0].boxes.conf.tolist()
            track_ids = result[0].boxes.id.int().cpu().tolist()

            # Visualize the results on the frame
            annotated_frame = result[0].plot()

            # draw centers
            for center in box_centers:
                cv2.circle(annotated_frame, center, 3, (0, 0, 255), thickness=-1)

            # check if objects inside areas
            for id, cls, conf, center in zip(track_ids, classes, confs, box_centers):
                for i, area in enumerate(areas_list):
                    objects = objects_list[i]
                    seg1 = area[[0, 1]]
                    seg2 = area[[2, 3]]
                    par1 = area[[1, 2]]
                    par2 = area[[0, 3]]
                    mid_par = np.array([(area[0] + area[1]) // 2, (area[2] + area[3]) // 2])

                    # debug
                    # cv2.putText(annotated_frame, f"{proj:.3f}", center, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

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

                    if objects[id].valid:
                        cv2.circle(annotated_frame, center, 5, (255, 0, 255), thickness=-1)

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

            # draw borders
            for area in areas_list:
                seg1 = area[[0, 1]]
                seg2 = area[[2, 3]]
                par1 = area[[1, 2]]
                par2 = area[[0, 3]]

                cv2.line(annotated_frame, seg1[0], seg1[1], (0, 0, 255), thickness=2)
                cv2.line(annotated_frame, par1[0], par1[1], (0, 255, 255), thickness=2)
                cv2.line(annotated_frame, seg2[0], seg2[1], (0, 0, 255), thickness=2)
                cv2.line(annotated_frame, par2[0], par2[1], (0, 255, 255), thickness=2)

                # cv2.line(annotated_frame, (areas[0] + areas[1]) // 2, (areas[2] + areas[3]) // 2, (255, 0, 0), thickness=2)

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

            # 2, 1, 4 for yolov8s_1.pt
            cars = [obj for obj in objects_res.values() if obj.cls == 2]
            car_count = len(cars)
            if car_count:
                kmpf = factor * 0.02 * sum([x.dist / x.frames for x in cars]) / car_count
                # kmpf = 0.02 * sum([car.dist for car in cars]) / sum([car.frames for car in cars])
                car_speed = kmpf * fps * 3600
            else:
                car_speed = float('nan')

            buses = [obj for obj in objects_res.values() if obj.cls == 1]
            bus_count = len(buses)
            if bus_count:
                kmpf = factor * 0.02 * sum([x.dist / x.frames for x in buses]) / bus_count

                # kmpf = 0.02 * sum([bus.dist for bus in buses]) / sum([bus.frames for bus in buses])
                bus_speed = kmpf * fps * 3600
            else:
                bus_speed = float('nan')

            vans = [obj for obj in objects_res.values() if obj.cls == 4]
            van_count = len(vans)
            if van_count:
                kmpf = factor * 0.02 * sum([x.dist / x.frames for x in vans]) / van_count

                # kmpf = 0.02 * sum([van.dist for van in vans]) / sum([van.frames for van in vans])
                van_speed = kmpf * fps * 3600
            else:
                van_speed = float('nan')

            cv2.putText(annotated_frame, f'Cars: {car_count}',
                        (10, 30),
                        font,
                        fontScale,
                        fontColor,
                        thickness,
                        lineType)
            cv2.putText(annotated_frame, f'Car speed: {car_speed:.3f}',
                        (10, 60),
                        font,
                        fontScale,
                        fontColor,
                        thickness,
                        lineType)

            cv2.putText(annotated_frame, f'Buses: {bus_count}',
                        (10, 90),
                        font,
                        fontScale,
                        fontColor,
                        thickness,
                        lineType)
            cv2.putText(annotated_frame, f'Bus speed: {bus_speed:.3f}',
                        (10, 120),
                        font,
                        fontScale,
                        fontColor,
                        thickness,
                        lineType)

            cv2.putText(annotated_frame, f'Vans: {van_count}',
                        (10, 150),
                        font,
                        fontScale,
                        fontColor,
                        thickness,
                        lineType)
            cv2.putText(annotated_frame, f'Van speed: {van_speed:.3f}',
                        (10, 180),
                        font,
                        fontScale,
                        fontColor,
                        thickness,
                        lineType)

            # Display the annotated frame
            cv2.imshow(markup_path, annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

    print(car_count, car_speed)
    print(van_count, van_speed)
    print(bus_count, bus_speed)


model = YOLO('models/yolov8s_1.pt')
fname = 'KRA-2-10-2023-09-11-morning'

res = visualize(model, f'videos/test/vids/{fname}.mp4', f'videos/markup/{fname}.json', vid_stride=0)
# visualize(model, f'videos/raw/output.mp4', f'videos/markup/{fname}.json', vid_stride=2)
