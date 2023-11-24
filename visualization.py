import cv2
import json
import numpy as np
import torch

from ultralytics import YOLO
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


def visualize(model, video_path, markup_path, vid_stride=0):
    # tracking: {id: [class, conf, first_frame, last_frame]}
    objects = {}

    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    fontColor = (0, 255, 0)
    thickness = 2
    lineType = 2

    # open markup
    with open(markup_path) as f:
        markup = json.load(f)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        # frame number
        frame_n = cap.get(1)

        if frame_n % (vid_stride + 1):
            continue

        if success:
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            result = model.track(frame, persist=True)

            if result[0].boxes.id is None:
                continue

            # Get the boxes and track IDs
            boxes = result[0].boxes.xywh.cpu()
            box_centers = torch.round(boxes[:, :2] + boxes[:, 2:] * torch.tensor([0, 0.25]))
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
            w, h, d = annotated_frame.shape
            for id, cls, conf, center in zip(track_ids, classes, confs, box_centers):
                for area in markup['areas']:
                    areas = (np.array(area) * np.array([h, w])).astype(np.int32)
                    # check for new ids
                    if id in objects:
                        # update class if required
                        if conf >= objects[id][1]:
                            objects[id][:2] = cls, conf
                    else:
                        objects[id] = [cls, conf, None, None]

                    # check if point inside area
                    if Polygon(areas).contains(Point(center)):
                        # first enter
                        if objects[id][2] is None:
                            objects[id][2] = frame_n
                        # not first enter
                        elif objects[id][3] is not None:
                            objects[id][3] = None
                    # exit
                    elif objects[id][2] is not None and objects[id][3] is None:
                        objects[id][3] = frame_n

            # draw borders
            for area in markup['areas']:
                areas = (np.array(area) * np.array([h, w])).astype(np.int32)
                cv2.line(annotated_frame, areas[0], areas[1], (0, 0, 255), thickness=2)
                cv2.line(annotated_frame, areas[1], areas[2], (0, 255, 0), thickness=2)
                cv2.line(annotated_frame, areas[2], areas[3], (0, 0, 255), thickness=2)
                cv2.line(annotated_frame, areas[3], areas[0], (0, 255, 0), thickness=2)
                # cv2.polylines(annotated_frame, [areas], isClosed=True, color=(0, 0, 255), thickness=2)

            # counters
            car_frames = [f2 - f1 for cls, _, f1, f2 in objects.values() if cls == 2 and f1 is not None and f2 is not None]
            car_count = len(car_frames)
            if car_count:
                car_speed = sum([0.02 / (f / fps / 3600) for f in car_frames]) / len(car_frames)
            else:
                car_speed = float('nan')

            bus_frames = [f2 - f1 for cls, _, f1, f2 in objects.values() if cls == 1 and f1 is not None and f2 is not None]
            bus_count = len(bus_frames)
            if bus_count:
                bus_speed = sum([0.02 / (f / fps / 3600) for f in bus_frames]) / len(bus_frames)
            else:
                bus_speed = float('nan')

            van_frames = [f2 - f1 for cls, _, f1, f2 in objects.values() if cls == 4 and f1 is not None and f2 is not None]
            van_count = len(van_frames)
            if van_count:
                van_speed = sum([0.02 / (f / fps / 3600) for f in van_frames]) / len(van_frames)
            else:
                van_speed = float('nan')

            cv2.putText(annotated_frame, f'Cars: {car_count}',
                        (10, 25),
                        font,
                        fontScale,
                        fontColor,
                        thickness,
                        lineType)
            cv2.putText(annotated_frame, f'Buses: {bus_count}',
                        (10, 55),
                        font,
                        fontScale,
                        fontColor,
                        thickness,
                        lineType)
            cv2.putText(annotated_frame, f'Vans: {van_count}',
                        (10, 85),
                        font,
                        fontScale,
                        fontColor,
                        thickness,
                        lineType)
            cv2.putText(annotated_frame, f'Car speed: {car_speed:.3f}',
                        (10, 115),
                        font,
                        fontScale,
                        fontColor,
                        thickness,
                        lineType)
            cv2.putText(annotated_frame, f'Bus speed: {bus_speed:.3f}',
                        (10, 145),
                        font,
                        fontScale,
                        fontColor,
                        thickness,
                        lineType)
            cv2.putText(annotated_frame, f'Van speed: {van_speed:.3f}',
                        (10, 175),
                        font,
                        fontScale,
                        fontColor,
                        thickness,
                        lineType)

            # Display the annotated frame
            cv2.imshow("YOLOv8 Tracking", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()
    print(objects[25])


model = YOLO('models/yolov8s_1.pt')
fname = 'KRA-2-10-2023-09-11-morning'

visualize(model, f'videos/van.mp4', f'videos/markup/{fname}.json', vid_stride=0)
