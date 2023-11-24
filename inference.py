from ultralytics import YOLO


def process(model, in_path, markup_path, out_path=None, show=False):
    for result in model.track('videos/short.mp4', save=out_path, show=show):
        pass


model = YOLO('models/yolov8s_1.pt')
fname = 'KRA-2-9-2023-08-22-evening'
process(model, f'videos/raw/{fname}.mp4', f'videos/markup/{fname}.json', show=True)
