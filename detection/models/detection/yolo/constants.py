import os


SIZE = 416
CONFIG_DIR = os.path.dirname(__file__)
CONFIGS = {"yolov3": os.path.join(CONFIG_DIR, "yolov3.cfg"),
           "yolov3-tiny": os.path.join(CONFIG_DIR, "yolov3-tiny.cfg")}
