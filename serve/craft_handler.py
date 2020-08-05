import io
import logging
import numpy as np
import os
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import cv2
import json

from detection import test_net, copyStateDict
from craft import CRAFT

logger = logging.getLogger(__name__)


class CraftHandler(object):
    def __init__(self):
        self.model = None
        self.mapping = None
        self.device = None
        self.initialized = False

    def initialize(self, ctx):
        properties = ctx.system_properties
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
        model_dir = properties.get("model_dir")

        # Read model serialize/pt file
        model_pt_path = os.path.join(model_dir, "craft_mlt_25k.pth")
        # Read model definition file
        model_def_path = os.path.join(model_dir, "craft.py")
        if not os.path.isfile(model_def_path):
            raise RuntimeError("Missing the model definition file")

        self.model = CRAFT()
        if self.device == 'cpu':
            self.model.load_state_dict(copyStateDict(torch.load(model_pt_path, map_location=self.device)))
        else:
            self.model.load_state_dict(copyStateDict(torch.load(model_pt_path, map_location=self.device)))
            self.model = torch.nn.DataParallel(self.model).to(self.device)
            cudnn.benchmark = False
        self.model.eval()

        logger.debug('Model file {0} loaded successfully'.format(model_pt_path))
        self.initialized = True

    def preprocess(self, data):
        image = data[0].get("data")
        if image is None:
            image = data[0].get("body")

        nparr = np.frombuffer(image, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def inference(self, img, topk=5):
        result = []
        canvas_size = 2560
        mag_ratio = 1.
        text_threshold = 0.7
        link_threshold = 0.4
        low_text = 0.4
        poly = False
        bboxes, polys = test_net(canvas_size, mag_ratio, self.model, img, text_threshold, link_threshold, low_text, poly, self.device)
        for i, box in enumerate(polys):
            poly = np.array(box).astype(np.int32).reshape((-1))
            result.append(poly.tolist())
        logger.debug(result)
        return [result]

    def postprocess(self, inference_output):
        return inference_output


_service = CraftHandler()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    data = _service.preprocess(data)
    data = _service.inference(data)
    data = _service.postprocess(data)

    return data