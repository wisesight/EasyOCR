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
from collections import OrderedDict

from utils import CTCLabelConverter, group_text_box, get_image_list
from recognition import get_text
from model import Model


logger = logging.getLogger(__name__)


class EasyOCRHandler(object):
    def __init__(self):
        self.model = None
        self.mapping = None
        self.device = None
        self.initialized = False
        self.converter = None

    def initialize(self, ctx):
        properties = ctx.system_properties
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
        model_dir = properties.get("model_dir")

        # Read model serialize/pt file
        model_pt_path = os.path.join(model_dir, "thai.pth")
        # Read model definition file
        model_def_path = os.path.join(model_dir, "model.py")
        if not os.path.isfile(model_def_path):
            raise RuntimeError("Missing the model definition file")

        os.environ["LRU_CACHE_CAPACITY"] = "1"

        # TODO: clear this
        lang_list = ['th', 'en']
        self.character = '''¢£¤¥!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZกขคฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮฤเแโใไะาุูิีืึั่้๊๋็์ำํฺฯๆ0123456789๑๒๓๔๕๖๗๘๙'''
        separator_list = {
                'th': ['\xa2', '\xa3'],
                'en': ['\xa4', '\xa5']
            }
        dict_list = {}
        for lang in lang_list:
            dict_list[lang] = lang + ".txt"
        self.converter = CTCLabelConverter(self.character, separator_list, dict_list)
        num_class = len(self.converter.character)
        input_channel = 1
        output_channel = 512
        hidden_size = 512
        self.model = Model(input_channel, output_channel, hidden_size, num_class)
        if self.device == 'cpu':
            state_dict = torch.load(model_pt_path, map_location=self.device)
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                new_key = key[7:]
                new_state_dict[new_key] = value
            self.model.load_state_dict(new_state_dict, strict=True)
        else:
            self.model = torch.nn.DataParallel(self.model).to(self.device)
            self.model.load_state_dict(torch.load(model_pt_path, map_location=self.device))

        logger.debug('Model file {0} loaded successfully'.format(model_pt_path))
        self.initialized = True

    def preprocess(self, data):
        image = data[0].get("data")
        if image is None:
            image = data[0].get("body")
        boxes = json.loads(data[0].get("boxes").decode())
        boxes = np.array(boxes)

        nparr = np.frombuffer(image, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_cv_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        slope_ths = 0.1
        ycenter_ths = 0.5
        height_ths = 0.5
        width_ths = 0.5
        add_margin = 0.1
        imgH = 64

        horizontal_list, free_list = group_text_box(boxes, slope_ths, ycenter_ths, height_ths, width_ths, add_margin)
        image_list, max_width = get_image_list(horizontal_list, free_list, img_cv_grey, model_height = imgH)
        return image_list, max_width

    def inference(self, image_list, max_width):
        imgH = 64
        ignore_char = ''
        decoder = 'greedy'
        beamWidth = 5
        batch_size = 1
        contrast_ths = 0.1
        adjust_contrast = 0.5
        filter_ths = 0.003
        workers = 0
        result = get_text(self.character, imgH, int(max_width), self.model, self.converter, image_list, ignore_char, decoder, beamWidth, batch_size, contrast_ths, adjust_contrast, filter_ths, workers, self.device)
        logger.debug(f"inference: {result}")
        return result

    def postprocess(self, inference_output):
        merged_results = []
        for i in inference_output:
            merged_results.append(i[1])
        logger.debug(f"postprocess: {merged_results}")
        return [merged_results]


_service = EasyOCRHandler()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    image_list, max_width = _service.preprocess(data)
    data = _service.inference(image_list, max_width)
    data = _service.postprocess(data)
    return data