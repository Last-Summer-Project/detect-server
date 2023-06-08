import os
import onnxruntime
import numpy as np
from PIL import Image

from utils.Image import image_to_numpy


class OnnxInference:
    def __init__(self):
        providers = ['CPUExecutionProvider']
        if os.getenv("ONNX_CUDA", False):
            providers.insert(0, 'CUDAExecutionProvider')

        # Load onnx inference session
        self.session = onnxruntime.InferenceSession(os.getenv("ONNX_MODEL"),
                                                    providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    # predict by image
    def predict_image(self, data: list[Image]):
        if len(data) == 0:
            return []
        return self.predict_array(np.stack([image_to_numpy(d) for d in data], axis=0))

    # predict by array
    def predict_array(self, data: np.ndarray):
        result = self.session.run([self.output_name], {self.input_name: data})
        prediction = np.argmax(np.array(result).squeeze(), -1)
        # wrapping up if it's only 1
        if isinstance(prediction, np.int64):
            return [prediction]
        return prediction
