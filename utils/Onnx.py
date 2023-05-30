import onnx
import onnxruntime
import numpy as np
from PIL import Image
from dotenv import dotenv_values

from utils.Image import image_to_numpy


class OnnxInference:
    def __init__(self):
        env: dict = dotenv_values()
        self.session = onnxruntime.InferenceSession(env.get("ONNX_MODEL"),
                                                    providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict_image(self, data: list[Image]):
        if len(data) == 0:
            return []
        return self.predict_array(np.stack([image_to_numpy(d) for d in data], axis=0))

    def predict_array(self, data: np.ndarray):
        result = self.session.run([self.output_name], {self.input_name: data})
        prediction = np.argmax(np.array(result).squeeze(), -1)
        if isinstance(prediction, np.int64):
            return [prediction]
        return prediction
