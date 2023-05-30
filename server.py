from utils.Onnx import OnnxInference
from utils.DBClient import DBConn
from utils.S3Client import S3Client
from time import sleep
import logging
import sys


def setup_log():
    log_formatter = logging.Formatter("%(asctime)-15s %(levelname)-8s %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # global log level

    file_handler = logging.FileHandler("log.txt", mode="a+", encoding="utf-8")
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)


def main():
    setup_log()

    db = DBConn()
    s3 = S3Client()
    onnx = OnnxInference()

    logging.info("Hello, World!")

    while True:
        sec = 1
        process = db.get_non_detected()
        process = [(log_id, s3.download_image(url)) for (log_id, url) in process]
        pred = onnx.predict_image([img for (_, img) in process])
        process = [(log_id, result) for ((log_id, _), result) in zip(process, pred)]
        for log_id, result in process:
            result = str(result)
            logging.debug(f"log_id '{log_id}' result was '{result}'")
            db.update_detected(log_id, 'done', result)
        if len(process) == 0:
            sec = 30
            logging.debug(f"Increase sleeping time to {sec} seconds as there wasn't any log.")
        logging.info(f"Sleeping {sec} sec...")
        sleep(sec)


if __name__ == "__main__":
    sys.exit(main())
