import io

import boto3
from mypy_boto3_s3 import Client
import logging
from dotenv import dotenv_values
from PIL import Image


class S3Client:
    client = None

    def __init__(self):
        env: dict = dotenv_values()
        self.client: Client = boto3.client(
            's3',
            endpoint_url=env.get("S3_ENDPOINT_URL"),
            aws_access_key_id=env.get("S3_ACCESS_KEY"),
            aws_secret_access_key=env.get("S3_SECRET_KEY"),
            config=boto3.session.Config(signature_version='s3v4'),
            verify=env.get("S3_VERIFY_TLS")
        )
        self.bucket = env.get("S3_BUCKET_NAME")

    def download_image(self, key: str) -> Image:
        logging.debug(f"Download image from s3 : {key}")
        stream = io.BytesIO()
        self.client.download_fileobj(self.bucket, key, stream)
        return Image.open(stream)

    def upload_image(self, key: str, image: Image, format: str = "jpg") -> Image:
        stream = io.BytesIO()
        image.save(stream, format)
        self.client.upload_fileobj(stream, self.bucket, key)
