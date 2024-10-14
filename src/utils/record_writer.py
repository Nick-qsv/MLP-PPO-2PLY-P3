# custom/record_writer.py

import io
import boto3  # pylint: disable=import-error
import botocore  # pylint: disable=import-error
from tensorboardX import record_writer  # pylint: disable=import-error
from tensorboardX.record_writer import S3RecordWriter  # pylint: disable=import-error
from utils.decorators import profile

# Save the original RecordWriter
original_RecordWriter = record_writer.RecordWriter


class MyS3RecordWriter(S3RecordWriter):
    def __init__(self, logdir, *args, **kwargs):
        super(MyS3RecordWriter, self).__init__(logdir, *args, **kwargs)

    def flush(self):
        self.buffer.seek(0)
        try:
            self.s3.upload_fileobj(self.buffer, self.bucket, self.path)
        except botocore.exceptions.ClientError as e:
            print(f"S3 upload failed: {e}")
            # Optionally, log the error or take other action
        except Exception as e:
            print(f"Unexpected exception during S3 upload: {e}")
        finally:
            self.buffer.close()
            self.buffer = io.BytesIO()


@profile
def MyRecordWriter(logdir, filename_suffix=""):
    if logdir.startswith("s3://"):
        return MyS3RecordWriter(logdir)
    else:
        # Use the original RecordWriter for local directories
        return original_RecordWriter(logdir, filename_suffix)


# Monkey-patch the RecordWriter in tensorboardX
record_writer.RecordWriter = MyRecordWriter
