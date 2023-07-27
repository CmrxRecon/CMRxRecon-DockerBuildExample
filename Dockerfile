FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

COPY example /app
RUN pip install -r /app/requirements.txt

# DO NOT EDIT THE FLLOWING LINES
COPY *_run.py /
COPY docker-entrypoint.sh /
RUN chmod +x /docker-entrypoint.sh
ENTRYPOINT ["/bin/bash", "/docker-entrypoint.sh"]
