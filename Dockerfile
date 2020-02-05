FROM conda/miniconda3-centos7

RUN pip install --upgrade gtfs-realtime-bindings

RUN pip install protobuf jupyter pandas boto3 tables

RUN python -m ipykernel install --name realtime-buses --display-name "Python (realtime-buses)"

RUN mkdir /data

ENV BUS_BUCKET_NAME="bus350-data"

ENV HOME=/home

ENV PATH ~/.local/bin:$PATH

RUN pip install awscli --upgrade --user