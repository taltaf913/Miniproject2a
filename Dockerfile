#FROM alpine:latest AS builder
FROM python:3.9

COPY dist/sentiment_analysis_model-0.0.1-py3-none-any.whl /etc/sentiment_analysis_model-0.0.1-py3-none-any.whl 
COPY sentiment_analysis_model_api /etc/sentiment_analysis_model_api
COPY sentiment_analysis_model/sentiment_analysis_tokenization0.0.1.json /etc/sentiment_analysis_model_api/sentiment_analysis_tokenization0.0.1.json

RUN pip3 install --upgrade pip

RUN pip3 install -r /etc/sentiment_analysis_model_api/requirements.txt
RUN pip3 install /etc/sentiment_analysis_model-0.0.1-py3-none-any.whl

EXPOSE 8000
ENV CONFIG_FILEPATH = config.yml

WORKDIR /etc/sentiment_analysis_model_api
CMD ["uvicorn" , "app.main:app",  "--host", "0.0.0.0", "--reload"]

