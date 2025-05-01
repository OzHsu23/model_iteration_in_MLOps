FROM python:3.11-slim


WORKDIR /app

RUN pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app



#FastAPI need expose
EXPOSE 8000

CMD ["/bin/bash"]
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]