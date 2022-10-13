FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8.12

COPY ./requirements.txt .

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY ./app /app/app