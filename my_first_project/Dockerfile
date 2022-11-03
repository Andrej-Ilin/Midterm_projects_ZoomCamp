FROM python:3.8-slim

RUN pip install pipenv

WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict.py","best_model.bin", "./"]

EXPOSE 9696

ENTRYPOINT ["waitress-serve", "--listen=*:9696", "predict:app"]

# ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]