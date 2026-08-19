# GreenCycle Garbage Classifier — Deployment

From-scratch CNN (dropout + heavy augmentation), served behind FastAPI,
with a simple drag-and-drop web UI, packaged for Docker.

## Run locally with Docker

    docker build -t garbage-classifier .
    docker run -p 8000:8000 garbage-classifier

Then open http://localhost:8000 in a browser to use the upload UI,
or call the API directly:

    curl -X POST -F "file=@item.jpg" http://localhost:8000/predict

## Run without Docker

    pip install -r requirements.txt
    uvicorn main:app --app-dir app --host 0.0.0.0 --port 8000
