FROM python:3.10-slim

RUN apt-get update && \
    apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    libgirepository1.0-dev \
    gobject-introspection \
    libdbus-1-dev \
    dbus \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /container

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

WORKDIR /container/GUI
CMD ["python", "app.py"]