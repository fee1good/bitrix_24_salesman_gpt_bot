# Use the official Python image as base
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Install Python dependencies
COPY ./app/requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY ./app/bot.py /app

# Run the application
ENTRYPOINT ["python", "bot.py"]