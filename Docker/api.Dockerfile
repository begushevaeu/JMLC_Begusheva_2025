# Base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy requirements and install dependencies
# The requirements.txt is in the root, so we go up one level from the Dockerfile context
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Copy the rest of the application code
COPY ./app /app/app
COPY ./shamlock_model /app/shamlock_model
COPY ./config.json /app/config.json

# Copy Alembic configuration
COPY ./alembic.ini /app/alembic.ini
COPY ./alembic /app/alembic


# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
