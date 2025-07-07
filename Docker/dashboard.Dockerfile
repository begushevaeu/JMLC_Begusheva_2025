# Base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy requirements and install dependencies
# The requirements.txt is in the root, so we go up one level from the Dockerfile context
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Copy the rest of the application code
COPY ./dashboard.py /app/dashboard.py
COPY ./shamlock_model /app/shamlock_model
COPY ./config.json /app/config.json
COPY ./app/neo4j_utils.py /app/app/neo4j_utils.py


# Expose the port the app runs on
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "dashboard.py"]
