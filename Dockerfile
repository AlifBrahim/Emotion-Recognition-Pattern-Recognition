# Use an official Python runtime as a parent image
FROM python:3.8-slim-buster

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir opencv_python keras pandas numpy imutils scikit_learn flask tensorflow

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run real_time_video.py when the container launches
CMD ["python", "real_time_video.py"]
