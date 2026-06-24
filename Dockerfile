# 1. Use an official, lightweight Python image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy your requirement files first
COPY requirements.txt .
COPY requirements-dev.txt .

# 4. Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements-dev.txt

# 5. Copy the rest of your project files into the container
COPY . .

# 6. Tell Python exactly where to find the 'src' folder
ENV PYTHONPATH="/app/phase1/project_root"

# 7. Set the default command to run your tests
CMD ["python", "-m", "pytest", "phase1/project_root/tests/"]