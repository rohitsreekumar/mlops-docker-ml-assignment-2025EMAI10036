# Use official Python image
FROM python:3.10

# Set working directory inside container
WORKDIR /app

# Copy all files into container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run the ML experiment
CMD ["python", "palmer-panguin-decisionTree.py"]
