# Use Node.js 20.11.1 image
FROM node:20.11.1

ENV BACKEND_URL=http://backend:3000
ENV FILE_DIRECTORY=/app/uploads

# Install tesseract
RUN apt update && \
    apt-get install -y tesseract-ocr=5.3.0-2
RUN tesseract -v

# Create the /app directory
RUN mkdir /app
RUN mkdir ${FILE_DIRECTORY}
WORKDIR /app
COPY ./src .

# Install dependencies
RUN npm install

# Start server
CMD [ "npm", "start" ]