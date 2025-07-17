# Weather Classification API

A FastAPI application that classifies weather conditions in images.

## Features

- Image upload and classification
- Weather condition prediction (Cloudy, Rain, Shine, Sunrise)
- Confidence scores for predictions

## Deployment

This application is configured for deployment on Render.

## API Endpoints

- `GET /`: Root endpoint, returns API status
- `GET /health`: Health check endpoint
- `POST /predict/`: Upload an image for weather classification