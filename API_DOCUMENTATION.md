# AI CSV Parser Agent API Documentation

## Overview

The AI CSV Parser Agent API provides REST endpoints for intelligent CSV to JSON conversion using GPT-4o-mini. The API supports file upload, processing, batch operations, and configuration management.

## Base URL

```
http://localhost:8000/api/v1
```

## Authentication

Currently, no authentication is required. In production, implement proper API key authentication or OAuth2.

## Endpoints

### Health Check

#### GET /health

Check API health status.

**Response:**

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2024-01-01T00:00:00.000Z",
  "services": {
    "openai": "connected",
    "file_storage": "active",
    "processing": "ready"
  }
}
```

### File Management

#### POST /upload

Upload a CSV file for processing.

**Request:**

- Content-Type: `multipart/form-data`
- Body: CSV file

**Response:**

```json
{
  "success": true,
  "message": "File uploaded successfully",
  "file_id": "uuid-string",
  "filename": "test_cases.csv",
  "file_size": 1024
}
```

#### GET /files/{file_id}/info

Get information about an uploaded CSV file.

**Response:**

```json
{
  "success": true,
  "file_info": {
    "file_path": "/path/to/file.csv",
    "file_size": 1024,
    "encoding": "utf-8",
    "is_loaded": true,
    "shape": [10, 3]
  },
  "column_info": {
    "total_columns": 3,
    "total_rows": 10,
    "columns": ["Test Title", "Test Steps", "Expected Outcome"],
    "column_types": {...},
    "non_empty_counts": {...}
  },
  "sample_data": [
    {
      "Test Title": "Login Test",
      "Test Steps": "1. Navigate to login page 2. Enter credentials",
      "Expected Outcome": "User should be logged in"
    }
  ]
}
```

#### GET /files

List all uploaded files.

**Response:**

```json
[
  {
    "file_id": "uuid-string",
    "filename": "test_cases.csv",
    "file_size": 1024,
    "upload_time": "2024-01-01T00:00:00.000Z",
    "content_type": "text/csv",
    "status": "pending"
  }
]
```

#### DELETE /files/{file_id}

Delete an uploaded file.

**Response:**

```json
{
  "success": true,
  "message": "File deleted successfully"
}
```

### Processing

#### POST /process

Process a CSV file and convert to JSON.

**Request:**

```json
{
  "file_id": "uuid-string",
  "output_format": "json",
  "include_metadata": false,
  "sample_percentage": 0.2
}
```

**Response:**

```json
{
  "success": true,
  "message": "Processing started",
  "processing_id": "uuid-string",
  "status": "processing"
}
```

#### GET /process/{job_id}/status

Get the status of a processing job.

**Response:**

```json
{
  "processing_id": "uuid-string",
  "status": "completed",
  "progress": 100.0,
  "message": null,
  "results": {...},
  "error": null,
  "created_at": "2024-01-01T00:00:00.000Z",
  "completed_at": "2024-01-01T00:01:00.000Z"
}
```

#### GET /process/{job_id}/result

Get the result of a completed processing job.

**Response:**

```json
{
  "success": true,
  "data": [
    {
      "data": {
        "test_title": "login test",
        "steps": [
          "navigate to login page",
          "enter valid username",
          "enter valid password",
          "click login button"
        ],
        "expected_outcome": "user should be successfully logged in and redirected to dashboard"
      }
    }
  ],
  "total_objects": 1,
  "validation_results": {
    "total_objects": 1,
    "valid_objects": 1,
    "invalid_objects": 0,
    "errors": [],
    "warnings": []
  }
}
```

### Batch Processing

#### POST /process/batch

Process multiple CSV files in batch.

**Request:**

```json
{
  "file_ids": ["uuid-1", "uuid-2", "uuid-3"],
  "output_format": "json",
  "include_metadata": false,
  "sample_percentage": 0.2
}
```

**Response:**

```json
{
  "success": true,
  "message": "Batch processing started",
  "batch_id": "uuid-string",
  "total_files": 3,
  "processed_files": 0,
  "failed_files": 0
}
```

### Configuration

#### GET /config

Get current configuration.

**Response:**

```json
{
  "success": true,
  "configuration": {
    "csv_parser": {
      "sample_percentage": 0.2,
      "max_file_size": 10485760,
      "supported_encodings": ["utf-8", "latin1", "cp1252"],
      "output_format": "json"
    },
    "llm_settings": {
      "model": "gpt-4o-mini",
      "temperature": 0.1,
      "max_tokens": 2000,
      "timeout": 30,
      "retry_attempts": 3
    },
    "pattern_recognition": {
      "confidence_threshold": 0.85,
      "max_patterns": 10,
      "validation_sample_size": 50
    },
    "logging": {
      "level": "INFO",
      "file": "logs/csv_parser.log",
      "max_size": "10MB",
      "backup_count": 5
    }
  },
  "message": "Configuration retrieved successfully"
}
```

#### PUT /config

Update configuration.

**Request:**

```json
{
  "csv_parser": {
    "sample_percentage": 0.3
  },
  "llm_settings": {
    "temperature": 0.2
  }
}
```

**Response:**

```json
{
  "success": true,
  "message": "Configuration updated successfully"
}
```

## Error Responses

All endpoints may return error responses in the following format:

```json
{
  "success": false,
  "error": "Error message",
  "error_code": "ERROR_CODE",
  "details": {
    "additional": "information"
  }
}
```

### Common Error Codes

- `FILE_NOT_FOUND`: File with specified ID not found
- `FILE_UPLOAD_ERROR`: File upload failed
- `PROCESSING_ERROR`: CSV processing failed
- `CONFIGURATION_ERROR`: Configuration is invalid
- `OPENAI_ERROR`: OpenAI API error
- `VALIDATION_ERROR`: Request validation failed
- `RATE_LIMIT_ERROR`: Rate limit exceeded
- `INTERNAL_ERROR`: Internal server error

## Status Codes

- `200 OK`: Request successful
- `201 Created`: Resource created successfully
- `400 Bad Request`: Invalid request
- `404 Not Found`: Resource not found
- `422 Unprocessable Entity`: Validation error
- `500 Internal Server Error`: Server error

## Rate Limiting

Currently, no rate limiting is implemented. In production, implement rate limiting based on your requirements.

## File Size Limits

- Maximum file size: 10MB (configurable)
- Supported formats: CSV only
- Supported encodings: UTF-8, Latin1, CP1252

## Processing Status

- `pending`: Job created, waiting to start
- `processing`: Job is currently running
- `completed`: Job completed successfully
- `failed`: Job failed with error

## Examples

### Complete Workflow

1. **Upload a file:**

   ```bash
   curl -X POST "http://localhost:8000/api/v1/upload" \
        -H "Content-Type: multipart/form-data" \
        -F "file=@test_cases.csv"
   ```

2. **Process the file:**

   ```bash
   curl -X POST "http://localhost:8000/api/v1/process" \
        -H "Content-Type: application/json" \
        -d '{"file_id": "your-file-id"}'
   ```

3. **Check processing status:**

   ```bash
   curl "http://localhost:8000/api/v1/process/your-job-id/status"
   ```

4. **Get results:**
   ```bash
   curl "http://localhost:8000/api/v1/process/your-job-id/result"
   ```

## Interactive Documentation

Visit `/docs` for interactive Swagger UI documentation and `/redoc` for ReDoc documentation.

## Support

For issues and questions, check the logs at `logs/csv_parser.log` or contact the development team.
