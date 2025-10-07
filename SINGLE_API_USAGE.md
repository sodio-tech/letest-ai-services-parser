# 🚀 Single API Endpoint Usage Guide

## New Endpoint: `/api/v1/process-csv`

This new endpoint allows you to upload a CSV file and get processed JSON results in a **single request** - no need for multiple API calls or status polling!

---

## 📤 **Simple Usage**

### **Basic Request**
```bash
curl -X POST "http://localhost:8000/api/v1/process-csv" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_test_cases.csv"
```

### **Advanced Request with Options**
```bash
curl -X POST "http://localhost:8000/api/v1/process-csv" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_test_cases.csv" \
     -F "output_format=json" \
     -F "include_metadata=true" \
     -F "sample_percentage=0.2"
```

---

## 📥 **Response Format**

### **Success Response**
```json
{
  "success": true,
  "message": "CSV processed successfully",
  "processing_time": 8.3,
  "total_objects": 3,
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
  "validation_results": {
    "total_objects": 3,
    "valid_objects": 3,
    "invalid_objects": 0,
    "errors": [],
    "warnings": []
  },
  "pattern_info": {
    "pattern_type": "numeric_dot",
    "confidence": 0.95,
    "examples_found": ["1. Navigate", "2. Enter", "3. Click"]
  }
}
```

### **Error Response**
```json
{
  "success": false,
  "message": "CSV processing failed",
  "error": "File too large for single processing. Use /process endpoint for larger files.",
  "processing_time": 0.1
}
```

---

## ⚙️ **Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | File | Required | CSV file to process |
| `output_format` | String | "json" | Output format (currently only "json") |
| `include_metadata` | Boolean | false | Include processing metadata |
| `sample_percentage` | Float | 0.2 | Percentage of data to sample (0.1-1.0) |

---

## 🎯 **When to Use Single API vs Multi-Step API**

### **Use Single API (`/process-csv`) When:**
- ✅ File size < 10MB
- ✅ You want immediate results
- ✅ Simple integration needed
- ✅ No need for progress tracking

### **Use Multi-Step API (`/upload` → `/process` → `/status` → `/result`) When:**
- ✅ File size > 10MB
- ✅ You need progress tracking
- ✅ Background processing required
- ✅ Batch processing multiple files

---

## 🧪 **Testing**

### **Run the Test Script**
```bash
python test_single_api.py
```

### **Test with Your Own File**
```bash
curl -X POST "http://localhost:8000/api/v1/process-csv" \
     -F "file=@your_file.csv" \
     -F "include_metadata=true" | jq '.'
```

---

## 📊 **Performance**

| File Size | Processing Time | Cost |
|-----------|----------------|------|
| 1MB (100 rows) | ~5-10 seconds | ~$0.01 |
| 5MB (500 rows) | ~15-25 seconds | ~$0.05 |
| 10MB (1000 rows) | ~25-35 seconds | ~$0.10 |

---

## 🔧 **Integration Examples**

### **Python Integration**
```python
import requests

def process_csv_single(file_path):
    with open(file_path, 'rb') as f:
        files = {'file': f}
        data = {
            'output_format': 'json',
            'include_metadata': True,
            'sample_percentage': 0.2
        }
        
        response = requests.post(
            'http://localhost:8000/api/v1/process-csv',
            files=files,
            data=data,
            timeout=60
        )
        
        return response.json()

# Usage
result = process_csv_single('test_cases.csv')
if result['success']:
    print(f"Processed {result['total_objects']} objects")
    print(f"Processing time: {result['processing_time']} seconds")
    for item in result['data']:
        print(f"Test: {item['data']['test_title']}")
        print(f"Steps: {item['data']['steps']}")
```

### **JavaScript Integration**
```javascript
async function processCSV(file) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('include_metadata', 'true');
    formData.append('sample_percentage', '0.2');
    
    const response = await fetch('/api/v1/process-csv', {
        method: 'POST',
        body: formData
    });
    
    return await response.json();
}

// Usage
const fileInput = document.getElementById('csvFile');
const file = fileInput.files[0];

processCSV(file).then(result => {
    if (result.success) {
        console.log(`Processed ${result.total_objects} objects`);
        console.log(`Processing time: ${result.processing_time} seconds`);
        result.data.forEach(item => {
            console.log(`Test: ${item.data.test_title}`);
            console.log(`Steps: ${item.data.steps}`);
        });
    } else {
        console.error('Processing failed:', result.error);
    }
});
```

---

## 🎉 **Benefits**

### **For Users:**
- ✅ **One Request**: Upload and get results immediately
- ✅ **No Polling**: No need to check status repeatedly
- ✅ **Simple Integration**: Easy to implement
- ✅ **Fast Results**: Direct processing without delays

### **For Developers:**
- ✅ **Stateless**: No server-side state management
- ✅ **Synchronous**: Direct request-response pattern
- ✅ **Error Handling**: Comprehensive error responses
- ✅ **Validation**: Built-in input validation

---

## 🚨 **Limitations**

- **File Size**: Limited to 10MB files
- **Timeout**: 60-second request timeout
- **Memory**: Processes in memory (not suitable for very large files)
- **No Progress**: No progress tracking during processing

---

## 📚 **API Documentation**

Visit `http://localhost:8000/docs` for interactive API documentation with the new endpoint!
