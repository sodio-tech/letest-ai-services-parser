# AI CSV Parser Agent - Technical Specification & Implementation Guide

## 📋 Project Overview

### Problem Statement

The current backend logic treats CSV files with multiple test steps as a single step, causing significant business issues. This occurs because the system cannot reliably identify the delimiter patterns used in test steps (e.g., "1. , 2. , 3." vs "a. , b. , c." vs "i. , ii.").

### Solution

An AI-powered agent that uses GPT-4o-mini to intelligently analyze CSV files, identify test step delimiters, and transform the data into a structured JSON format for proper processing.

## 🤖 Why LLM-Based Approach?

### Advantages over Traditional Pattern Matching

1. **Superior Pattern Recognition**: GPT-4o-mini can understand context and identify patterns that traditional regex cannot handle
2. **Adaptability**: Automatically adapts to new delimiter formats without code changes
3. **Context Awareness**: Understands the semantic meaning of test steps, not just syntax
4. **Error Recovery**: Can intelligently handle malformed or inconsistent data
5. **Future-Proof**: Easily adapts to new CSV formats and requirements

### Cost-Effective Solution

- GPT-4o-mini is highly cost-effective for this use case
- Estimated cost: <$0.10 per CSV file
- No need for complex ML model training or maintenance
- Immediate deployment without training data requirements

## 🎯 Core Requirements

### Functional Requirements

1. **CSV Header Detection**: Use GPT-4o-mini to intelligently identify:

   - Test Steps column
   - Test Title column
   - Expected Outcome column

2. **LLM-Based Pattern Recognition**: Use GPT-4o-mini to analyze 20% random sample and identify delimiter patterns with high accuracy

3. **Data Transformation**: Convert CSV data into structured JSON format with LLM-validated step separation

4. **Advanced Pattern Support**: Handle any delimiter format through LLM understanding:
   - Numeric: "1. , 2. , 3."
   - Alphabetic: "a. , b. , c."
   - Roman numerals: "i. , ii. , iii."
   - Custom patterns: "Step 1:, Step 2:"
   - Mixed patterns: "1) , 2) , 3)"
   - Bullet points: "• , • , •"
   - Any other human-readable format

### Non-Functional Requirements

- **Accuracy**: >98% delimiter pattern recognition using LLM intelligence
- **Performance**: Process CSV files up to 10MB within 30 seconds (including API calls)
- **Reliability**: Handle malformed CSV data gracefully with LLM fallback strategies
- **Scalability**: Support batch processing with cost-effective API usage
- **Cost Efficiency**: <$0.10 per CSV file processing cost

## 🏗️ Technical Architecture

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CSV Reader    │───▶│ Pattern Analyzer│───▶│ JSON Generator  │
│                 │    │   (AI Engine)   │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Header Detector │    │ Sample Collector│    │ Data Validator  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Technology Stack

#### Core Technologies

- **Python 3.9+**: Primary programming language
- **Pandas**: CSV file handling and data manipulation
- **NumPy**: Numerical operations for pattern analysis
- **scikit-learn**: Machine learning for pattern classification

#### AI/ML Libraries

- **OpenAI API**: GPT-4o-mini for intelligent pattern recognition and text analysis
- **regex**: Advanced pattern matching for validation
- **json**: JSON data handling and validation

#### Data Processing

- **OpenPyXL**: Excel file support (if needed)
- **chardet**: Character encoding detection
- **validators**: Data validation utilities
- **requests**: HTTP client for OpenAI API calls

## 🔧 Implementation Guidelines

### Phase 1: Core Infrastructure Setup

#### 1.1 Project Structure

```
csv_parser_agent/
├── src/
│   ├── core/
│   │   ├── csv_reader.py
│   │   ├── header_detector.py
│   │   └── llm_analyzer.py
│   ├── ai/
│   │   ├── openai_client.py
│   │   ├── prompt_engineer.py
│   │   └── pattern_validator.py
│   ├── transformers/
│   │   ├── json_generator.py
│   │   └── data_validator.py
│   └── utils/
│       ├── logger.py
│       ├── config.py
│       └── env_manager.py
├── tests/
├── data/
│   ├── samples/
│   └── outputs/
├── requirements.txt
├── config.yaml
├── .env
└── main.py
```

#### 1.2 Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

#### 1.3 Dependencies (requirements.txt)

```
pandas>=1.5.0
numpy>=1.21.0
openai>=1.0.0
requests>=2.28.0
regex>=2022.7.9
chardet>=5.0.0
pyyaml>=6.0
python-dotenv>=1.0.0
pytest>=7.0.0
pytest-cov>=3.0.0
```

### Phase 2: Core Components Implementation

#### 2.1 CSV Reader Module

```python
# src/core/csv_reader.py
class CSVReader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.encoding = self._detect_encoding()
        self.df = self._load_csv()

    def _detect_encoding(self) -> str:
        """Detect file encoding using chardet"""
        pass

    def _load_csv(self) -> pd.DataFrame:
        """Load CSV with proper encoding and delimiter detection"""
        pass

    def get_sample_data(self, percentage: float = 0.2) -> pd.DataFrame:
        """Extract random sample of data for analysis"""
        pass
```

#### 2.2 Header Detector Module

```python
# src/core/header_detector.py
class HeaderDetector:
    def __init__(self, df: pd.DataFrame, llm_client):
        self.df = df
        self.llm_client = llm_client
        self.headers = self._analyze_headers()

    def _analyze_headers(self) -> dict:
        """Use LLM to identify test title, steps, and expected outcome columns"""
        pass

    def find_columns_with_llm(self) -> dict:
        """Use GPT-4o-mini to intelligently identify column purposes"""
        pass

    def validate_column_mapping(self, mapping: dict) -> bool:
        """Validate the LLM-identified column mapping"""
        pass
```

#### 2.3 LLM Analyzer Module

```python
# src/core/llm_analyzer.py
class LLMAnalyzer:
    def __init__(self, llm_client, sample_data: list):
        self.llm_client = llm_client
        self.sample_data = sample_data
        self.patterns = self._extract_patterns_with_llm()

    def _extract_patterns_with_llm(self) -> dict:
        """Use GPT-4o-mini to analyze and extract delimiter patterns"""
        pass

    def analyze_delimiter_patterns(self, text_samples: list) -> dict:
        """Use LLM to identify delimiter patterns in test step text"""
        pass

    def generate_regex_pattern(self, pattern_description: str) -> str:
        """Generate regex pattern based on LLM analysis"""
        pass

    def validate_pattern_with_llm(self, pattern: str, test_data: list) -> dict:
        """Use LLM to validate pattern accuracy and suggest improvements"""
        pass
```

### Phase 3: AI Engine Implementation

#### 3.1 OpenAI Client Module

```python
# src/ai/openai_client.py
class OpenAIClient:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"

    def analyze_text_patterns(self, text_samples: list) -> dict:
        """Use GPT-4o-mini to analyze delimiter patterns in text"""
        pass

    def identify_columns(self, headers: list, sample_rows: list) -> dict:
        """Use LLM to identify column purposes from headers and sample data"""
        pass

    def extract_steps_with_llm(self, text: str, pattern_info: dict) -> list:
        """Use LLM to intelligently extract individual steps"""
        pass

    def validate_extraction(self, original_text: str, extracted_steps: list) -> dict:
        """Use LLM to validate step extraction accuracy"""
        pass
```

#### 3.2 Prompt Engineering Module

```python
# src/ai/prompt_engineer.py
class PromptEngineer:
    def __init__(self):
        self.prompts = self._initialize_prompts()

    def _initialize_prompts(self) -> dict:
        """Initialize specialized prompts for different tasks"""
        pass

    def create_column_identification_prompt(self, headers: list, sample_data: list) -> str:
        """Create prompt for identifying column purposes"""
        pass

    def create_pattern_analysis_prompt(self, text_samples: list) -> str:
        """Create prompt for analyzing delimiter patterns"""
        pass

    def create_step_extraction_prompt(self, text: str, pattern_info: dict) -> str:
        """Create prompt for extracting individual steps"""
        pass

    def create_validation_prompt(self, original: str, extracted: list) -> str:
        """Create prompt for validating extraction accuracy"""
        pass
```

#### 3.3 LLM-Based Pattern Recognition Pipeline

1. **Data Preprocessing**: Clean and prepare text samples for LLM analysis
2. **LLM Analysis**: Use GPT-4o-mini to intelligently identify patterns
3. **Pattern Validation**: Use LLM to validate and refine identified patterns
4. **Step Extraction**: Use LLM to extract individual steps with high accuracy
5. **Quality Assurance**: Use LLM to validate extraction results

### Phase 4: Data Transformation

#### 4.1 JSON Generator Module

```python
# src/transformers/json_generator.py
class JSONGenerator:
    def __init__(self, csv_data: pd.DataFrame, headers: dict, pattern: str):
        self.csv_data = csv_data
        self.headers = headers
        self.pattern = pattern

    def transform_to_json(self) -> list:
        """Transform CSV data to JSON format"""
        pass

    def generate_output_format(self, row_data: dict) -> dict:
        """Generate the required JSON structure"""
        return {
            "data": {
                "test_title": row_data.get('title', ''),
                "steps": self._extract_steps(row_data.get('steps', '')),
                "expected_outcome": row_data.get('expected_outcome', '')
            }
        }

    def _extract_steps(self, steps_text: str) -> list:
        """Extract individual steps using identified pattern"""
        pass
```

## 🧪 Testing Strategy

### Unit Tests

- CSV reading functionality
- Header detection accuracy
- Pattern recognition algorithms
- JSON generation correctness

### Integration Tests

- End-to-end processing pipeline
- Error handling scenarios
- Performance benchmarks

### Test Data Requirements

- Various CSV formats and encodings
- Different delimiter patterns
- Edge cases (empty cells, malformed data)
- Large file performance tests

## 📊 Performance Metrics

### Accuracy Metrics

- **Pattern Recognition Accuracy**: >95%
- **Header Detection Accuracy**: >98%
- **Step Extraction Accuracy**: >97%

### Performance Metrics

- **Processing Time**: <30 seconds for 10MB files
- **Memory Usage**: <512MB peak memory
- **Throughput**: >100 rows/second

## 🚀 Deployment Strategy

### Development Environment

- Local development with sample CSV files
- Unit testing with pytest
- Code coverage >90%

### Production Deployment

- Docker containerization
- API endpoint for CSV processing
- Batch processing capabilities
- Error logging and monitoring

### Configuration Management

```yaml
# config.yaml
csv_parser:
  sample_percentage: 0.2
  max_file_size: 10485760 # 10MB
  supported_encodings: ["utf-8", "latin1", "cp1252"]
  output_format: "json"

llm_settings:
  model: "gpt-4o-mini"
  temperature: 0.1
  max_tokens: 2000
  timeout: 30
  retry_attempts: 3

pattern_recognition:
  confidence_threshold: 0.85
  max_patterns: 10
  validation_sample_size: 50

logging:
  level: "INFO"
  file: "logs/csv_parser.log"
  max_size: "10MB"
  backup_count: 5
```

```bash
# .env
OPENAI_API_KEY=your_openai_api_key_here
```

## 🔒 Error Handling & Validation

### Input Validation

- File format validation (CSV/Excel)
- File size limits
- Character encoding detection
- Malformed data handling

### Error Recovery

- Graceful degradation for unrecognized patterns
- Fallback to manual delimiter specification
- Detailed error logging and reporting

### Data Quality Checks

- Empty cell handling
- Inconsistent formatting detection
- Data type validation

## 📈 Future Enhancements

### Phase 2 Features

- Support for Excel files (.xlsx, .xls)
- Multi-language delimiter recognition
- Custom delimiter pattern learning
- Batch processing optimization

### Advanced AI Features

- Machine learning model training on user data
- Adaptive pattern recognition
- User feedback integration
- Continuous learning capabilities

## 🛠️ Development Timeline

### Week 1-2: Foundation

- Project setup and environment configuration
- OpenAI API integration and client setup
- Core CSV reading and LLM-based header detection
- Basic prompt engineering framework

### Week 3-4: LLM Engine

- GPT-4o-mini integration for pattern analysis
- Advanced prompt engineering for different tasks
- LLM-based step extraction implementation
- JSON generation module with LLM validation

### Week 5-6: Testing & Optimization

- Comprehensive testing suite with LLM validation
- Performance optimization for API calls
- Error handling and retry mechanisms
- Cost optimization strategies

### Week 7-8: Integration & Deployment

- API development with LLM integration
- Documentation completion
- Production deployment with API key management

## 📝 Success Criteria

### Technical Success

- ✅ >98% accuracy in delimiter pattern recognition using LLM
- ✅ Processing time <30 seconds for 10MB files (including API calls)
- ✅ Successful handling of 98% of test CSV files
- ✅ Zero data loss during transformation
- ✅ Cost-effective API usage (<$0.10 per CSV file)

### Business Success

- ✅ Elimination of multi-step processing issues
- ✅ Improved test case management efficiency
- ✅ Reduced manual intervention requirements
- ✅ Scalable solution for growing data volumes

---

_This specification serves as the comprehensive guide for implementing the AI CSV Parser Agent. Regular updates and iterations based on testing results and user feedback are recommended._
