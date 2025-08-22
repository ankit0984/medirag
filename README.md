# MediPyRAG - Medical AI Assistant

A Retrieval-Augmented Generation (RAG) system for medical information queries using LangChain, AstraDB, and Google's Gemini AI.

## 🏥 Overview

MediPyRAG is an intelligent medical assistant that combines document retrieval with large language models to provide structured, accurate medical information. The system processes medical documents, stores them in a vector database, and uses semantic search to provide contextually relevant answers to medical queries.

## ✨ Features

- **Structured Medical Responses**: Returns organized medical information including definitions, symptoms, treatments, and more
- **Vector Database Storage**: Uses AstraDB for efficient document storage and retrieval
- **Semantic Search**: Leverages sentence transformers for intelligent document matching
- **Clean Output**: Automatically strips markdown formatting for clean, readable responses
- **Batch Processing**: Efficient document processing and storage with progress tracking

## 🛠️ Tech Stack

- **LangChain**: Framework for building LLM applications
- **AstraDB**: Vector database for document storage and retrieval
- **Google Gemini**: Large language model for response generation
- **Sentence Transformers**: For document embeddings
- **Pydantic**: Data validation and structured output
- **PyPDF**: PDF document processing

## 📋 Prerequisites

- Python 3.8+
- AstraDB account and database
- Google Gemini API key
- Virtual environment (recommended)

## 🚀 Installation

1. **Clone the repository**

   ```cmd
   git clone https://github.com/ankit0984/medirag.git
   cd medipyrag
   ```

2. **Create and activate virtual environment**

   ```cmd
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install dependencies**

   ```cmd
   pip install -r requirements.txt
   ```

4. **Set up environment variables**

   Create a `.env` file in the root directory:

   ```env
   ASTRADB_ENDPOINT=your_astradb_endpoint
   ASTRA_TOKEN=your_astra_token
   DB_COLLECTION_NAME=medical_docs
   DB_COLLECTION_NAMES=medical_docs
   GEMINI_KEY=your_gemini_api_key
   GEMINI_MODEL=gemini-pro
   ASTRA_NAMESPACE=default_keyspace
   ```

## 📁 Project Structure

```
medipyrag/
├── src/
│   ├── genllm.py          # Main LLM and response generation
│   ├── store.py           # Document storage and vector database
│   ├── autorag.py         # Auto retrieval functionality
│   └── db.py              # Database connection utilities
├── ragutils/              # RAG utility functions
├── data/                  # Medical documents (PDFs)
├── config.py              # Configuration management
├── main.py                # Main application entry point
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🏃‍♂️ Usage

1. **Prepare your medical documents**

   Place your PDF medical documents in the `data/` directory. The system expects a file named `medical_book_fixed.pdf` by default.

2. **Run the application**

   ```cmd
   python main.py
   ```

3. **First run setup**

   On the first run, the system will:

   - Process your PDF documents
   - Create embeddings
   - Store documents in AstraDB
   - Set up the retrieval system

4. **Ask medical questions**

   ```
   Enter your medical question: What is diabetes?
   ```

5. **Get structured responses**

   The system returns structured JSON responses with:

   - Definition
   - Causes & Risk Factors
   - Symptoms
   - Diagnosis methods
   - Treatment options
   - Prognosis & Complications
   - Prevention & Lifestyle advice
   - Additional notes

## 🔧 Configuration

### Model Settings

- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Chunk Size**: 1000 characters
- **Chunk Overlap**: 100 characters
- **Batch Size**: 100 documents per batch

### Customization

You can modify these settings in `config.py`:

```python
Model_name = "sentence-transformers/all-MiniLM-L6-v2"
Chunk_Size = 1000
Chunk_Overlap = 100
```


## 📊 Response Format

The system returns structured responses in this format:

```json
{
	"definition": "Brief definition of the condition",
	"causes_risk_factors": ["Risk factor 1", "Risk factor 2"],
	"symptoms": ["Symptom 1", "Symptom 2"],
	"diagnosis": ["Diagnostic method 1", "Diagnostic method 2"],
	"treatment_cure": ["Treatment 1", "Treatment 2"],
	"prognosis_complications": ["Prognosis info", "Complications"],
	"prevention_lifestyle": ["Prevention tip 1", "Lifestyle advice"],
	"additional_notes": ["Additional information"]
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This tool is for educational and informational purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical concerns.

## 🙏 Acknowledgments

- LangChain community for the excellent framework
- AstraDB for vector database capabilities
- Google for the Gemini AI model
- Sentence Transformers for embedding models

---