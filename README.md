# 🤖 Agentic RAG with DeepSeek-R1

> **Intelligent Multi-Agent RAG System powered by DeepSeek-R1 and CrewAI**

Build a sophisticated RAG system that uses multiple AI agents to retrieve information from PDFs and the web, then synthesize coherent responses.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![CrewAI](https://img.shields.io/badge/CrewAI-Latest-green.svg)](https://www.crewai.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

![Agentic RAG Demo](assets/thumbnail.png)

---

## 🎯 What is Agentic RAG?

**Agentic RAG** combines the power of **Retrieval-Augmented Generation** with **Multi-Agent Systems**. Instead of a single retrieval step, multiple specialized AI agents work together to:

1. **Retrieve** relevant information from multiple sources
2. **Reason** about the best information to use
3. **Synthesize** a coherent, accurate response

### Why DeepSeek-R1?

**DeepSeek-R1** is a powerful reasoning model that excels at:
- Complex multi-step reasoning
- Understanding context deeply
- Making intelligent decisions
- Running locally with Ollama

---

## ✨ Features

### 🎭 Multi-Agent Architecture
- **Retriever Agent**: Intelligently searches PDFs and web
- **Synthesizer Agent**: Creates coherent responses
- **Sequential Processing**: Agents work in coordinated steps

### 📄 PDF Intelligence
- Upload any PDF document
- Automatic indexing with GroundX
- Semantic search capabilities
- Preview PDFs in-app

### 🌐 Web Search Integration
- Fallback to web search via Serper
- Real-time information retrieval
- Combines PDF and web knowledge

### 💬 Interactive Chat Interface
- Beautiful Streamlit UI
- Streaming responses
- Chat history
- Clear conversation option

### 🚀 Local & Private
- Runs DeepSeek-R1 locally via Ollama
- No data sent to external LLM APIs
- Complete privacy and control

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                  User Query                         │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│              Retriever Agent                        │
│  (DeepSeek-R1 via Ollama)                          │
│                                                     │
│  1. Try PDF Search Tool (GroundX)                  │
│  2. If not found, use Web Search (Serper)          │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│           Response Synthesizer Agent                │
│  (DeepSeek-R1 via Ollama)                          │
│                                                     │
│  Synthesizes coherent response from retrieved info  │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│              Final Answer                           │
└─────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

1. **Python 3.11+**
2. **Ollama** installed and running
3. **DeepSeek-R1 model** pulled in Ollama
4. **API Keys**:
   - GroundX API Key
   - Serper API Key

### Installation

```bash
# Clone the repository
git clone https://github.com/siddugarlapati/AGENTIC-RAG.git
cd AGENTIC-RAG

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Setup Ollama & DeepSeek-R1

```bash
# Install Ollama (if not already installed)
# Visit: https://ollama.ai/download

# Pull DeepSeek-R1 model
ollama pull deepseek-r1:7b

# Verify it's running
ollama list
```

### Configure API Keys

Edit `.env` file:

```env
GROUNDX_API_KEY=your_groundx_api_key_here
SERPER_API_KEY=your_serper_api_key_here
```

**Get API Keys:**
- **GroundX**: [https://www.groundx.ai/](https://www.groundx.ai/)
- **Serper**: [https://serper.dev/](https://serper.dev/)

### Run the Application

```bash
streamlit run app_deep_seek.py
```

The app will open at `http://localhost:8501`

---

## 📖 How to Use

### 1. Upload Your PDF

1. Click **"Choose a PDF file"** in the sidebar
2. Select your PDF document
3. Wait for indexing to complete
4. PDF preview will appear in sidebar

### 2. Ask Questions

1. Type your question in the chat input
2. The retriever agent will search the PDF first
3. If not found, it will search the web
4. The synthesizer agent creates the final response

### 3. View Responses

- Responses stream in real-time
- Sources are automatically cited
- Chat history is maintained
- Click "Clear Chat" to start fresh

---

## 💡 Example Use Cases

### 1. Research Paper Analysis
```
Upload: Research paper PDF
Ask: "What are the main findings of this study?"
Result: Detailed summary with key insights
```

### 2. Technical Documentation
```
Upload: API documentation PDF
Ask: "How do I authenticate API requests?"
Result: Step-by-step authentication guide
```

### 3. Legal Document Review
```
Upload: Contract PDF
Ask: "What are the termination clauses?"
Result: Extracted and explained clauses
```

### 4. Educational Content
```
Upload: Textbook chapter PDF
Ask: "Explain the concept of neural networks"
Result: Clear explanation with examples
```

---

## 🛠️ Technology Stack

### Core Framework
- **Streamlit**: Interactive web interface
- **CrewAI**: Multi-agent orchestration
- **LangChain**: RAG pipeline components

### AI Models
- **DeepSeek-R1 (7B)**: Local reasoning model via Ollama
- **GroundX**: PDF indexing and search
- **Serper**: Web search API

### Tools & Libraries
- **PyPDF2**: PDF processing
- **FAISS**: Vector similarity search
- **Sentence Transformers**: Text embeddings

---

## 📁 Project Structure

```
agentic_rag_deepseek/
├── app_deep_seek.py              # Main Streamlit application
├── src/
│   └── agentic_rag/
│       ├── tools/
│       │   └── custom_tool.py    # DocumentSearchTool implementation
│       ├── config/               # Configuration files
│       ├── crew.py              # Crew setup
│       └── main.py              # CLI entry point
├── knowledge/                    # Sample PDFs
│   └── dspy.pdf
├── assets/                       # Images and assets
│   ├── deep-seek.png
│   └── thumbnail.png
├── .env.example                  # Environment template
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## ⚙️ Configuration

### Agent Configuration

**Retriever Agent:**
- **Role**: Retrieve relevant information
- **Goal**: Find most relevant info from PDF or web
- **Tools**: DocumentSearchTool, SerperDevTool
- **Strategy**: Try PDF first, fallback to web

**Synthesizer Agent:**
- **Role**: Synthesize responses
- **Goal**: Create coherent answers
- **Tools**: None (uses retrieved context)
- **Strategy**: Combine and summarize information

### LLM Configuration

```python
llm = LLM(
    model="ollama/deepseek-r1:7b",
    base_url="http://localhost:11434"
)
```

**Customize:**
- Change model: `deepseek-r1:14b` for better quality
- Adjust temperature for creativity
- Modify max_tokens for longer responses

---

## 🎨 Customization

### Add More Agents

```python
# Add a fact-checker agent
fact_checker = Agent(
    role="Fact Checker",
    goal="Verify information accuracy",
    backstory="You're a meticulous fact-checker...",
    llm=load_llm()
)
```

### Add More Tools

```python
# Add calculator tool
from crewai_tools import CalculatorTool

calculator = CalculatorTool()
```

### Change Processing Order

```python
# Use hierarchical instead of sequential
crew = Crew(
    agents=[...],
    tasks=[...],
    process=Process.hierarchical  # Manager coordinates agents
)
```

---

## 🔧 Troubleshooting

### Ollama Connection Issues

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama
ollama serve
```

### Model Not Found

```bash
# Pull the model again
ollama pull deepseek-r1:7b

# List available models
ollama list
```

### PDF Indexing Fails

- Check GroundX API key is valid
- Ensure PDF is not corrupted
- Try a smaller PDF first
- Check internet connection

### Slow Responses

- Use smaller model: `deepseek-r1:1.5b`
- Reduce chunk size in DocumentSearchTool
- Limit number of retrieved documents
- Use GPU acceleration if available

---

## 📊 Performance

### Response Times
- **PDF Search**: 2-5 seconds
- **Web Search**: 3-7 seconds
- **Response Generation**: 5-10 seconds
- **Total**: 10-20 seconds per query

### Resource Usage
- **RAM**: 8-16 GB (depending on model)
- **CPU**: 4+ cores recommended
- **GPU**: Optional but recommended
- **Storage**: 5 GB for model

### Optimization Tips
1. Use GPU for faster inference
2. Cache frequently accessed documents
3. Reduce model size for speed
4. Batch similar queries

---

## 🔐 Security & Privacy

### Data Privacy
- ✅ All processing happens locally
- ✅ No data sent to external LLM APIs
- ✅ PDFs stored temporarily only
- ✅ Chat history in session only

### API Keys
- Store in `.env` file (not in code)
- Never commit `.env` to git
- Use environment variables
- Rotate keys regularly

### Best Practices
- Don't upload sensitive PDFs to public services
- Use local models for confidential data
- Clear chat history after sensitive queries
- Monitor API usage and costs

---

## 🚀 Deployment

### Local Deployment

```bash
# Run with custom port
streamlit run app_deep_seek.py --server.port 8080

# Run in background
nohup streamlit run app_deep_seek.py &
```

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Pull model
RUN ollama pull deepseek-r1:7b

CMD ["streamlit", "run", "app_deep_seek.py"]
```

### Cloud Deployment

**Streamlit Cloud:**
- Push to GitHub
- Connect to Streamlit Cloud
- Add secrets (API keys)
- Deploy

**AWS/GCP/Azure:**
- Use container service
- Ensure Ollama is installed
- Configure environment variables
- Set up load balancer

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file

---

## 🌟 Why Use This?

### For Researchers
- Analyze research papers quickly
- Extract key findings
- Compare multiple papers
- Generate summaries

### For Developers
- Query technical documentation
- Learn from code examples
- Understand APIs
- Debug issues

### For Students
- Study from textbooks
- Get explanations
- Prepare for exams
- Research topics

### For Businesses
- Analyze contracts
- Review documents
- Extract insights
- Automate Q&A

---

## 📚 Learn More

### About Agentic RAG
- [CrewAI Documentation](https://docs.crewai.com/)
- [LangChain RAG Guide](https://python.langchain.com/docs/use_cases/question_answering/)
- [Multi-Agent Systems](https://arxiv.org/abs/2308.08155)

### About DeepSeek-R1
- [DeepSeek-R1 Paper](https://arxiv.org/abs/2401.14196)
- [Ollama Documentation](https://ollama.ai/docs)
- [Model Comparison](https://ollama.ai/library/deepseek-r1)

### About RAG
- [RAG Survey Paper](https://arxiv.org/abs/2312.10997)
- [Advanced RAG Techniques](https://arxiv.org/abs/2312.10997)
- [RAG Best Practices](https://www.pinecone.io/learn/retrieval-augmented-generation/)

---

## 🎯 Roadmap

### Current (v1.0)
- ✅ Multi-agent RAG system
- ✅ PDF and web search
- ✅ DeepSeek-R1 integration
- ✅ Streamlit interface

### Coming Soon
- [ ] Support for multiple PDFs
- [ ] Advanced filtering options
- [ ] Export chat history
- [ ] Custom agent templates
- [ ] API endpoint
- [ ] Docker image
- [ ] Batch processing
- [ ] Analytics dashboard

---

## 💬 Support

- **Issues**: [GitHub Issues](https://github.com/siddugarlapati/AGENTIC-RAG/issues)
- **Discussions**: [GitHub Discussions](https://github.com/siddugarlapati/AGENTIC-RAG/discussions)
- **Repository**: [https://github.com/siddugarlapati/AGENTIC-RAG](https://github.com/siddugarlapati/AGENTIC-RAG)

---

## 🏆 Acknowledgments

- **DeepSeek** for the amazing R1 model
- **CrewAI** for the multi-agent framework
- **GroundX** for PDF indexing
- **Serper** for web search API
- **Ollama** for local model serving

---

<p align="center">
  
  <em>Making Agentic RAG accessible to everyone</em>
</p>

<p align="center">
  <a href="https://github.com/siddugarlapati/AGENTIC-RAG">⭐ Star on GitHub</a>
</p>

---

**Agentic RAG with DeepSeek-R1 - Intelligent Multi-Agent Document Q&A**
