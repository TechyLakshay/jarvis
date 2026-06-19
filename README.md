# Voice AI Backend

Production-ready Python Voice AI backend powered by **LangChain** with multi-provider support for STT, LLM, and TTS services.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (HTML/JS)                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                         WebSocket / REST
                              │
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI Backend (Python)                    │
├─────────────────────────────────────────────────────────────────┤
│  ┌───────────────┐  ┌──────────────────┐  ┌─────────────────┐ │
│  │   STT Layer    │  │  LangChain LLM    │  │    TTS Layer     │ │
│  │ - OpenAI/Whisper│  │ - Anthropic/Claude│  │ - ElevenLabs     │ │
│  │ - Deepgram     │  │ - OpenAI/GPT      │  │ - OpenAI         │ │
│  │ - Google       │  │ - Cohere          │  │ - Azure          │ │
│  │ - AssemblyAI   │  │ - Google/Gemini   │  │ - Cartesia       │ │
│  │ - Azure        │  │ - Ollama (Local)  │  │ - Play.ht        │ │
│  └───────────────┘  └──────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Features

- **LangChain Integration**: Modern LLM orchestration with memory and tool support
- **Multi-Provider Support**: Easy switching between STT, LLM, and TTS providers
- **Ollama Support**: Run local LLMs without API keys (Llama, Mistral, etc.)
- **WebSocket & REST**: Both streaming and request-response modes
- **Conversation Memory**: Maintains context across interactions
- **Production Ready**: Proper error handling, logging, and configuration
- **Session Management**: Multi-user support with isolated sessions
- **LangSmith Tracing**: Optional debugging and monitoring via LangSmith

## Quick Start

### 1. Install Dependencies

```bash
# Activate virtual environment (if not already activated)
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API keys
```

**Minimum required (one each):**

```env
# LLM - Choose one
ANTHROPIC_API_KEY=sk-ant-xxx          # Recommended
# or
OPENAI_API_KEY=sk-proj-xxx

# STT - Usually same as LLM provider
# If using OpenAI for LLM, you already have Whisper STT

# TTS - Usually same as LLM provider
# If using OpenAI for LLM, you already have OpenAI TTS
```

### 3. Run the Server

```bash
python -m backend.main
```

The server will start at `http://localhost:8000`

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/providers` | GET | List available providers |
| `/api/providers/check` | POST | Check if specific providers are available |
| `/api/chat` | POST | Text-only chat (no audio) |
| `/api/ws` | WebSocket | Full voice pipeline |

## Configuration

### Providers

Set your preferred providers in `.env`:

```env
STT_PROVIDER=openai
LLM_PROVIDER=anthropic
TTS_PROVIDER=elevenlabs
```

### Available Providers

**STT (Speech-to-Text):**
- `openai` (Whisper)
- `deepgram`
- `google`
- `assemblyai`
- `azure`

**LLM (Large Language Model):**
- `anthropic` (Claude) - Recommended
- `openai` (GPT-4/GPT-3.5)
- `cohere` (Command R/R+)
- `google` (Gemini/Gemma)
- `openrouter` (Multi-model)

**TTS (Text-to-Speech):**
- `openai` (Fast, decent quality)
- `elevenlabs` (High quality)
- `azure` (Multiple voices)
- `cartesia` (Low latency)
- `playht` (Ultra realistic)

## Usage Examples

### WebSocket Client (JavaScript)

```javascript
const ws = new WebSocket('ws://localhost:8000/api/ws');

ws.onopen = () => {
  console.log('Connected to Voice AI');
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);

  switch (data.type) {
    case 'transcript':
      console.log('User said:', data.text);
      break;
    case 'response':
      console.log('AI responded:', data.text);
      break;
    case 'audio_complete':
      // Audio bytes were sent before this message
      break;
  }
};

// Send audio
ws.send(audioBytes);

// Send text
ws.send(JSON.stringify({
  type: 'text',
  message: 'Hello, Carvia!'
}));
```

### REST API (Python)

```python
import requests

# Text-only chat
response = requests.post('http://localhost:8000/api/chat', json={
    'message': 'What time is it?'
})

print(response.json()['response'])
```

## Getting API Keys

### Anthropic (Claude)
1. Go to https://console.anthropic.com/
2. Create account or sign in
3. Navigate to Settings → API Keys
4. Create new key

### OpenAI
1. Go to https://platform.openai.com/
2. Create account or sign in
3. Navigate to API Keys
4. Create new secret key

### ElevenLabs
1. Go to https://elevenlabs.io/
2. Create account or sign in
3. Navigate to Settings → API Keys
4. Copy your API key

### Deepgram
1. Go to https://console.deepgram.com/
2. Create account
3. Navigate to API Keys
4. Create new key

### Google AI
1. Go to https://makersuite.google.com/
2. Create account
3. Navigate to API Key
4. Create new key

## Directory Structure

```
voice ai/
├── backend/
│   ├── api/
│   │   └── routes.py           # API endpoints
│   ├── core/
│   │   └── config.py           # Configuration management
│   ├── providers/
│   │   ├── stt.py              # Speech-to-Text providers
│   │   ├── llm.py              # LLM providers
│   │   └── tts.py              # Text-to-Speech providers
│   ├── services/
│   │   └── voice_pipeline.py   # Voice pipeline orchestration
│   ├── utils/
│   │   └── logger.py           # Logging configuration
│   └── main.py                 # FastAPI application
├── venv/                       # Virtual environment
├── .env.example                # Example configuration
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Troubleshooting

### Import Errors

If you get import errors:
```bash
pip install -r requirements.txt --upgrade
```

### Provider Not Working

1. Check your API key is correct in `.env`
2. Verify the key has necessary permissions
3. Use `/api/providers/check` endpoint to test:
```bash
curl -X POST http://localhost:8000/api/providers/check \
  -H "Content-Type: application/json" \
  -d '{"stt_provider": "openai"}'
```

### WebSocket Connection Issues

1. Check firewall settings
2. Verify CORS origins in `.env`
3. Check browser console for errors

## License

MIT License - Feel free to use for personal or commercial projects.
