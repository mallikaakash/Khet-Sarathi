# Khet Sarathi - The all-in-one Agentic Farm Assistant

Khet Sarathi is an intelligent WhatsApp-based farming assistant that helps Indian farmers with crop advisory, market prices, weather information, government schemes, and image analysis of Mandi receits, Legal documents, fertilizer component analysis ad many more using Sarvam AI and Gemini technologies.

## Features

### 🌾 Core Capabilities
- **Multi-language Support**: Hindi, Bengali, Tamil, Telugu, Kannada, Malayalam, Gujarati, Punjabi, Marathi, Urdu, and English
- **Voice & Text Communication**: Speech-to-text and text-to-speech in Indian languages
- **Image Analysis**: Fertilizer packets, soil test reports, market receipts, and crop photos
- **Market Intelligence**: Real-time mandi prices and MSP comparisons
- **Weather Advisory**: Location-based weather forecasts with farming recommendations
- **Government Schemes**: Information about PM-KISAN, crop insurance, and credit schemes

### 🤖 AI-Powered Features
- **Intent Classification**: Smart understanding of farmer queries
- **Image Processing**: Multi-step analysis of agricultural images
- **Contextual Responses**: Memory of previous conversations
- **Tool Integration**: Automated data fetching from various APIs

## Demo Video
  
[![Watch the video]()](https://youtu.be/ymTH8WDIFfA)


## Technology Stack

- **Backend**: FastAPI (Python)
- **AI/ML**: Google Gemini AI, Sarvam AI (TTS/STT)
- **Communication**: Twilio WhatsApp API
- **Audio Processing**: pydub, AudioSegment
- **Image Processing**: PIL, Google Gemini Vision
- **Database**: SQLite (with conversation memory)

## Prerequisites

Before deploying KrishiSarathi, ensure you have:

1. **Python 3.8+** installed
2. **API Keys** for the following services:
   - Twilio Account (for WhatsApp)
   - Sarvam AI API (for Indian language TTS/STT)
   - Google Gemini API (for AI responses and image analysis)
3. **ngrok** or similar tunneling service (for local development)
4. **Domain/Server** (for production deployment)

## Installation & Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd KhetSarathi
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root directory:

```bash
cp env.example .env
```

Edit the `.env` file with your actual API keys:

```env
# Twilio Configuration
TWILIO_ACCOUNT_SID=your_twilio_account_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_NUMBER=+1234567890

# Sarvam AI Configuration
SARVAM_API_KEY=your_sarvam_api_key

# Google Gemini Configuration
GEMINI_API_KEY=your_gemini_api_key

# Base URL (update for production)
BASE_URL=http://localhost:8000

# Optional configurations
DATABASE_PATH=farming_assistant.db
AUDIO_DIR=voice_responses
LOG_LEVEL=INFO
```

### 5. Create Required Directories

```bash
mkdir -p voice_responses
mkdir -p whatsapp
```

## API Keys Configuration

### Twilio Setup
1. Sign up at [Twilio Console](https://console.twilio.com/)
2. Create a new project
3. Enable WhatsApp Sandbox or get WhatsApp Business approval
4. Copy Account SID, Auth Token, and WhatsApp number

### Sarvam AI Setup
1. Register at [Sarvam AI](https://www.sarvam.ai/)
2. Get API key for text-to-speech and speech-to-text services
3. Ensure you have access to Indian language models

### Google Gemini Setup
1. Go to [Google AI Studio](https://makersuite.google.com/)
2. Create a new API key
3. Enable Gemini 1.5 Flash model access

## Local Development

### 1. Start the Application

```bash
python main.py
```

The application will start on `http://localhost:8000`

### 2. Setup ngrok (for Twilio webhooks)

```bash
# Install ngrok
npm install -g ngrok

# Create tunnel
ngrok http 8000
```

Copy the ngrok URL and update your `.env` file:
```env
BASE_URL=https://your-ngrok-url.ngrok.io
```

### 3. Configure Twilio Webhook

In your Twilio Console:
1. Go to WhatsApp Sandbox/Business settings
2. Set webhook URL: `https://your-ngrok-url.ngrok.io/webhook/message`
3. Set HTTP method to `POST`

### 4. Test the Application

- Send a WhatsApp message to your Twilio number
- Check logs for processing
- Test various endpoints at `http://localhost:8000/docs`


## Testing

### 1. API Testing

Visit `http://your-domain.com/docs` for interactive API documentation.

### 2. Test endpoints

```bash
# Test message processing
curl -X POST "http://your-domain.com/api/test/message" \
  -H "Content-Type: application/json" \
  -d '{
    "phone_number": "+919876543210",
    "message": "गेहूं का भाव क्या है?"
  }'

# Test TTS
curl -X POST "http://your-domain.com/api/test/text-to-speech" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "नमस्ते किसान भाई",
    "language": "hi-IN"
  }'
```

### 3. WhatsApp Testing

Send test messages to your WhatsApp number:
- Text: "गेहूं का भाव क्या है?"
- Voice message in Hindi
- Image of fertilizer packet


## Troubleshooting

### Common Issues

1. **Twilio Error 12300**: Invalid Content-Type for audio
   - Check audio file format and headers
   - Ensure BASE_URL is accessible

2. **TTS Not Working**: 
   - Verify Sarvam API key
   - Check API quotas and limits

3. **Gemini API Errors**:
   - Verify API key and quotas
   - Check model availability

4. **Audio File Issues**:
   - Ensure ffmpeg is installed
   - Check file permissions in voice_responses/
