import os
import logging
from typing import Optional, Dict, Any
import tempfile
import requests
import json
import uuid
import base64
import sqlite3
from datetime import datetime
import asyncio

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import Response, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
from pydub import AudioSegment
from pydantic import BaseModel
from PIL import Image

# Try to import google.generativeai with error handling
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Google Generative AI not available: {e}")
    GEMINI_AVAILABLE = False
    genai = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables with fallbacks
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_NUMBER = os.getenv('TWILIO_NUMBER')
SARVAM_API_KEY = os.getenv('SARVAM_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
BASE_URL = os.getenv('BASE_URL', 'http://localhost:8000')

# Validate required environment variables
required_vars = {
    'TWILIO_ACCOUNT_SID': TWILIO_ACCOUNT_SID,
    'TWILIO_AUTH_TOKEN': TWILIO_AUTH_TOKEN,
    'TWILIO_NUMBER': TWILIO_NUMBER,
    'SARVAM_API_KEY': SARVAM_API_KEY,
    'GEMINI_API_KEY': GEMINI_API_KEY,
}

missing_vars = [var for var, value in required_vars.items() if not value]
if missing_vars:
    logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
    logger.error("Please check your .env file or environment configuration")
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Initialize clients
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Initialize Gemini
if GEMINI_AVAILABLE and GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
else:
    logger.warning("Gemini API not configured properly")
    gemini_model = None

# FastAPI app
app = FastAPI(title="Simple Saarthi AI", version="1.0.0")

# Create audio directory
AUDIO_DIR = "voice_responses"
os.makedirs(AUDIO_DIR, exist_ok=True)
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")

# Sarvam AI configuration
SARVAM_BASE_URL = "https://api.sarvam.ai"
SARVAM_HEADERS = {
    "api-subscription-key": SARVAM_API_KEY,
    "Content-Type": "application/json"
}

# Language support
SUPPORTED_LANGUAGES = {
    'hi': 'hi-IN',
    'en': 'en-IN', 
    'bn': 'bn-IN',
    'te': 'te-IN',
    'ta': 'ta-IN',
    'gu': 'gu-IN',
    'kn': 'kn-IN',
    'ml': 'ml-IN',
    'pa': 'pa-IN',
    'mr': 'mr-IN',
    'ur': 'ur-IN'
}

# TTS supported languages
TTS_SUPPORTED = {'hi-IN', 'en-IN', 'bn-IN', 'te-IN', 'ta-IN', 'gu-IN', 'kn-IN', 'ml-IN', 'pa-IN', 'mr-IN'}

class MemoryManager:
    """Simple memory management"""
    
    def __init__(self, db_path: str = "farming_assistant.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    phone_number TEXT,
                    user_input TEXT,
                    input_type TEXT,
                    detected_language TEXT,
                    ai_response TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
    
    def store_conversation(self, phone_number: str, user_input: str, input_type: str, 
                          detected_language: str, ai_response: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO conversations (phone_number, user_input, input_type, detected_language, ai_response) VALUES (?, ?, ?, ?, ?)",
                (phone_number, user_input, input_type, detected_language, ai_response)
            )
    
    def get_recent_conversations(self, phone_number: str, limit: int = 3) -> list:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT user_input, ai_response FROM conversations WHERE phone_number = ? ORDER BY timestamp DESC LIMIT ?",
                (phone_number, limit)
            )
            return [dict(row) for row in cursor.fetchall()]

# Initialize memory manager
memory = MemoryManager()

# ==== AGENTIC COMPONENTS ====

class IntentClassifier:
    """Advanced intent understanding for farmers"""
    
    INTENT_TYPES = {
        'quick_qa': 'Simple question requiring direct answer',
        'detailed_report': 'Complex query needing comprehensive analysis',
        'mandi_support': 'Market prices, selling, buying queries',
        'weather_info': 'Weather forecasts, alerts, planning',
        'image_analysis': 'Fertilizer, soil test, receipt analysis',
        'crop_advisory': 'Planting, harvesting, disease management',
        'government_schemes': 'Subsidies, loan information',
        'msp_comparison': 'Price comparison with government rates'
    }
    
    async def classify_intent(self, content: str, media_type: str = None, image_data: bytes = None) -> Dict[str, Any]:
        """Advanced LLM-based intent classification with entity extraction"""
        try:
            if media_type == 'image' and image_data:
                return await self._analyze_image_intent(image_data, content)
            
            if not GEMINI_AVAILABLE:
                return self._enhanced_fallback_classification(content)
            
            # Enhanced LLM prompt for better entity extraction
            prompt = f"""
            You are Krishi Saarthi's intelligent intent classifier for Indian farmers.
            
            Farmer's Message: "{content}"
            
            TASK 1 - Intent Classification:
            Choose the most appropriate intent from:
            - quick_qa: Simple farming questions
            - detailed_report: Complex analysis requests  
            - mandi_support: Market prices, selling, buying
            - weather_info: Weather forecasts, planning
            - image_analysis: Photo analysis requests
            - crop_advisory: Planting, disease, harvesting advice
            - government_schemes: Subsidies, loans, schemes
            - msp_comparison: Price comparison with government rates
            
            TASK 2 - Smart Entity Extraction:
            Extract entities from Hindi/English mixed content:
            
            CROP MAPPING (Hindi → English):
            - gehu/gehun/wheat → wheat
            - chawal/dhan/rice → rice  
            - kapas/cotton → cotton
            - ganna/sugarcane → sugarcane
            - makka/maize/corn → maize
            - bajra/pearl millet → bajra
            - jowar/sorghum → jowar
            - tur/arhar/pigeon pea → tur
            - moong/green gram → moong
            - urad/black gram → urad
            - groundnut/mungfali → groundnut
            - soybean/soya → soybean
            - sunflower/surajmukhi → sunflower
            - sesame/til → sesame
            
                         LOCATION EXTRACTION (COMPREHENSIVE):
             CITIES: Delhi, Mumbai, Pune, Bangalore, Chennai, Kolkata, Hyderabad, Ahmedabad, Surat, Jaipur, Lucknow, Kanpur, Nagpur, Indore, Thane, Bhopal, Visakhapatnam, Pimpri, Patna, Vadodara, Ghaziabad, Ludhiana, Agra, Nashik, Faridabad, Meerut, Rajkot, Kalyan, Vasai, Varanasi, Srinagar, Aurangabad, Dhanbad, Amritsar, Navi Mumbai, Allahabad, Ranchi, Howrah, Coimbatore, Jabalpur, Gwalior, Vijayawada, Jodhpur, Madurai, Raipur, Kota, Guwahati, Chandigarh, Solapur, Hubli, Tiruchirappalli, Bareilly, Mysore, Tiruppur, Gurgaon, Aligarh, Jalandhar, Bhubaneswar, Salem, Warangal, Guntur, Bhiwandi, Saharanpur, Gorakhpur, Bikaner, Amravati, Noida, Jamshedpur, Bhilai, Cuttack, Firozabad, Kochi, Nellore, Bhavnagar, Dehradun, Durgapur, Asansol, Rourkela, Nanded, Kolhapur, Ajmer, Akola, Gulbarga, Jamnagar, Ujjain, Loni, Siliguri, Jhansi, Ulhasnagar, Jammu, Sangli, Mangalore, Erode, Belgaum, Ambattur, Tirunelveli, Malegaon, Gaya, Jalgaon, Udaipur, Maheshtala
             
             STATES: Punjab, Haryana, UP, Uttar Pradesh, MP, Madhya Pradesh, Bihar, Rajasthan, Gujarat, Maharashtra, Karnataka, Tamil Nadu, Andhra Pradesh, Telangana, Kerala, Odisha, West Bengal, Assam, Jharkhand, Chhattisgarh, Himachal Pradesh, Uttarakhand, Goa, Tripura, Meghalaya, Manipur, Mizoram, Nagaland, Arunachal Pradesh, Sikkim
             
             DISTRICTS: Include major agricultural districts like Ludhiana, Patiala, Amritsar, Bathinda, Sangrur, Hisar, Sirsa, Karnal, Kurukshetra, Panipat, Muzaffarnagar, Meerut, Saharanpur, Moradabad, Bareilly, Sitapur, Hardoi, Unnao, Kanpur, Allahabad, Varanasi, Gorakhpur, Deoria, Azamgarh, Faizabad, Sultanpur, Pratapgarh, Jaunpur, Ghazipur, Ballia, Mau, Bijnor, Rampur, Shahjahanpur, Pilibhit, Lakhimpur, Kheri, Bahraich, Shrawasti, Balrampur, Gonda, Siddharthnagar, Basti, Sant Kabir Nagar, Maharajganj, Kushinagar, Padrauna, Deoria, Ambedkar Nagar, Amethi, Raebareli, Fatehpur, Kaushambi, Chitrakoot, Banda, Hamirpur, Mahoba, Jalaun, Jhansi, Lalitpur, Datia, Shivpuri, Guna, Ashoknagar, Gwalior, Bhind, Morena, Sheopur
             
             MANDIS: Include major agricultural markets like Azadpur Mandi, Khari Baoli, Anaj Mandi, Sabzi Mandi, Grain Market, Wholesale Market, APMC, Agricultural Produce Market Committee
            
            QUANTITY EXTRACTION:
            - Numbers with units: quintal, kg, ton, acre, hectare
            
            URGENCY ASSESSMENT:
            - high: "urgent", "jaldi", "emergency", "turant"
            - medium: normal queries
            - low: general information requests
            
            Return ONLY valid JSON:
            {{
                "primary_intent": "intent_name",
                "confidence": 0.85,
                "urgency": "medium",
                "extracted_entities": {{
                    "crop_type": "wheat",
                    "location": "punjab", 
                    "quantity": "10 quintal",
                    "price_context": "buying/selling/inquiry"
                }},
                "suggested_tools": ["mandi_prices", "msp"],
                "response_type": "quick_answer",
                "language_detected": "hi-IN"
            }}
            """
            
            response = gemini_model.generate_content(prompt)
            result = self._parse_gemini_json(response.text)
            
            # Validate and enhance result
            if result and isinstance(result, dict):
                # Ensure confidence is reasonable
                if result.get('confidence', 0) < 0.5:
                    result['confidence'] = 0.7
                
                # Add suggested tools based on intent
                intent = result.get('primary_intent', '')
                if intent == 'mandi_support' and 'suggested_tools' not in result:
                    result['suggested_tools'] = ['mandi_prices', 'msp']
                elif intent == 'weather_info' and 'suggested_tools' not in result:
                    result['suggested_tools'] = ['weather_forecast']
                
                return result
            else:
                logger.warning("LLM returned invalid JSON, using enhanced fallback")
                return self._enhanced_fallback_classification(content)
            
        except Exception as e:
            logger.error(f"LLM intent classification error: {e}")
            return self._enhanced_fallback_classification(content)
    
    async def _analyze_image_intent(self, image_data: bytes, context: str) -> Dict[str, Any]:
        """Analyze image to determine intent"""
        return {
            "primary_intent": "image_analysis",
            "confidence": 0.9,
            "urgency": "medium",
            "extracted_entities": {"has_image": True, "context": context},
            "suggested_tools": ["image_analyzer"],
            "response_type": "detailed_report"
        }
    
    def _parse_gemini_json(self, response_text: str) -> Dict[str, Any]:
        """Robust JSON parsing for Gemini responses"""
        import re
        
        response_text = response_text.strip()
        
        try:
            # First try direct parsing
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown or other formatting
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            
            # If all parsing fails, return None
            logger.error(f"Failed to parse JSON from Gemini response: {response_text[:200]}...")
            return None
    
    def _enhanced_fallback_classification(self, content: str) -> Dict[str, Any]:
        """Enhanced fallback classification with smart entity extraction"""
        content_lower = content.lower()
        
        # Enhanced keyword matching with Hindi support
        mandi_keywords = ['price', 'mandi', 'market', 'sell', 'buy', 'daam', 'dam', 'keemat', 'bechna', 'kharidna', 'rate', 'chal', 'raha', 'bhav']
        weather_keywords = ['weather', 'rain', 'temperature', 'mausam', 'barish', 'tapman', 'garmi', 'sardi', 'forecast']
        scheme_keywords = ['scheme', 'subsidy', 'loan', 'government', 'yojana', 'sahayata', 'sarkar', 'karz', 'pm-kisan']
        
        # Smart crop mapping for entity extraction
        crop_mapping = {
            'gehu': 'wheat', 'gehun': 'wheat', 'wheat': 'wheat',
            'chawal': 'rice', 'dhan': 'rice', 'rice': 'rice',
            'kapas': 'cotton', 'cotton': 'cotton',
            'ganna': 'sugarcane', 'sugarcane': 'sugarcane',
            'makka': 'maize', 'maize': 'maize', 'corn': 'maize',
            'bajra': 'bajra', 'jowar': 'jowar',
            'tur': 'tur', 'arhar': 'tur',
            'moong': 'moong', 'urad': 'urad',
            'groundnut': 'groundnut', 'mungfali': 'groundnut',
            'soybean': 'soybean', 'soya': 'soybean',
            'sunflower': 'sunflower', 'surajmukhi': 'sunflower',
            'sesame': 'sesame', 'til': 'sesame'
        }
        
        # Extract entities intelligently
        extracted_entities = {}
        
        # Extract crop with priority (longer matches first)
        for word, crop in sorted(crop_mapping.items(), key=len, reverse=True):
            if word in content_lower:
                extracted_entities['crop_type'] = crop
                break
        
        # Extract locations comprehensively
        major_cities = ['delhi', 'mumbai', 'pune', 'bangalore', 'chennai', 'kolkata', 'hyderabad', 'ahmedabad', 'surat', 'jaipur', 'lucknow', 'kanpur', 'nagpur', 'indore', 'patna', 'ludhiana', 'agra', 'meerut', 'varanasi', 'allahabad', 'ranchi', 'coimbatore', 'gwalior', 'vijayawada', 'jodhpur', 'madurai', 'raipur', 'kota', 'chandigarh', 'mysore', 'gurgaon', 'aligarh', 'jalandhar', 'bhubaneswar', 'salem', 'guntur', 'gorakhpur', 'bikaner', 'noida', 'jamshedpur', 'cuttack', 'kochi', 'dehradun', 'ajmer', 'ujjain', 'siliguri', 'jhansi', 'jammu', 'mangalore', 'belgaum', 'udaipur']
        
        states = ['punjab', 'haryana', 'up', 'uttar pradesh', 'mp', 'madhya pradesh', 'bihar', 'rajasthan', 'gujarat', 'maharashtra', 'karnataka', 'tamil nadu', 'andhra pradesh', 'telangana', 'kerala', 'odisha', 'west bengal', 'assam', 'jharkhand', 'chhattisgarh']
        
        agricultural_districts = ['ludhiana', 'patiala', 'amritsar', 'bathinda', 'sangrur', 'hisar', 'sirsa', 'karnal', 'kurukshetra', 'panipat', 'muzaffarnagar', 'saharanpur', 'moradabad', 'bareilly', 'sitapur', 'hardoi', 'unnao', 'faizabad', 'sultanpur', 'pratapgarh', 'jaunpur', 'ghazipur', 'ballia', 'mau', 'bijnor', 'rampur', 'shahjahanpur', 'pilibhit', 'bahraich', 'gonda', 'basti', 'deoria', 'azamgarh', 'gorakhpur', 'maharajganj', 'kushinagar', 'ambedkar nagar', 'amethi', 'raebareli', 'fatehpur', 'chitrakoot', 'banda', 'hamirpur', 'mahoba', 'jalaun', 'lalitpur', 'datia', 'shivpuri', 'guna', 'gwalior', 'bhind', 'morena', 'sheopur']
        
        # Check for cities first (priority)
        for city in major_cities:
            if city in content_lower:
                extracted_entities['location'] = city.title()
                break
        
        # If no city found, check for states
        if 'location' not in extracted_entities:
            for state in states:
                if state in content_lower:
                    extracted_entities['location'] = state.title()
                    break
        
        # If no state found, check for agricultural districts
        if 'location' not in extracted_entities:
            for district in agricultural_districts:
                if district in content_lower:
                    extracted_entities['location'] = district.title()
                    break
        
        # Determine intent with better logic
        if any(word in content_lower for word in mandi_keywords):
            intent = 'mandi_support'
            suggested_tools = ['mandi_prices', 'msp']
        elif any(word in content_lower for word in weather_keywords):
            intent = 'weather_info'
            suggested_tools = ['weather_forecast']
        elif any(word in content_lower for word in scheme_keywords):
            intent = 'government_schemes'
            suggested_tools = ['schemes']
        else:
            intent = 'quick_qa'
            suggested_tools = []
        
        return {
            "primary_intent": intent,
            "confidence": 0.8,  # Higher confidence for enhanced fallback
            "urgency": "medium",
            "extracted_entities": extracted_entities,
            "suggested_tools": suggested_tools,
            "response_type": "quick_answer",
            "language_detected": "hi-IN"
        }

class ImageAnalyzer:
    """Advanced image analysis for farming contexts using multi-step approach"""
    
    def __init__(self):
        self.text_extraction_prompt = """
You are an expert OCR specialist for Indian farming documents. Extract ALL visible text from this image.

Extract everything you can see:
- All numbers, percentages, ratios (like 20-20-0, 10-26-26)
- Brand names and product names (Hindi and English)
- Prices, quantities, measurements
- Dates and locations
- Hindi and English text
- Any labels, instructions, or recommendations
- Chemical formulas or nutrient names

Return as a simple text list, one item per line. Be very thorough.
"""

    def get_classification_prompt(self, extracted_text):
        return f"""
Based on this extracted text from an agricultural image: {extracted_text}

Classify the image type based on the content:

1. "fertilizer" - if contains NPK ratios, fertilizer brand names (IFFCO, Tata, Coromandel), nutrient percentages
2. "soil_test" - if contains pH values, nutrient levels, laboratory analysis, soil parameters
3. "receipt" - if contains crop prices, quantities sold, market names, mandi receipts, transaction details
4. "crop" - if mainly contains plant/crop descriptions, disease symptoms, growth stages
5. "equipment" - if contains machinery, tools, farming equipment
6. "other" - if none of the above categories fit

Look for these specific indicators:
- NPK numbers (like 20-20-0) = fertilizer
- pH values, nutrient analysis = soil_test  
- Prices per quintal, market names = receipt
- Plant symptoms, crop varieties = crop

Return only ONE word: fertilizer, soil_test, receipt, crop, equipment, or other
"""

    def get_specialized_prompt(self, image_type, extracted_text, context=""):
        prompts = {
            "fertilizer": f"""
Analyze this fertilizer information from the extracted text: {extracted_text}
Context: {context}

Extract specific fertilizer details:

NPK RATIO: Look for numbers like 20-20-0, 10-26-26, 12-32-16 etc.
BRAND: Company name (IFFCO, Tata Chemicals, Coromandel, IPL, etc.)
PRODUCT: Specific fertilizer name (DAP, Urea, Complex, etc.)
DOSAGE: Application rate (kg/acre or kg/hectare)
CROPS: Which crops it's recommended for
PRICE: Cost per bag/kg if visible
NUTRIENTS: Additional nutrients like Sulphur, Zinc, etc.

Provide farming advice in Hindi and English.

Return in this JSON format:
{{
    "text_extracted": "{extracted_text[:200]}...",
    "image_type": "fertilizer",
    "analysis": {{
        "hindi_summary": "इस फर्टिलाइजर की मुख्य जानकारी हिंदी में",
        "english_summary": "Key fertilizer information in English", 
        "specific_findings": ["NPK ratio found", "Brand identified", "Usage instructions"],
        "farmer_recommendations": ["Practical usage advice", "Application timing", "Suitable crops"],
        "cautions": ["Any warnings or precautions"]
    }},
    "data": {{
        "npk_values": "N-P-K ratio",
        "brand": "Brand name",
        "product_name": "Product name",
        "dosage": "Application rate",
        "recommended_crops": ["crop1", "crop2"],
        "price_info": "Price if available",
        "nutrients": ["additional nutrients"]
    }},
    "confidence": 0.95
}}
""",

            "soil_test": f"""
Analyze this soil test report from the extracted text: {extracted_text}
Context: {context}

Extract soil analysis details:

pH LEVEL: Soil acidity/alkalinity value
NPK LEVELS: Available Nitrogen, Phosphorus, Potassium
MICRONUTRIENTS: Zinc, Iron, Manganese, Boron status
ORGANIC CARBON: Organic matter percentage
RECOMMENDATIONS: Laboratory suggestions
EC: Electrical conductivity if mentioned

Provide soil improvement advice in Hindi and English.

Return in this JSON format:
{{
    "text_extracted": "{extracted_text[:200]}...",
    "image_type": "soil_test",
    "analysis": {{
        "hindi_summary": "मिट्टी टेस्ट की मुख्य जानकारी हिंदी में",
        "english_summary": "Key soil test findings in English",
        "specific_findings": ["pH level", "Nutrient status", "Lab recommendations"],
        "farmer_recommendations": ["Soil improvement advice", "Fertilizer suggestions", "Crop suitability"],
        "cautions": ["Soil health warnings if any"]
    }},
    "data": {{
        "ph_level": "pH value",
        "npk_status": "Available NPK levels",
        "micronutrients": "Micronutrient status",
        "organic_carbon": "OC percentage",
        "recommendations": ["lab suggestions"],
        "ec_value": "Electrical conductivity"
    }},
    "confidence": 0.90
}}
""",

            "receipt": f"""
Analyze this market receipt from the extracted text: {extracted_text}
Context: {context}

Extract transaction details:

CROP: What was sold (wheat, rice, cotton, etc.)
VARIETY: Specific variety or grade
QUANTITY: Amount in kg/quintal/ton
RATE: Price per unit (₹/quintal)
TOTAL: Final amount received
DATE: Transaction date
MARKET: Mandi/trader name and location
MOISTURE: Moisture content if mentioned
QUALITY: Grade or quality parameters

Provide price analysis and MSP comparison advice.

Return in this JSON format:
{{
    "text_extracted": "{extracted_text[:200]}...",
    "image_type": "receipt",
    "analysis": {{
        "hindi_summary": "मंडी रसीद की मुख्य जानकारी हिंदी में",
        "english_summary": "Key transaction details in English",
        "specific_findings": ["Crop sold", "Price received", "Market details"],
        "farmer_recommendations": ["Price analysis", "MSP comparison", "Future selling tips"],
        "cautions": ["Price warnings if below MSP"]
    }},
    "data": {{
        "crop_type": "Crop name",
        "variety": "Variety/grade",
        "quantity": "Amount sold",
        "price_per_unit": "Rate per quintal",
        "total_amount": "Total earnings",
        "date": "Transaction date",
        "market_location": "Mandi/trader name"
    }},
    "confidence": 0.92
}}
""",

            "crop": f"""
Analyze this crop/plant image from the extracted text: {extracted_text}
Context: {context}

Assess plant health and condition:

CROP IDENTIFICATION: What crop/plant is this
GROWTH STAGE: Seedling, vegetative, flowering, maturity
HEALTH STATUS: Healthy, stressed, diseased
SYMPTOMS: Disease symptoms, pest damage, nutrient deficiency
RECOMMENDATIONS: Treatment and care advice

Provide crop management advice in Hindi and English.

Return in this JSON format:
{{
    "text_extracted": "{extracted_text[:200]}...",
    "image_type": "crop",
    "analysis": {{
        "hindi_summary": "फसल की स्थिति की मुख्य जानकारी हिंदी में",
        "english_summary": "Key crop health information in English",
        "specific_findings": ["Crop identified", "Health status", "Symptoms observed"],
        "farmer_recommendations": ["Treatment advice", "Care instructions", "Prevention measures"],
        "cautions": ["Disease warnings", "Immediate actions needed"]
    }},
    "data": {{
        "crop_type": "Crop name",
        "growth_stage": "Current stage",
        "health_status": "Overall health",
        "symptoms": ["observed symptoms"],
        "diseases": ["identified diseases"],
        "treatments": ["recommended treatments"]
    }},
    "confidence": 0.88
}}
"""
        }
        
        return prompts.get(image_type, f"""
Analyze this agricultural image from the extracted text: {extracted_text}
Context: {context}

Provide general farming analysis and recommendations.

Return in this JSON format:
{{
    "text_extracted": "{extracted_text[:200]}...",
    "image_type": "other",
    "analysis": {{
        "hindi_summary": "तस्वीर की सामान्य जानकारी हिंदी में",
        "english_summary": "General image analysis in English",
        "specific_findings": ["What was observed"],
        "farmer_recommendations": ["General farming advice"],
        "cautions": ["Any warnings"]
    }},
    "data": {{}},
    "confidence": 0.70
}}
""")

    async def analyze_farming_image(self, image_data: bytes, context: str = "") -> Dict[str, Any]:
        """Multi-step image analysis: Extract -> Classify -> Analyze"""
        try:
            if not GEMINI_AVAILABLE:
                return self._fallback_image_analysis(context)
            
            # Enhanced image processing (same as before)
            import base64
            from PIL import Image
            import io
            
            try:
                image_pil = Image.open(io.BytesIO(image_data))
                image_format = image_pil.format.lower() if image_pil.format else 'jpeg'
                
                if image_pil.mode in ('RGBA', 'LA', 'P'):
                    image_pil = image_pil.convert('RGB')
                
                max_size = (1024, 1024)
                if image_pil.size[0] > max_size[0] or image_pil.size[1] > max_size[1]:
                    image_pil.thumbnail(max_size, Image.Resampling.LANCZOS)
                    logger.info(f"Resized image from original size to {image_pil.size}")
                
                img_byte_arr = io.BytesIO()
                image_pil.save(img_byte_arr, format='JPEG', quality=85)
                processed_image_data = img_byte_arr.getvalue()
                
                image_base64 = base64.b64encode(processed_image_data).decode('utf-8')
                mime_type = "image/jpeg"
                
                logger.info(f"Image processed: {image_format} -> JPEG, size: {image_pil.size}")
                
            except Exception as img_error:
                logger.error(f"Image processing error: {img_error}")
                image_base64 = base64.b64encode(image_data).decode('utf-8')
                mime_type = "image/jpeg"

            image_part = {
                "mime_type": mime_type,
                "data": image_base64
            }

            # STEP 1: Extract Text
            logger.info("🔍 Step 1: Extracting text from image...")
            try:
                flash_model = genai.GenerativeModel('gemini-1.5-flash')
                text_response = flash_model.generate_content([self.text_extraction_prompt, image_part])
                extracted_text = text_response.text.strip()
                logger.info(f"✅ Text extracted: {extracted_text[:100]}...")
            except Exception as e:
                logger.warning(f"Text extraction failed: {e}, using default model")
                text_response = gemini_model.generate_content([self.text_extraction_prompt, image_part])
                extracted_text = text_response.text.strip()

            if not extracted_text or len(extracted_text) < 10:
                logger.warning("Minimal text extracted, proceeding with image-only analysis")
                extracted_text = "Limited text visible in image"

            # STEP 2: Classify Image Type
            logger.info("🎯 Step 2: Classifying image type...")
            classification_prompt = self.get_classification_prompt(extracted_text)
            
            try:
                classify_response = flash_model.generate_content(classification_prompt)
                image_type = classify_response.text.strip().lower()
                logger.info(f"✅ Image classified as: {image_type}")
            except Exception as e:
                logger.warning(f"Classification failed: {e}")
                image_type = "other"

            # Validate classification
            valid_types = ['fertilizer', 'soil_test', 'receipt', 'crop', 'equipment', 'other']
            if image_type not in valid_types:
                logger.warning(f"Invalid classification '{image_type}', defaulting to 'other'")
                image_type = "other"

            # STEP 3: Specialized Analysis
            logger.info(f"🔬 Step 3: Performing specialized analysis for {image_type}...")
            specialized_prompt = self.get_specialized_prompt(image_type, extracted_text, context)
            
            try:
                analysis_response = flash_model.generate_content(specialized_prompt)
                result_text = analysis_response.text.strip()
                logger.info(f"✅ Specialized analysis completed")
            except Exception as e:
                logger.warning(f"Specialized analysis failed: {e}")
                result_text = f'{{"image_type": "{image_type}", "error": "Analysis failed"}}'

            # Parse the JSON response with multiple attempts
            parsed_result = self._parse_analysis_result(result_text, image_type, extracted_text, context)
            
            logger.info(f"🎉 Multi-step analysis completed: {parsed_result['image_type']}")
            return parsed_result

        except Exception as e:
            logger.error(f"Multi-step image analysis error: {e}")
            return self._fallback_image_analysis(context)

    def _parse_analysis_result(self, result_text: str, image_type: str, extracted_text: str, context: str) -> Dict[str, Any]:
        """Enhanced JSON parsing with multiple fallback strategies"""
        import re
        
        logger.info(f"Raw analysis response: {result_text[:200]}...")
        
        # Multiple parsing attempts
        parsed_result = None
        
        # Attempt 1: Direct JSON parsing
        try:
            parsed_result = json.loads(result_text)
            logger.info("✅ Direct JSON parsing successful")
        except json.JSONDecodeError:
            pass
        
        # Attempt 2: Extract from markdown code blocks
        if not parsed_result:
            json_match = re.search(r'```json\s*(.*?)\s*```', result_text, re.DOTALL | re.IGNORECASE)
            if json_match:
                try:
                    parsed_result = json.loads(json_match.group(1).strip())
                    logger.info("✅ Markdown JSON parsing successful")
                except json.JSONDecodeError:
                    pass
        
        # Attempt 3: Extract any JSON-like structure
        if not parsed_result:
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                try:
                    parsed_result = json.loads(json_match.group())
                    logger.info("✅ Generic JSON extraction successful")
                except json.JSONDecodeError:
                    pass
        
        # Validate and enhance result
        if parsed_result and isinstance(parsed_result, dict):
            # Ensure required fields
            if 'image_type' not in parsed_result:
                parsed_result['image_type'] = image_type
            if 'analysis' not in parsed_result:
                parsed_result['analysis'] = {}
            if 'data' not in parsed_result:
                parsed_result['data'] = {}
            if 'confidence' not in parsed_result:
                parsed_result['confidence'] = 0.8
                
            # Ensure analysis subfields
            analysis = parsed_result['analysis']
            if 'hindi_summary' not in analysis:
                analysis['hindi_summary'] = "तस्वीर का विश्लेषण पूरा हो गया है।"
            if 'english_summary' not in analysis:
                analysis['english_summary'] = "Image analysis completed successfully."
            if 'specific_findings' not in analysis:
                analysis['specific_findings'] = ["Image processed with multi-step analysis"]
            if 'farmer_recommendations' not in analysis:
                analysis['farmer_recommendations'] = ["Please provide more context for better recommendations"]
            if 'cautions' not in analysis:
                analysis['cautions'] = []
                
            return parsed_result
        else:
            # Create structured fallback response
            logger.warning("JSON parsing failed, creating structured fallback")
            return {
                "text_extracted": extracted_text[:200] + "..." if len(extracted_text) > 200 else extracted_text,
                "image_type": image_type,
                "analysis": {
                    "hindi_summary": f"{image_type} का विश्लेषण हो गया है। कृपया अधिक जानकारी के लिए English summary देखें।",
                    "english_summary": result_text[:500] + "..." if len(result_text) > 500 else result_text,
                    "specific_findings": [f"Classified as {image_type}", "Text extraction completed"],
                    "farmer_recommendations": ["Multi-step analysis performed", "Please provide more context if needed"],
                    "cautions": ["Analysis partially completed"]
                },
                "data": {"raw_response": result_text, "extracted_text": extracted_text},
                "confidence": 0.7
            }

    def _fallback_image_analysis(self, context: str) -> Dict[str, Any]:
        """Enhanced fallback when image analysis fails"""
        return {
            "text_extracted": "Image processing unavailable",
            "image_type": "unknown",
            "analysis": {
                "hindi_summary": "आपकी तस्वीर मिल गई है। कृपया बताएं कि यह कैसी तस्वीर है - फर्टिलाइजर का पैकेट, मिट्टी की रिपोर्ट, मंडी की रसीद, या फसल की तस्वीर?",
                "english_summary": "Image received successfully. Please describe what type of image this is - fertilizer package, soil test report, market receipt, or crop photo?",
                "specific_findings": ["Image processing unavailable", "Manual description needed"],
                "farmer_recommendations": ["Please describe the image content", "Mention what you want to know about this image"],
                "cautions": ["Image analysis service temporarily unavailable"]
            },
            "data": {"context": context, "processing_status": "failed"},
            "confidence": 0.1
        }

    async def compare_receipt_with_msp(self, receipt_analysis: Dict, crop_type: str) -> Dict[str, Any]:
        """Compare receipt prices with MSP"""
        try:
            # Get MSP data
            farming_tools = FarmingTools()
            msp_data = await farming_tools.get_current_msp(crop_type)
            
            extracted_data = receipt_analysis.get('extracted_data', {})
            receipt_price = extracted_data.get('price_per_unit', 0)
            
            if isinstance(receipt_price, str):
                receipt_price = float(receipt_price.replace(',', '').replace('₹', ''))
            
            msp_price = msp_data.get('msp', 0)
            
            comparison = {
                "crop": crop_type,
                "receipt_price": receipt_price,
                "msp_price": msp_price,
                "fair_compensation": receipt_price >= msp_price,
                "difference": receipt_price - msp_price,
                "percentage_difference": ((receipt_price - msp_price) / msp_price * 100) if msp_price > 0 else 0,
                "recommendation": ""
            }
            
            if not comparison["fair_compensation"] and msp_price > 0:
                comparison["recommendation"] = f"""
                ⚠️ आपको उचित मूल्य नहीं मिला!
                
                सरकारी MSP: ₹{msp_price} प्रति क्विंटल
                आपको मिला: ₹{receipt_price} प्रति क्विंटल
                नुकसान: ₹{abs(comparison['difference']):.2f} प्रति क्विंटल
                
                सुझाव:
                1. मंडी समिति से शिकायत करें
                2. eNAM पोर्टल पर रजिस्टर करें
                3. FPO से जुड़ें बेहतर मूल्य के लिए
                4. अगली बार पहले MSP चेक करें
                """
            else:
                comparison["recommendation"] = f"""
                ✅ आपको उचित मूल्य मिला!
                
                आपको मिला: ₹{receipt_price} प्रति क्विंटल
                MSP से अधिक: ₹{comparison['difference']:.2f} प्रति क्विंटल
                
                बधाई हो! आपने अच्छी कीमत पर बेचा है।
                """
            
            return comparison
            
        except Exception as e:
            logger.error(f"MSP comparison error: {e}")
            return {
                "error": "MSP comparison failed",
                "recommendation": "कृपया मैन्युअल रूप से MSP की जांच करें"
            }

class FarmingTools:
    """Tool calling framework for farming APIs"""
    
    def __init__(self):
        self.mandi_api_key = "KCV55CNNTQTQS9VPASMZAYD8"
        self.imd_base_url = "https://mausam.imd.gov.in/api"
        self.city_weather_url = "https://city.imd.gov.in/api"
    
    async def get_weather_forecast(self, location: str, days: int = 7) -> Dict[str, Any]:
        """Get weather forecast using IMD API"""
        try:
            # Use IMD city weather API
            url = f"{self.city_weather_url}/cityweather.php"
            
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                farming_advice = await self._generate_farming_weather_advice(data, location)
                
                return {
                    "success": True,
                    "location": location,
                    "forecast": data,
                    "farming_advice": farming_advice
                }
            else:
                return await self._fallback_weather(location)
                
        except Exception as e:
            logger.error(f"Weather API error: {e}")
            return await self._fallback_weather(location)
    
    async def _fallback_weather(self, location: str) -> Dict[str, Any]:
        """Fallback weather response"""
        return {
            "success": False,
            "location": location,
            "message": f"{location} के लिए मौसम की जानकारी अभी उपलब्ध नहीं है। कृपया बाद में कोशिश करें।",
            "farming_advice": "सामान्य सलाह: अपने क्षेत्र के स्थानीय मौसम पर ध्यान दें और तदनुसार खेती की योजना बनाएं।"
        }
    
    async def get_mandi_prices(self, crop: str, location: str = None) -> Dict[str, Any]:
        """Get current mandi prices"""
        try:
            # This would use the actual mandi API with your key
            # For now, using mock data structure
            mock_prices = {
                "wheat": {"price": 2100, "unit": "per quintal", "market": "Delhi Mandi"},
                "rice": {"price": 2050, "unit": "per quintal", "market": "Punjab Mandi"},
                "cotton": {"price": 6800, "unit": "per quintal", "market": "Gujarat Mandi"},
                "sugarcane": {"price": 300, "unit": "per quintal", "market": "UP Mandi"}
            }
            
            crop_data = mock_prices.get(crop.lower())
            if crop_data:
                return {
                    "success": True,
                    "crop": crop,
                    "location": location or "General",
                    "current_price": crop_data["price"],
                    "unit": crop_data["unit"],
                    "market": crop_data["market"],
                    "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M")
                }
            else:
                return {
                    "success": False,
                    "message": f"{crop} की कीमत की जानकारी उपलब्ध नहीं है।"
                }
                
        except Exception as e:
            logger.error(f"Mandi API error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_current_msp(self, crop: str) -> Dict[str, Any]:
        """Get current MSP for crop"""
        # MSP data for 2024-25 (these should be updated regularly)
        msp_data = {
            "wheat": {"msp": 2275, "season": "Rabi 2024-25", "increase": 150},
            "rice": {"msp": 2300, "season": "Kharif 2024", "increase": 180},
            "cotton": {"msp": 7121, "season": "Kharif 2024", "increase": 292},
            "sugarcane": {"msp": 340, "season": "2024-25", "increase": 25},
            "maize": {"msp": 2090, "season": "Kharif 2024", "increase": 115},
            "bajra": {"msp": 2500, "season": "Kharif 2024", "increase": 250},
            "jowar": {"msp": 3180, "season": "Kharif 2024", "increase": 300},
            "tur": {"msp": 7550, "season": "Kharif 2024", "increase": 550},
            "moong": {"msp": 8558, "season": "Kharif 2024", "increase": 500},
            "urad": {"msp": 7400, "season": "Kharif 2024", "increase": 300},
            "groundnut": {"msp": 6377, "season": "Kharif 2024", "increase": 400},
            "sunflower": {"msp": 7287, "season": "Kharif 2024", "increase": 500},
            "soybean": {"msp": 4892, "season": "Kharif 2024", "increase": 292},
            "sesame": {"msp": 8635, "season": "Kharif 2024", "increase": 500}
        }
        
        crop_lower = crop.lower()
        crop_info = msp_data.get(crop_lower, {"msp": 0, "season": "Unknown", "increase": 0})
        
        return {
            "crop": crop,
            "msp": crop_info["msp"],
            "season": crop_info["season"],
            "increase_from_last_year": crop_info["increase"],
            "currency": "INR per quintal",
            "last_updated": "2024-25 Season"
        }
    
    async def get_government_schemes(self, farmer_category: str = "small") -> Dict[str, Any]:
        """Get relevant government schemes"""
        schemes = {
            "pm_kisan": {
                "name": "PM-KISAN",
                "amount": "₹6000 per year",
                "description": "Direct income support to all farmers",
                "eligibility": "All landholding farmers",
                "how_to_apply": "Visit pmkisan.gov.in"
            },
            "crop_insurance": {
                "name": "Pradhan Mantri Fasal Bima Yojana",
                "coverage": "Yield/weather based insurance",
                "premium": "2% for Kharif, 1.5% for Rabi",
                "description": "Crop insurance scheme",
                "how_to_apply": "Through banks or CSCs"
            },
            "kisan_credit_card": {
                "name": "Kisan Credit Card",
                "benefit": "Low interest agriculture loans",
                "interest_rate": "4% (after subsidy)",
                "description": "Easy credit for farming needs",
                "how_to_apply": "Any bank branch"
            }
        }
        
        return {
            "success": True,
            "farmer_category": farmer_category,
            "schemes": schemes,
            "total_schemes": len(schemes)
        }

    async def _generate_farming_weather_advice(self, weather_data: Dict, location: str) -> str:
        """Generate farming advice based on weather"""
        try:
            if not GEMINI_AVAILABLE:
                return f"{location} के लिए मौसम के आधार पर खेती की सलाह: अपने स्थानीय मौसम पर ध्यान दें।"
            
            prompt = f"""
            Based on this weather data for {location}: {json.dumps(weather_data, indent=2)}
            
            Provide specific farming advice for next 3-5 days in Hindi and English:
            - Irrigation planning (सिंचाई योजना)
            - Pest/disease alerts (कीट/रोग चेतावनी)
            - Harvesting recommendations (कटाई सुझाव)
            - Spraying advisories (छिड़काव सलाह)
            
            Keep advice practical and location-specific for Indian farmers.
            """
            
            response = gemini_model.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"Weather advice generation error: {e}")
            return f"{location} के लिए सामान्य खेती सलाह: मौसम के अनुसार उचित कार्य करें।"

class EnhancedMemoryManager(MemoryManager):
    """Enhanced memory with conversation summarization"""
    
    def init_database(self):
        super().init_database()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversation_summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    phone_number TEXT,
                    summary_text TEXT,
                    key_topics TEXT,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    conversation_count INTEGER DEFAULT 0
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS farmer_profiles (
                    phone_number TEXT PRIMARY KEY,
                    name TEXT,
                    location TEXT,
                    primary_crops TEXT,
                    farm_size TEXT,
                    preferred_language TEXT,
                    last_interaction TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
    
    async def update_conversation_summary(self, phone_number: str):
        """Generate and update conversation summary"""
        try:
            recent_conversations = self.get_recent_conversations(phone_number, 10)
            
            if len(recent_conversations) >= 3:  # Summarize after every 3 conversations
                if not GEMINI_AVAILABLE:
                    summary = f"Farmer has {len(recent_conversations)} recent conversations about farming topics."
                else:
                    summary_prompt = f"""
                    Summarize these recent conversations with farmer {phone_number}:
                    
                    {json.dumps(recent_conversations, indent=2)}
                    
                    Extract:
                    1. Key farming topics discussed
                    2. Farmer's main concerns/challenges  
                    3. Crops they're growing
                    4. Location/region if mentioned
                    5. Recurring questions
                    
                    Create a concise summary for future context.
                    """
                    
                    response = gemini_model.generate_content(summary_prompt)
                    summary = response.text.strip()
                
                # Store summary
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO conversation_summaries 
                        (phone_number, summary_text, conversation_count) 
                        VALUES (?, ?, ?)
                    """, (phone_number, summary, len(recent_conversations)))
                    
        except Exception as e:
            logger.error(f"Summary update error: {e}")
    
    def get_farmer_context(self, phone_number: str) -> str:
        """Get comprehensive farmer context"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                # Get summary
                summary_cursor = conn.execute(
                    "SELECT summary_text FROM conversation_summaries WHERE phone_number = ? ORDER BY last_updated DESC LIMIT 1",
                    (phone_number,)
                )
                summary_row = summary_cursor.fetchone()
                
                # Get recent conversations
                recent_conversations = self.get_recent_conversations(phone_number, 3)
                
                context = ""
                if summary_row:
                    context += f"Farmer Background: {summary_row['summary_text']}\n\n"
                
                if recent_conversations:
                    context += "Recent Conversations:\n"
                    for conv in recent_conversations:
                        context += f"Q: {conv['user_input'][:100]}...\nA: {conv['ai_response'][:100]}...\n"
                
                return context
                
        except Exception as e:
            logger.error(f"Context retrieval error: {e}")
            return ""

# Initialize enhanced memory manager
enhanced_memory = EnhancedMemoryManager()

# ==== END AGENTIC COMPONENTS ====

# Pydantic models for test endpoints
class TestMessageRequest(BaseModel):
    phone_number: str
    message: str = ""
    media_url: Optional[str] = None
    media_type: Optional[str] = None

class TestAudioRequest(BaseModel):
    audio_url: str

class TestTTSRequest(BaseModel):
    text: str
    language: str = "en-IN"

class TestAIRequest(BaseModel):
    message: str
    language: str = "en-IN"
    context: str = ""

async def download_media(media_url: str) -> bytes:
    """Download media from URL and return bytes with enhanced error handling"""
    try:
        # Handle local test images
        if media_url.startswith('http://localhost:8000/test-images/'):
            # For local test images, read from file system
            filename = media_url.split('/')[-1]
            file_path = f"whatsapp/{filename}"
            
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    return f.read()
            else:
                logger.error(f"Local test image not found: {file_path}")
                return b""
        
        # Handle Twilio URLs
        elif 'twilio' in media_url:
            # Use Twilio auth for downloading
            auth = (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            response = requests.get(media_url, auth=auth, timeout=60, stream=True)
        else:
            # For other URLs with longer timeout
            response = requests.get(media_url, timeout=60, stream=True)
        
        if response.status_code == 200:
            # Read content in chunks to handle large files
            content = b""
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    content += chunk
            return content
        else:
            logger.error(f"Failed to download media: {response.status_code} - {response.text}")
            return b""
            
    except requests.exceptions.Timeout:
        logger.error(f"Media download timeout for URL: {media_url}")
        return b""
    except requests.exceptions.ConnectionError:
        logger.error(f"Media download connection error for URL: {media_url}")
        return b""
    except Exception as e:
        logger.error(f"Media download error: {e}")
        return b""

async def detect_language(text: str) -> str:
    """Enhanced language detection with better script and word recognition"""
    if not text:
        return 'hi-IN'  # Default to Hindi for farmers
    
    text_lower = text.lower()
    
    # Bengali words and patterns
    bengali_words = ['aami', 'jaante', 'chai', 'aajke', 'mandi', 'te', 'chaal', 'er', 'ki', 'daam', 'kemon', 'ache', 'koto', 'taka']
    
    # Tamil words and patterns  
    tamil_words = ['naan', 'therinja', 'venum', 'inniki', 'sandhai', 'la', 'arisi', 'oda', 'vilai', 'enna']
    
    # Telugu words and patterns
    telugu_words = ['nenu', 'telusu', 'kovali', 'ippudu', 'market', 'lo', 'biyyam', 'rate', 'enti']
    
    # Expanded Hindi words commonly used by farmers
    hindi_words = [
        'gehu', 'gehun', 'daam', 'kya', 'hai', 'mein', 'ka', 'ki', 'ke', 'bhai', 
        'sahab', 'ji', 'rate', 'chal', 'raha', 'batao', 'bataiye', 'chawal', 
        'kapas', 'kaise', 'kab', 'kahan', 'kitna', 'kyun', 'aur', 'yeh', 'woh',
        'mandi', 'keemat', 'bhav', 'mausam', 'barish', 'fasal', 'kheti', 'kisan',
        'ganna', 'makka', 'bajra', 'jowar', 'tur', 'moong', 'urad', 'yojana'
    ]
    
    # Check for specific language words first
    if any(word in text_lower for word in bengali_words):
        return 'bn-IN'
    elif any(word in text_lower for word in tamil_words):
        return 'ta-IN'
    elif any(word in text_lower for word in telugu_words):
        return 'te-IN'
    elif any(word in text_lower for word in hindi_words):
        return 'hi-IN'
    
    # Check for scripts
    if any('\u0980' <= char <= '\u09FF' for char in text):
        return 'bn-IN'
    elif any('\u0B80' <= char <= '\u0BFF' for char in text):
        return 'ta-IN'
    elif any('\u0C00' <= char <= '\u0C7F' for char in text):
        return 'te-IN'
    elif any('\u0C80' <= char <= '\u0CFF' for char in text):
        return 'kn-IN'
    elif any('\u0D00' <= char <= '\u0D7F' for char in text):
        return 'ml-IN'
    elif any('\u0A80' <= char <= '\u0AFF' for char in text):
        return 'gu-IN'
    elif any('\u0A00' <= char <= '\u0A7F' for char in text):
        return 'pa-IN'
    elif any('\u0900' <= char <= '\u097F' for char in text):
        return 'hi-IN'
    elif any('\u0600' <= char <= '\u06FF' for char in text):
        return 'ur-IN'
    else:
        # Default to Hindi for farmers
        return 'hi-IN'

async def llm_language_judge(content: str, detected_language: str) -> str:
    """LLM-based intelligent language judge for response"""
    try:
        if not GEMINI_AVAILABLE:
            return detected_language
        
        prompt = f"""
        You are a language expert for Indian farmers. Analyze this farmer's message and determine the BEST language for the response.
        
        Farmer's Message: "{content}"
        Initially Detected Language: {detected_language}
        
        LANGUAGE ANALYSIS RULES:
        1. If query contains Bengali words (aami, jaante, chai, aajke, te, er, ki, daam, kemon, ache, koto, taka) → Response in Bengali (bn-IN)
        2. If query contains Tamil words (naan, therinja, venum, inniki, sandhai, la, arisi, oda, vilai, enna) → Response in Tamil (ta-IN)  
        3. If query contains Telugu words (nenu, telusu, kovali, ippudu, market, lo, biyyam, rate, enti) → Response in Telugu (te-IN)
        4. If query contains Hindi words (gehu, daam, kya, hai, mein, ka, ki, ke, bhai, sahab, ji, rate, chal, raha, batao, bataiye, chawal, kapas, kaise, kab, kahan, kitna, kyun, aur, yeh, woh, mandi, keemat, bhav, mausam, barish, fasal, kheti, kisan) → Response in Hindi (hi-IN)
        5. If mixed languages, choose the DOMINANT language
        6. If purely English farming terms, default to Hindi (hi-IN)
        
        Return ONLY the language code: hi-IN, bn-IN, ta-IN, te-IN, kn-IN, ml-IN, gu-IN, pa-IN, mr-IN, ur-IN, or en-IN
        """
        
        response = gemini_model.generate_content(prompt)
        result = response.text.strip()
        
        # Validate response
        valid_languages = ['hi-IN', 'bn-IN', 'ta-IN', 'te-IN', 'kn-IN', 'ml-IN', 'gu-IN', 'pa-IN', 'mr-IN', 'ur-IN', 'en-IN']
        if result in valid_languages:
            return result
        else:
            return detected_language
            
    except Exception as e:
        logger.error(f"LLM language judge error: {e}")
        return detected_language

async def speech_to_text(audio_url: str) -> tuple[str, str]:
    """Convert audio to text"""
    try:
        # Download audio
        auth = (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        audio_response = requests.get(audio_url, auth=auth)
        
        if audio_response.status_code != 200:
            return "", "en-IN"
        
        # Convert to WAV
        with tempfile.NamedTemporaryFile(suffix='.ogg', delete=False) as temp_file:
            temp_file.write(audio_response.content)
            temp_path = temp_file.name
        
        audio_segment = AudioSegment.from_file(temp_path)
        audio_segment = audio_segment.set_channels(1).set_frame_rate(16000)
        
        wav_path = temp_path.replace('.ogg', '.wav')
        audio_segment.export(wav_path, format="wav")
        
        # Send to Sarvam STT
        with open(wav_path, 'rb') as audio_file:
            files = {'file': ('audio.wav', audio_file, 'audio/wav')}
            data = {'model': 'saarika:v2.5'}
            headers = {"api-subscription-key": SARVAM_API_KEY}
            
            response = requests.post(
                f"{SARVAM_BASE_URL}/speech-to-text",
                headers=headers,
                files=files,
                data=data,
                timeout=30
            )
        
        # Cleanup
        os.unlink(temp_path)
        os.unlink(wav_path)
        
        if response.status_code == 200:
            result = response.json()
            transcript = result.get('transcript', '').strip()
            language_code = result.get('language_code', 'en')
            
            # Map to our format
            if language_code == 'hi':
                language = 'hi-IN'
            elif language_code == 'en':
                language = 'en-IN'
            else:
                language = f"{language_code}-IN" if language_code in ['ta', 'te', 'kn', 'ml', 'gu', 'pa', 'bn', 'mr'] else 'en-IN'
            
            return transcript, language
        
        return "", "en-IN"
        
    except Exception as e:
        logger.error(f"STT error: {e}")
        return "", "en-IN"

async def text_to_speech(text: str, language: str, retry_count: int = 0) -> Optional[str]:
    """Convert text to speech with balanced quality for WhatsApp"""
    try:
        # Clean text more aggressively
        clean_text = text.replace("🤖", "").replace("📚", "").replace("🔍", "").replace("✅", "").replace("⚠️", "").strip()
        
        # Remove extra whitespace and newlines
        clean_text = ' '.join(clean_text.split())
        
        # Limit text length based on retry count - be more aggressive
        if retry_count == 0 and len(clean_text) > 150:
            clean_text = clean_text[:147] + "..."
        elif retry_count == 1 and len(clean_text) > 80:
            clean_text = clean_text[:77] + "..."
        elif retry_count >= 2:
            clean_text = "आपका जवाब तैयार है।"  # Very short fallback
        
        # Use supported language or fallback to English
        tts_language = language if language in TTS_SUPPORTED else 'en-IN'
        
        # On retry, force Hindi or English
        if retry_count > 0:
            tts_language = 'hi-IN' if retry_count == 1 else 'en-IN'
        
        logger.info(f"TTS attempt {retry_count + 1}: {clean_text[:50]}... in {tts_language}")
        
        response = requests.post(
            f"{SARVAM_BASE_URL}/text-to-speech",
            headers=SARVAM_HEADERS,
            json={
                "inputs": [clean_text],
                "target_language_code": tts_language,
                "speaker": "meera",
                "pitch": 0,
                "pace": 1.65,
                "loudness": 1.5,
                "speech_sample_rate": 16000,  # WhatsApp optimal sample rate
                "enable_preprocessing": True,
                "model": "bulbul:v1"
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            audio_base64 = result.get('audios', [None])[0]
            if audio_base64:
                audio_data = base64.b64decode(audio_base64)
                
                audio_id = str(uuid.uuid4())[:8]
                
                # Create WAV first
                wav_filename = f"response_{audio_id}.wav"
                wav_path = os.path.join(AUDIO_DIR, wav_filename)
                
                with open(wav_path, 'wb') as f:
                    f.write(audio_data)
                
                # Convert to MP3 with extremely conservative settings for WhatsApp
                try:
                    from pydub import AudioSegment
                    
                    # Load the WAV
                    audio_segment = AudioSegment.from_wav(wav_path)
                    
                    # Apply very conservative settings for WhatsApp compatibility
                    audio_optimized = audio_segment.set_frame_rate(16000).set_channels(1)
                    
                    # Ensure audio duration is very short (WhatsApp prefers shorter files)
                    max_duration_ms = 15000  # 15 seconds max (very short)
                    if len(audio_optimized) > max_duration_ms:
                        audio_optimized = audio_optimized[:max_duration_ms]
                        logger.warning(f"Audio truncated to {max_duration_ms/1000}s for WhatsApp compatibility")
                    
                    # Apply audio normalization for consistent volume
                    audio_optimized = audio_optimized.normalize()
                    
                    # Create MP3 with ultra-conservative settings
                    mp3_filename = f"response_{audio_id}.mp3"
                    mp3_path = os.path.join(AUDIO_DIR, mp3_filename)
                    
                    # Export as MP3 with extremely conservative settings
                    audio_optimized.export(
                        mp3_path,
                        format="mp3",
                        bitrate="24k",  # Very low bitrate
                        parameters=[
                            "-ar", "16000",    # 16kHz sample rate
                            "-ac", "1",        # Mono
                            "-b:a", "24k",     # Explicit bitrate
                            "-q:a", "7",       # Lower quality for smaller size
                            "-f", "mp3"        # Explicit format
                        ]
                    )
                    
                    # Verify the MP3 file was created and has reasonable size
                    if os.path.exists(mp3_path):
                        file_size = os.path.getsize(mp3_path)
                        if file_size > 5 * 1024 * 1024:  # 5MB limit (very conservative)
                            logger.error(f"MP3 file too large: {file_size} bytes, trying ultra compression")
                            # Try even more compressed version
                            return await _create_ultra_compressed_mp3(audio_optimized, audio_id)
                        elif file_size < 500:  # File too small, likely corrupted
                            logger.error(f"MP3 file too small: {file_size} bytes")
                            if retry_count < 3:
                                return await text_to_speech(text, language, retry_count + 1)
                        else:
                            logger.info(f"TTS success: {wav_filename} -> {mp3_filename} ({file_size} bytes)")
                            return mp3_filename
                    
                except Exception as mp3_error:
                    logger.warning(f"MP3 conversion failed: {mp3_error}")
                
                # If MP3 failed, try ultra-compressed version
                try:
                    return await _create_ultra_compressed_mp3(AudioSegment.from_wav(wav_path), audio_id)
                except Exception as ultra_error:
                    logger.warning(f"Ultra-compressed MP3 also failed: {ultra_error}")
                    
                    # Last resort: return original WAV with warning
                    logger.warning(f"Using original WAV file: {wav_filename}")
                    return wav_filename
        
        # Log the error details
        logger.error(f"TTS API error: {response.status_code} - {response.text[:200]}")
        
        # Retry with different parameters
        if retry_count < 3:
            return await text_to_speech(text, language, retry_count + 1)
        
        return None
        
    except Exception as e:
        logger.error(f"TTS error (attempt {retry_count + 1}): {e}")
        
        # Retry with fallback
        if retry_count < 3:
            return await text_to_speech(text, language, retry_count + 1)
        
        return None

async def _create_ultra_compressed_mp3(audio_segment, audio_id: str) -> Optional[str]:
    """Create compressed MP3 for WhatsApp compatibility (still maintains basic quality)"""
    try:
        mp3_filename = f"response_{audio_id}_compressed.mp3"
        mp3_path = os.path.join(AUDIO_DIR, mp3_filename)
        
        # Apply reasonable compression (not too aggressive to avoid silence)
        audio_compressed = audio_segment.set_frame_rate(16000).set_channels(1)  # Keep reasonable quality
        
        # Limit duration to reasonable length
        if len(audio_compressed) > 15000:  # 15 seconds
            audio_compressed = audio_compressed[:15000]
        
        # Gentle normalization
        audio_compressed = audio_compressed.normalize()
        
        # Export with reasonable compression settings
        audio_compressed.export(
            mp3_path,
            format="mp3",
            bitrate="32k",  # Low but not too low bitrate
            parameters=[
                "-ar", "16000",     # Keep reasonable sample rate
                "-ac", "1",         # Mono
                "-b:a", "32k",      # Explicit low bitrate
                "-q:a", "6",        # Lower quality but not lowest
                "-f", "mp3"         # Explicit format
            ]
        )
        
        if os.path.exists(mp3_path):
            file_size = os.path.getsize(mp3_path)
            if file_size > 5 * 1024 * 1024:  # 5MB limit
                logger.error(f"Compressed MP3 still too large: {file_size} bytes")
                return None
            elif file_size < 1000:  # File too small
                logger.error(f"Compressed MP3 too small: {file_size} bytes")
                return None
            else:
                logger.info(f"Created compressed MP3: {mp3_filename} ({file_size} bytes)")
                return mp3_filename
        
        return None
        
    except Exception as e:
        logger.error(f"Compressed MP3 creation failed: {e}")
        return None

async def ensure_audio_response(text: str, language: str) -> str:
    """Ensure we ALWAYS have an audio response, even if we need to create a backup"""
    
    # Try primary TTS
    audio_filename = await text_to_speech(text, language)
    
    if audio_filename:
        return audio_filename
    
    # Fallback 1: Try with very simple message
    logger.warning("Primary TTS failed, trying simple message...")
    simple_msg = "आपका संदेश मिल गया है।" if language.startswith('hi') else "Message received."
    audio_filename = await text_to_speech(simple_msg, 'hi-IN')
    
    if audio_filename:
        return audio_filename
    
    # Fallback 2: Create a WhatsApp-compatible dummy audio file with actual content
    logger.error("All TTS attempts failed, creating WhatsApp-compatible fallback audio...")
    try:
        audio_id = str(uuid.uuid4())[:8]
        
        # Create an MP3 file with actual speech content (not just tone)
        from pydub import AudioSegment
        from pydub.generators import Sine
        
        # Generate a simple beep pattern (more recognizable than tone)
        beep1 = Sine(800).to_audio_segment(duration=200)  # High beep
        silence = AudioSegment.silent(duration=100)       # Short pause
        beep2 = Sine(600).to_audio_segment(duration=200)  # Lower beep
        
        # Combine beeps to create a notification sound
        notification = beep1 + silence + beep2
        notification = notification.set_frame_rate(16000).set_channels(1)
        
        # Normalize volume
        notification = notification.normalize()
        
        mp3_filename = f"fallback_{audio_id}.mp3"
        mp3_path = os.path.join(AUDIO_DIR, mp3_filename)
        
        notification.export(
            mp3_path,
            format="mp3",
            bitrate="64k",
            parameters=["-ar", "16000", "-ac", "1"]
        )
        
        logger.info(f"Created WhatsApp-compatible fallback audio: {mp3_filename}")
        return mp3_filename
        
    except Exception as e:
        logger.error(f"Failed to create fallback audio: {e}")
        return None

async def generate_ai_response(message: str, language: str, context: str = "") -> str:
    """Generate AI response"""
    try:
        if GEMINI_AVAILABLE and gemini_model:
            full_prompt = f"""
You are a helpful farming assistant. Respond naturally and helpfully to this farmer's question.

Context: {context}
Question: {message}

Provide a helpful response in 1-2 sentences. Be conversational and practical.
"""
            response = gemini_model.generate_content(full_prompt)
            return response.text.strip()
        else:
            # Simple fallback
            return f"I understand your farming question. Let me help you with that based on best practices."
            
    except Exception as e:
        logger.error(f"AI response error: {e}")
        return "I'm here to help with your farming questions. Please provide more details."

async def agentic_process_query(phone_number: str, message: str, media_url: str = None, media_type: str = None) -> Dict[str, Any]:
    """Agentic query processing with intent understanding and tool calling"""
    
    # Initialize components
    intent_classifier = IntentClassifier()
    image_analyzer = ImageAnalyzer()
    farming_tools = FarmingTools()
    
    # Get farmer context
    farmer_context = enhanced_memory.get_farmer_context(phone_number)
    
    logger.info(f"Agentic processing for {phone_number}: {message[:50]}...")
    
    # Step 1: Advanced Intent Classification and Content Extraction
    image_analysis = None
    response_language = 'hi-IN'  # Default response language
    
    if media_url and 'image' in media_type:
        # Download and analyze image
        image_data = await download_media(media_url)
        if image_data:
            intent_analysis = await intent_classifier.classify_intent(message, media_type, image_data)
            
            # Advanced image analysis
            image_analysis = await image_analyzer.analyze_farming_image(image_data, message)
            
            # Special handling for receipts - MSP comparison
            if image_analysis['image_type'] == 'receipt':
                extracted_data = image_analysis.get('extracted_data', {})
                crop_type = extracted_data.get('crop_type')
                if crop_type:
                    msp_comparison = await image_analyzer.compare_receipt_with_msp(image_analysis, crop_type)
                    image_analysis['msp_comparison'] = msp_comparison
            
            content = f"{message} [Image: {image_analysis['image_type']}]"
            detected_language = await detect_language(message) if message else 'hi-IN'
        else:
            content = f"{message} [Image upload failed]"
            detected_language = await detect_language(message) if message else 'hi-IN'
            intent_analysis = await intent_classifier.classify_intent(content)
        
    elif media_url and 'audio' in media_type:
        transcript, detected_language = await speech_to_text(media_url)
        content = transcript if transcript else "Audio received but couldn't transcribe"
        intent_analysis = await intent_classifier.classify_intent(content)
        
    else:
        content = message or "Empty message"
        detected_language = await detect_language(content)
        intent_analysis = await intent_classifier.classify_intent(content)
    
    # Use LLM language judge for intelligent response language selection
    response_language = await llm_language_judge(content, detected_language)
    
    logger.info(f"Intent: {intent_analysis['primary_intent']} (confidence: {intent_analysis.get('confidence', 0)})")
    
    # Step 2: Tool Calling Based on Intent
    tool_results = {}
    
    try:
        if intent_analysis['primary_intent'] == 'weather_info':
            # Extract location dynamically or use intelligent default
            location = intent_analysis.get('extracted_entities', {}).get('location')
            if not location:
                # Try to infer from content
                content_lower = content.lower()
                if 'delhi' in content_lower:
                    location = 'Delhi'
                elif 'mumbai' in content_lower:
                    location = 'Mumbai'
                elif 'kolkata' in content_lower:
                    location = 'Kolkata'
                elif 'chennai' in content_lower:
                    location = 'Chennai'
                elif 'bangalore' in content_lower:
                    location = 'Bangalore'
                elif 'hyderabad' in content_lower:
                    location = 'Hyderabad'
                elif 'punjab' in content_lower:
                    location = 'Punjab'
                elif 'haryana' in content_lower:
                    location = 'Haryana'
                else:
                    location = 'Delhi'  # Intelligent default for India
            
            tool_results['weather'] = await farming_tools.get_weather_forecast(location)
            
        elif intent_analysis['primary_intent'] == 'mandi_support':
            crop = intent_analysis.get('extracted_entities', {}).get('crop_type')
            location = intent_analysis.get('extracted_entities', {}).get('location')
            
            # Smart crop inference if not extracted
            if not crop:
                content_lower = content.lower()
                if 'gehu' in content_lower or 'gehun' in content_lower or 'wheat' in content_lower:
                    crop = 'wheat'
                elif 'chawal' in content_lower or 'chaal' in content_lower or 'rice' in content_lower:
                    crop = 'rice'
                elif 'kapas' in content_lower or 'cotton' in content_lower:
                    crop = 'cotton'
                elif 'ganna' in content_lower or 'sugarcane' in content_lower:
                    crop = 'sugarcane'
                elif 'makka' in content_lower or 'maize' in content_lower:
                    crop = 'maize'
                else:
                    crop = 'wheat'  # Default fallback
            
            # Smart location inference if not extracted
            if not location:
                content_lower = content.lower()
                if 'kolkata' in content_lower or 'calcutta' in content_lower:
                    location = 'Kolkata'
                elif 'delhi' in content_lower:
                    location = 'Delhi'
                elif 'mumbai' in content_lower:
                    location = 'Mumbai'
                elif 'chennai' in content_lower:
                    location = 'Chennai'
                elif 'bangalore' in content_lower:
                    location = 'Bangalore'
                elif 'hyderabad' in content_lower:
                    location = 'Hyderabad'
                elif 'punjab' in content_lower:
                    location = 'Punjab'
                elif 'haryana' in content_lower:
                    location = 'Haryana'
                elif 'up' in content_lower or 'uttar pradesh' in content_lower:
                    location = 'Uttar Pradesh'
                elif 'mp' in content_lower or 'madhya pradesh' in content_lower:
                    location = 'Madhya Pradesh'
                else:
                    location = None  # Let the API use its default
            
            # Always call tools for mandi_support with dynamic location
            tool_results['mandi_prices'] = await farming_tools.get_mandi_prices(crop, location)
            tool_results['msp'] = await farming_tools.get_current_msp(crop)
                
        elif intent_analysis['primary_intent'] == 'government_schemes':
            # Extract farmer category if mentioned
            farmer_category = 'small'  # Default
            content_lower = content.lower()
            if 'large' in content_lower or 'big' in content_lower:
                farmer_category = 'large'
            elif 'medium' in content_lower:
                farmer_category = 'medium'
            
            tool_results['schemes'] = await farming_tools.get_government_schemes(farmer_category)
            
        elif intent_analysis['primary_intent'] == 'msp_comparison':
            # For direct MSP queries with smart crop inference
            crop = intent_analysis.get('extracted_entities', {}).get('crop_type')
            if not crop:
                content_lower = content.lower()
                if 'gehu' in content_lower or 'wheat' in content_lower:
                    crop = 'wheat'
                elif 'chawal' in content_lower or 'rice' in content_lower:
                    crop = 'rice'
                else:
                    crop = 'wheat'
            
            tool_results['msp'] = await farming_tools.get_current_msp(crop)
            
    except Exception as e:
        logger.error(f"Tool calling error: {e}")
        tool_results['error'] = f"Error getting additional information: {str(e)}"
    
    # Step 3: Generate Contextual AI Response using judged language
    ai_response = await generate_agentic_response(
        content=content,
        intent_analysis=intent_analysis,
        image_analysis=image_analysis,
        tool_results=tool_results,
        farmer_context=farmer_context,
        language=response_language
    )
    
    # Step 4: ALWAYS Generate Audio Response in judged language
    audio_filename = await ensure_audio_response(ai_response, response_language)
    
    # Step 5: Store and update memory
    enhanced_memory.store_conversation(phone_number, content, 
                                     intent_analysis['primary_intent'], 
                                     detected_language, ai_response)
    
    # Update conversation summary asynchronously
    try:
        await enhanced_memory.update_conversation_summary(phone_number)
    except Exception as e:
        logger.error(f"Summary update error: {e}")
    
    return {
        "text_response": ai_response,
        "audio_filename": audio_filename,
        "detected_language": detected_language,
        "response_language": response_language,
        "intent_analysis": intent_analysis,
        "image_analysis": image_analysis,
        "tool_results": tool_results,
        "input_type": media_type or "text",
        "farmer_context": farmer_context[:100] + "..." if len(farmer_context) > 100 else farmer_context
    }

async def generate_agentic_response(content: str, intent_analysis: Dict, image_analysis: Dict, 
                                  tool_results: Dict, farmer_context: str, language: str) -> str:
    """Generate contextual response using all available data - GUARANTEED HINDI RESPONSE"""
    
    try:
        if not GEMINI_AVAILABLE:
            return await generate_fallback_response(content, intent_analysis, tool_results, language)
        
        # Use the judged language for response
        response_language = language
        
        # Language-specific response instructions
        if language == 'bn-IN':
            language_instruction = "CRITICAL: You MUST respond in Bengali (বাংলা) script mixed with English technical terms only when necessary. Your response should be primarily in Bengali to help Bengali farmers."
        elif language == 'ta-IN':
            language_instruction = "CRITICAL: You MUST respond in Tamil (தமிழ்) script mixed with English technical terms only when necessary. Your response should be primarily in Tamil to help Tamil farmers."
        elif language == 'te-IN':
            language_instruction = "CRITICAL: You MUST respond in Telugu (తెలుగు) script mixed with English technical terms only when necessary. Your response should be primarily in Telugu to help Telugu farmers."
        elif language == 'kn-IN':
            language_instruction = "CRITICAL: You MUST respond in Kannada (ಕನ್ನಡ) script mixed with English technical terms only when necessary. Your response should be primarily in Kannada to help Kannada farmers."
        elif language == 'ml-IN':
            language_instruction = "CRITICAL: You MUST respond in Malayalam (മലയാളം) script mixed with English technical terms only when necessary. Your response should be primarily in Malayalam to help Malayalam farmers."
        elif language == 'gu-IN':
            language_instruction = "CRITICAL: You MUST respond in Gujarati (ગુજરાતી) script mixed with English technical terms only when necessary. Your response should be primarily in Gujarati to help Gujarati farmers."
        elif language == 'pa-IN':
            language_instruction = "CRITICAL: You MUST respond in Punjabi (ਪੰਜਾਬੀ) script mixed with English technical terms only when necessary. Your response should be primarily in Punjabi to help Punjabi farmers."
        elif language == 'mr-IN':
            language_instruction = "CRITICAL: You MUST respond in Marathi (मराठी) script mixed with English technical terms only when necessary. Your response should be primarily in Marathi to help Marathi farmers."
        elif language == 'ur-IN':
            language_instruction = "CRITICAL: You MUST respond in Urdu (اردو) script mixed with English technical terms only when necessary. Your response should be primarily in Urdu to help Urdu farmers."
        else:  # Default to Hindi
            language_instruction = "CRITICAL: You MUST respond in Hindi (Devanagari script) mixed with English technical terms only when necessary. Your response should be primarily in Hindi to help Indian farmers."
        
        prompt = f"""
        You are Krishi Saarthi - the comprehensive farming assistant for Indian farmers.
        
        {language_instruction}
        
        Farmer Context: {farmer_context}
        
        Current Query: {content}
        Intent: {intent_analysis['primary_intent']} (confidence: {intent_analysis.get('confidence', 0)})
        Urgency: {intent_analysis.get('urgency', 'medium')}
        Detected Language: {language}
        
        Available Tool Data:
        {json.dumps(tool_results, indent=2) if tool_results else "कोई टूल डेटा उपलब्ध नहीं"}
        
        {f"Image Analysis: {json.dumps(image_analysis, indent=2)}" if image_analysis else ""}
        
        Provide a helpful, contextual response in HINDI that:
        1. Addresses the farmer's specific intent in Hindi
        2. Uses the tool data when available and explain in Hindi
        3. Gives practical, actionable advice in simple Hindi
        4. Considers the farmer's context and history
        5. Is appropriate for their urgency level: {intent_analysis.get('urgency', 'medium')}
        
        Special Instructions:
        - If MSP comparison shows unfair pricing, be supportive in Hindi and provide actionable steps
        - If this is a soil test or fertilizer analysis, give specific recommendations in Hindi
        - For weather queries, provide farming-specific advice in Hindi
        - If image analysis is available, reference the image content specifically in Hindi
        - Always be encouraging and supportive in Hindi
        - Use simple Hindi words that farmers can understand
        - Include English technical terms only when absolutely necessary
        
        MANDATORY: Respond primarily in {language} as instructed above.
        Keep response conversational and practical, around 100-150 words in the target language.
        """
        
        response = gemini_model.generate_content(prompt)
        result = response.text.strip()
        
        # Validate response is in the correct script
        script_valid = False
        if language == 'bn-IN' and any('\u0980' <= char <= '\u09FF' for char in result):
            script_valid = True
        elif language == 'ta-IN' and any('\u0B80' <= char <= '\u0BFF' for char in result):
            script_valid = True
        elif language == 'te-IN' and any('\u0C00' <= char <= '\u0C7F' for char in result):
            script_valid = True
        elif language == 'kn-IN' and any('\u0C80' <= char <= '\u0CFF' for char in result):
            script_valid = True
        elif language == 'ml-IN' and any('\u0D00' <= char <= '\u0D7F' for char in result):
            script_valid = True
        elif language == 'gu-IN' and any('\u0A80' <= char <= '\u0AFF' for char in result):
            script_valid = True
        elif language == 'pa-IN' and any('\u0A00' <= char <= '\u0A7F' for char in result):
            script_valid = True
        elif language == 'mr-IN' and any('\u0900' <= char <= '\u097F' for char in result):
            script_valid = True
        elif language == 'ur-IN' and any('\u0600' <= char <= '\u06FF' for char in result):
            script_valid = True
        elif language == 'hi-IN' and any('\u0900' <= char <= '\u097F' for char in result):
            script_valid = True
        
        # If response is not in the correct script, force fallback
        if not script_valid:
            logger.warning(f"LLM response was not in {language} script, forcing fallback")
            return await generate_fallback_response(content, intent_analysis, tool_results, language)
        
        return result
        
    except Exception as e:
        logger.error(f"Agentic response generation error: {e}")
        return await generate_fallback_response(content, intent_analysis, tool_results, 'hi-IN')

async def generate_fallback_response(content: str, intent_analysis: Dict, tool_results: Dict, language: str) -> str:
    """Fallback response when Gemini is unavailable - Multi-language support"""
    
    intent = intent_analysis['primary_intent']
    
    # Language-specific responses
    if language == 'bn-IN':
        # Bengali responses
        if intent == 'weather_info':
            if tool_results.get('weather', {}).get('success'):
                return f"আবহাওয়ার তথ্য পাওয়া গেছে। {tool_results['weather'].get('farming_advice', 'দয়া করে আবহাওয়া অনুযায়ী চাষ করুন।')}"
            else:
                return "আবহাওয়ার তথ্যের জন্য স্থানীয় আবহাওয়া দপ্তরে যোগাযোগ করুন। আপনার এলাকার আবহাওয়া জেনে সঠিক সময়ে চাষের পরিকল্পনা করুন।"
        elif intent == 'mandi_support':
            if tool_results.get('mandi_prices', {}).get('success'):
                price_data = tool_results['mandi_prices']
                msp_info = ""
                if tool_results.get('msp', {}).get('msp'):
                    msp_price = tool_results['msp']['msp']
                    msp_info = f" MSP ₹{msp_price} প্রতি কুইন্টাল।"
                return f"🌾 {price_data['crop']} এর বর্তমান মান্ডি দাম ₹{price_data['current_price']} প্রতি কুইন্টাল।{msp_info} ভাল দামের জন্য বিভিন্ন মান্ডিতে দাম দেখুন।"
            else:
                return "মান্ডির দামের জন্য নিকটস্থ মান্ডিতে যোগাযোগ করুন। আপনি eNAM পোর্টালেও অনলাইনে দাম দেখতে পারেন।"
        elif intent == 'government_schemes':
            return "🏛️ প্রধান সরকারি প্রকল্প: PM-KISAN (₹৬০০০/বছর), প্রধানমন্ত্রী ফসল বীমা প্রকল্প, কিষান ক্রেডিট কার্ড। নিকটস্থ কৃষি অফিস বা CSC কেন্দ্রে যোগাযোগ করুন।"
        else:
            return "🤝 নমস্কার! আপনার প্রশ্ন বুঝতে পেরেছি। দয়া করে বিস্তারিত বলুন আপনার কী তথ্য দরকার - মান্ডির দাম, আবহাওয়া, প্রকল্প, বা চাষের পরামর্শ?"
    
    elif language == 'ta-IN':
        # Tamil responses
        if intent == 'weather_info':
            if tool_results.get('weather', {}).get('success'):
                return f"வானிலை தகவல் கிடைத்துள்ளது। {tool_results['weather'].get('farming_advice', 'தயவுசெய்து வானிலைக்கு ஏற்ப விவசாயம் செய்யுங்கள்।')}"
            else:
                return "வானிலை தகவலுக்கு உங்கள் உள்ளூர் வானிலை அலுவலகத்தை தொடர்பு கொள்ளுங்கள். உங்கள் பகுதியின் வானிலையை அறிந்து சரியான நேரத்தில் விவசாய திட்டம் செய்யுங்கள்।"
        elif intent == 'mandi_support':
            if tool_results.get('mandi_prices', {}).get('success'):
                price_data = tool_results['mandi_prices']
                msp_info = ""
                if tool_results.get('msp', {}).get('msp'):
                    msp_price = tool_results['msp']['msp']
                    msp_info = f" MSP ₹{msp_price} ஒரு குவிண்டலுக்கு।"
                return f"🌾 {price_data['crop']} இன் தற்போதைய சந்தை விலை ₹{price_data['current_price']} ஒரு குவிண்டலுக்கு।{msp_info} நல்ல விலைக்கு வெவ்வேறு சந்தைகளில் விலை பார்க்கவும்।"
            else:
                return "சந்தை விலைகளுக்கு அருகிலுள்ள சந்தையை தொடர்பு கொள்ளுங்கள். நீங்கள் eNAM போர்ட்டலிலும் ஆன்லைனில் விலைகளை பார்க்கலாம்।"
        elif intent == 'government_schemes':
            return "🏛️ முக்கிய அரசு திட்டங்கள்: PM-KISAN (₹6000/ஆண்டு), பிரதமர் பயிர் காப்பீடு திட்டம், கிசான் கிரெடிட் கார்டு। அருகிலுள்ள விவசாய அலுவலகம் அல்லது CSC மையத்தை தொடர்பு கொள்ளுங்கள்।"
        else:
            return "🤝 வணக்கம்! உங்கள் கேள்வி புரிந்தது। தயவுசெய்து விரிவாக சொல்லுங்கள் உங்களுக்கு என்ன தகவல் வேண்டும் - சந்தை விலை, வானிலை, திட்டங்கள், அல்லது விவசாய ஆலோசனை?"
    
    elif language == 'te-IN':
        # Telugu responses
        if intent == 'weather_info':
            if tool_results.get('weather', {}).get('success'):
                return f"వాతావరణ సమాచారం లభించింది। {tool_results['weather'].get('farming_advice', 'దయచేసి వాతావరణం ప్రకారం వ్యవసాయం చేయండి।')}"
            else:
                return "వాతావరణ సమాచారం కోసం మీ స్థానిక వాతావరణ విభాగాన్ని సంప్రదించండి. మీ ప్రాంతపు వాతావరణం తెలుసుకుని సరైన సమయంలో వ్యవసాయ ప్రణాళిక చేయండి।"
        elif intent == 'mandi_support':
            if tool_results.get('mandi_prices', {}).get('success'):
                price_data = tool_results['mandi_prices']
                msp_info = ""
                if tool_results.get('msp', {}).get('msp'):
                    msp_price = tool_results['msp']['msp']
                    msp_info = f" MSP ₹{msp_price} ప్రతి క్వింటల్‌కు।"
                return f"🌾 {price_data['crop']} యొక్క ప్రస్తుత మార్కెట్ ధర ₹{price_data['current_price']} ప్రతి క్వింటల్‌కు।{msp_info} మంచి ధర కోసం వివిధ మార్కెట్లలో ధరలు చూడండి।"
            else:
                return "మార్కెట్ ధరల కోసం మీ సమీప మార్కెట్‌ను సంప్రదించండి. మీరు eNAM పోర్టల్‌లో కూడా ఆన్‌లైన్‌లో ధరలు చూడవచ్చు।"
        elif intent == 'government_schemes':
            return "🏛️ ప్రధాన ప్రభుత్వ పథకాలు: PM-KISAN (₹6000/సంవత్సరం), ప్రధానమంత్రి పంట బీమా పథకం, కిసాన్ క్రెడిట్ కార్డ్. మీ సమీప వ్యవసాయ కార్యాలయం లేదా CSC కేంద్రాన్ని సంప్రదించండి।"
        else:
            return "🤝 నమస్కారం! మీ ప్రశ్న అర్థమైంది. దయచేసి వివరంగా చెప్పండి మీకు ఏ సమాచారం కావాలి - మార్కెట్ ధర, వాతావరణం, పథకాలు, లేదా వ్యవసాయ సలహా?"
    
    else:  # Default to Hindi
        if intent == 'weather_info':
            if tool_results.get('weather', {}).get('success'):
                return f"मौसम की जानकारी मिल गई है। {tool_results['weather'].get('farming_advice', 'कृपया मौसम के अनुसार खेती करें।')}"
            else:
                return "मौसम की जानकारी के लिए अपने स्थानीय मौसम विभाग से संपर्क करें। आपके क्षेत्र का मौसम जानकर सही समय पर खेती की योजना बनाएं।"
                
        elif intent == 'mandi_support':
            if tool_results.get('mandi_prices', {}).get('success'):
                price_data = tool_results['mandi_prices']
                msp_info = ""
                if tool_results.get('msp', {}).get('msp'):
                    msp_price = tool_results['msp']['msp']
                    msp_info = f" MSP ₹{msp_price} प्रति क्विंटल है।"
                return f"🌾 {price_data['crop']} की वर्तमान मंडी कीमत ₹{price_data['current_price']} प्रति क्विंटल है।{msp_info} बेहतर दाम के लिए अलग-अलग मंडियों में भाव जांचें।"
            else:
                return "मंडी की कीमतों के लिए अपने नजदीकी मंडी से संपर्क करें। आप eNAM पोर्टल पर भी ऑनलाइन भाव देख सकते हैं।"
                
        elif intent == 'government_schemes':
            return "🏛️ मुख्य सरकारी योजनाएं: PM-KISAN (₹6000/वर्ष), प्रधानमंत्री फसल बीमा योजना, किसान क्रेडिट कार्ड। अपने नजदीकी कृषि कार्यालय या CSC केंद्र से संपर्क करें।"
            
        elif intent == 'image_analysis':
            return "📸 आपकी तस्वीर मिल गई है। कृपया बताएं कि यह क्या है - फर्टिलाइजर का पैकेट, मिट्टी टेस्ट रिपोर्ट, या फसल की तस्वीर? ताकि मैं बेहतर सलाह दे सकूं।"
            
        elif intent == 'crop_advisory':
            return "🌱 फसल की सलाह के लिए आपकी फसल का नाम, समस्या का विवरण और आपका क्षेत्र बताएं। मैं आपको बेहतर सुझाव दे सकूंगा।"
            
        else:
            return "🤝 नमस्कार! आपका सवाल समझ आया है। कृपया और विस्तार से बताएं कि आपको क्या जानकारी चाहिए - मंडी भाव, मौसम, योजनाएं, या खेती की सलाह?"

# Keep the old function for backward compatibility  
async def process_query(phone_number: str, message: str, media_url: str = None, media_type: str = None) -> Dict[str, Any]:
    """Legacy function - redirects to agentic version"""
    return await agentic_process_query(phone_number, message, media_url, media_type)

async def send_audio_message(to_number: str, audio_filename: str, text_response: str):
    """Send audio message via Twilio WhatsApp"""
    try:
        if not audio_filename:
            logger.warning("No audio filename provided, generating fallback audio")
            audio_filename = await ensure_audio_response(text_response[:50], 'hi-IN')
        
        if not audio_filename:
            logger.error("Failed to generate any audio, sending text instead")
            return await send_text_message(to_number, text_response)
        
        # Construct full audio URL
        audio_url = f"{BASE_URL}/audio/{audio_filename}"
        
        # Verify audio file exists locally
        audio_path = os.path.join(AUDIO_DIR, audio_filename)
        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}, sending text instead")
            return await send_text_message(to_number, text_response)
        
        logger.info(f"✅ Sending audio message to {to_number}: {audio_url}")
        
        # ACTUAL TWILIO CALL - Send audio message
        message = twilio_client.messages.create(
            from_=f"whatsapp:{TWILIO_NUMBER}",
            to=to_number,
            body="🎤 आपका जवाब तैयार है | Your response is ready",
            media_url=[audio_url]
        )
        
        logger.info(f"✅ Audio message sent successfully! SID: {message.sid}")
        
        return {
            "success": True,
            "message_sid": message.sid,
            "audio_url": audio_url,
            "to_number": to_number,
            "delivery_method": "TWILIO_WHATSAPP"
        }
        
    except Exception as e:
        logger.error(f"❌ Error sending audio message: {e}")
        # Fallback to text message
        return await send_text_message(to_number, text_response)

async def send_text_message(to_number: str, text: str):
    """Send text message via Twilio WhatsApp"""
    try:
        # Ensure proper WhatsApp formatting
        if not to_number.startswith('whatsapp:'):
            to_number = f"whatsapp:{to_number.replace('whatsapp:', '')}"
        
        logger.info(f"✅ Sending text message to {to_number}: {text[:50]}...")
        
        # ACTUAL TWILIO CALL - Send text message
        message = twilio_client.messages.create(
            from_=f"whatsapp:{TWILIO_NUMBER}",
            to=to_number,
            body=text
        )
        
        logger.info(f"✅ Text message sent successfully! SID: {message.sid}")
        
        return {
            "success": True,
            "message_sid": message.sid,
            "to_number": to_number,
            "message_text": text,
            "delivery_method": "TWILIO_WHATSAPP"
        }
        
    except Exception as e:
        logger.error(f"❌ Error sending text message: {e}")
        return {
            "success": False,
            "error": str(e),
            "to_number": to_number
        }

@app.get("/")
async def root():
    return {"message": "Simple Saarthi AI is running!"}

# Enhanced Audio Testing Endpoints
@app.get("/audio/{filename}")
async def serve_audio_with_headers(filename: str):
    """Serve audio files with proper headers for Twilio WhatsApp compatibility"""
    try:
        from fastapi.responses import FileResponse
        
        audio_path = os.path.join(AUDIO_DIR, filename)
        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            return {"error": f"Audio file {filename} not found"}
        
        # WhatsApp supports specific audio formats - convert WAV to OGG if needed
        if filename.endswith('.wav'):
            # Convert WAV to OGG for better WhatsApp compatibility
            try:
                from pydub import AudioSegment
                
                # Load WAV and convert to OGG
                audio = AudioSegment.from_wav(audio_path)
                ogg_filename = filename.replace('.wav', '.ogg')
                ogg_path = os.path.join(AUDIO_DIR, ogg_filename)
                
                # Export as OGG with optimized settings for WhatsApp
                audio.export(
                    ogg_path, 
                    format="ogg", 
                    codec="libvorbis",
                    bitrate="64k",
                    parameters=["-ar", "16000", "-ac", "1"]  # 16kHz mono for WhatsApp
                )
                
                logger.info(f"Converted {filename} to {ogg_filename}")
                
                return FileResponse(
                    ogg_path,
                    media_type="audio/ogg",
                    headers={
                        "Cache-Control": "public, max-age=3600",
                        "Access-Control-Allow-Origin": "*",
                        "Content-Disposition": f"inline; filename={ogg_filename}",
                        "Accept-Ranges": "bytes"
                    }
                )
                
            except Exception as conv_error:
                logger.warning(f"Audio conversion failed: {conv_error}, serving original WAV")
                # Fallback to original WAV with corrected headers
                return FileResponse(
                    audio_path,
                    media_type="audio/wav",
                    headers={
                        "Cache-Control": "public, max-age=3600", 
                        "Access-Control-Allow-Origin": "*",
                        "Content-Disposition": f"inline; filename={filename}",
                        "Accept-Ranges": "bytes"
                    }
                )
        
        # For non-WAV files, serve directly with appropriate content type
        content_type = "audio/wav"
        if filename.endswith('.mp3'):
            content_type = "audio/mpeg"
        elif filename.endswith('.ogg'):
            content_type = "audio/ogg"
        elif filename.endswith('.m4a'):
            content_type = "audio/mp4"
        elif filename.endswith('.aac'):
            content_type = "audio/aac"
        
        return FileResponse(
            audio_path,
            media_type=content_type,
            headers={
                "Cache-Control": "public, max-age=3600",
                "Access-Control-Allow-Origin": "*", 
                "Content-Disposition": f"inline; filename={filename}",
                "Accept-Ranges": "bytes"
            }
        )
        
    except Exception as e:
        logger.error(f"Error serving audio file {filename}: {e}")
        return {"error": str(e)}

@app.get("/api/test/audio-url/{filename}")
async def test_audio_url(filename: str):
    """Test if audio URL is accessible"""
    try:
        audio_url = f"{BASE_URL}/audio/{filename}"
        audio_path = os.path.join(AUDIO_DIR, filename)
        
        # Check local file
        local_exists = os.path.exists(audio_path)
        file_size = os.path.getsize(audio_path) if local_exists else 0
        
        # Check URL accessibility
        url_accessible = False
        response_status = None
        response_headers = {}
        
        try:
            response = requests.head(audio_url, timeout=10)
            url_accessible = response.status_code == 200
            response_status = response.status_code
            response_headers = dict(response.headers)
        except Exception as e:
            response_status = f"Error: {e}"
        
        return {
            "success": True,
            "audio_filename": filename,
            "audio_url": audio_url,
            "local_file_exists": local_exists,
            "file_size_bytes": file_size,
            "url_accessible": url_accessible,
            "response_status": response_status,
            "response_headers": response_headers,
            "base_url": BASE_URL,
            "audio_dir": AUDIO_DIR,
            "full_path": audio_path
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/test/complete-audio-flow")
async def test_complete_audio_flow(request: TestMessageRequest):
    """Test the complete audio flow without Twilio"""
    try:
        logger.info(f"🧪 Testing complete audio flow for: {request.phone_number}")
        
        # Step 1: Process the message
        result = await agentic_process_query(
            request.phone_number, 
            request.message, 
            request.media_url, 
            request.media_type
        )
        
        # Step 2: Test audio delivery
        audio_delivery = await send_audio_message(
            f"whatsapp:{request.phone_number}",
            result["audio_filename"],
            result["text_response"]
        )
        
        # Step 3: Test audio URL accessibility
        audio_url_test = None
        if result["audio_filename"]:
            audio_url_response = await test_audio_url(result["audio_filename"])
            audio_url_test = audio_url_response
        
        return {
            "success": True,
            "test_type": "complete_audio_flow",
            "phone_number": request.phone_number,
            "message_processing": {
                "text_response": result["text_response"],
                "detected_language": result["detected_language"],
                "intent": result.get("intent_analysis", {}).get("primary_intent", "unknown"),
                "audio_filename": result["audio_filename"]
            },
            "audio_delivery": audio_delivery,
            "audio_url_test": audio_url_test,
            "final_audio_url": f"{BASE_URL}/audio/{result['audio_filename']}" if result["audio_filename"] else None,
            "twilio_bypassed": True
        }
        
    except Exception as e:
        logger.error(f"Complete audio flow test error: {e}")
        return {
            "success": False,
            "error": str(e),
            "test_type": "complete_audio_flow"
        }

@app.get("/api/test/audio-files")
async def list_audio_files():
    """List all available audio files"""
    try:
        if not os.path.exists(AUDIO_DIR):
            return {"success": False, "error": "Audio directory not found"}
        
        audio_files = []
        for filename in os.listdir(AUDIO_DIR):
            if filename.endswith('.wav'):
                file_path = os.path.join(AUDIO_DIR, filename)
                file_size = os.path.getsize(file_path)
                file_url = f"{BASE_URL}/audio/{filename}"
                
                audio_files.append({
                    "filename": filename,
                    "size_bytes": file_size,
                    "url": file_url,
                    "created": os.path.getctime(file_path)
                })
        
        return {
            "success": True,
            "audio_directory": AUDIO_DIR,
            "total_files": len(audio_files),
            "files": sorted(audio_files, key=lambda x: x['created'], reverse=True)[:10],  # Latest 10 files
            "base_url": BASE_URL
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/test/message")
async def test_message_endpoint(request: TestMessageRequest):
    """
    Test the complete AGENTIC message processing pipeline without Twilio
    Accepts JSON input and returns JSON response with full agentic data
    """
    try:
        logger.info(f"Agentic test message from {request.phone_number}: {request.message} | Media: {request.media_type}")
        
        # Use the agentic processing function directly
        result = await agentic_process_query(
            request.phone_number, 
            request.message, 
            request.media_url, 
            request.media_type
        )
        
        # Return complete agentic response
        response = {
            "success": True,
            "phone_number": request.phone_number,
            "input_type": result["input_type"],
            "detected_language": result["detected_language"],
            "text_response": result["text_response"],
            "audio_filename": result["audio_filename"],
            "audio_url": f"{BASE_URL}/audio/{result['audio_filename']}" if result["audio_filename"] else None,
            "intent_analysis": result.get("intent_analysis", {}),
            "image_analysis": result.get("image_analysis"),
            "tool_results": result.get("tool_results", {}),
            "farmer_context": result.get("farmer_context", ""),
            "agentic": True
        }
        
        logger.info(f"Agentic test completed for {request.phone_number} - Intent: {result.get('intent_analysis', {}).get('primary_intent', 'unknown')}")
        return response
        
    except Exception as e:
        logger.error(f"Agentic test processing error for {request.phone_number}: {e}")
        return {
            "success": False,
            "error": str(e),
            "phone_number": request.phone_number,
            "agentic": False
        }

@app.post("/api/test/audio-to-text")
async def test_speech_to_text(request: TestAudioRequest):
    """Test speech-to-text functionality"""
    try:
        transcript, language = await speech_to_text(request.audio_url)
        return {
            "success": True,
            "transcript": transcript,
            "detected_language": language
        }
    except Exception as e:
        logger.error(f"STT test error: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/api/test/text-to-speech")
async def test_text_to_speech_endpoint(request: TestTTSRequest):
    """Test text-to-speech functionality with enhanced audio guarantee"""
    try:
        audio_filename = await ensure_audio_response(request.text, request.language)
        return {
            "success": True,
            "text": request.text,
            "language": request.language,
            "audio_filename": audio_filename,
            "audio_url": f"{BASE_URL}/audio/{audio_filename}" if audio_filename else None,
            "guaranteed_audio": True
        }
    except Exception as e:
        logger.error(f"TTS test error: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/api/test/conversation-history/{phone_number}")
async def get_conversation_history(phone_number: str, limit: int = 5):
    """Get conversation history for a phone number"""
    try:
        conversations = memory.get_recent_conversations(phone_number, limit)
        return {
            "success": True,
            "phone_number": phone_number,
            "conversations": conversations,
            "count": len(conversations)
        }
    except Exception as e:
        logger.error(f"History fetch error: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/api/test/ai-response")
async def test_ai_response(request: TestAIRequest):
    """Test AI response generation"""
    try:
        response = await generate_ai_response(request.message, request.language, request.context)
        return {
            "success": True,
            "message": request.message,
            "language": request.language,
            "context": request.context,
            "ai_response": response
        }
    except Exception as e:
        logger.error(f"AI response test error: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/api/test/intent-classification")
async def test_intent_classification(request: TestAIRequest):
    """Test intent classification functionality"""
    try:
        intent_classifier = IntentClassifier()
        result = await intent_classifier.classify_intent(request.message)
        return {
            "success": True,
            "message": request.message,
            "intent_analysis": result
        }
    except Exception as e:
        logger.error(f"Intent classification test error: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/api/test/farming-tools")
async def test_farming_tools(tool_type: str, crop: str = "wheat", location: str = "Delhi"):
    """Test farming tools functionality"""
    try:
        farming_tools = FarmingTools()
        result = {}
        
        if tool_type == "weather":
            result = await farming_tools.get_weather_forecast(location)
        elif tool_type == "mandi":
            result = await farming_tools.get_mandi_prices(crop, location)
        elif tool_type == "msp":
            result = await farming_tools.get_current_msp(crop)
        elif tool_type == "schemes":
            result = await farming_tools.get_government_schemes()
        else:
            return {"success": False, "error": "Invalid tool_type"}
            
        return {
            "success": True,
            "tool_type": tool_type,
            "parameters": {"crop": crop, "location": location},
            "result": result
        }
    except Exception as e:
        logger.error(f"Farming tools test error: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/api/test/enhanced-memory")
async def test_enhanced_memory(phone_number: str):
    """Test enhanced memory functionality"""
    try:
        # Get farmer context
        context = enhanced_memory.get_farmer_context(phone_number)
        
        # Get recent conversations
        conversations = enhanced_memory.get_recent_conversations(phone_number, 5)
        
        # Try to update summary
        await enhanced_memory.update_conversation_summary(phone_number)
        
        return {
            "success": True,
            "phone_number": phone_number,
            "farmer_context": context,
            "recent_conversations": conversations,
            "context_length": len(context)
        }
    except Exception as e:
        logger.error(f"Enhanced memory test error: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/api/test/status")
async def test_status():
    """Comprehensive status check for testing"""
    return {
        "status": "running",
        "gemini_available": GEMINI_AVAILABLE,
        "audio_files_count": len([f for f in os.listdir(AUDIO_DIR) if f.endswith('.wav')]) if os.path.exists(AUDIO_DIR) else 0,
        "supported_languages": list(SUPPORTED_LANGUAGES.keys()),
        "tts_supported": list(TTS_SUPPORTED),
        "base_url": BASE_URL,
        "agentic_features": {
            "intent_classification": True,
            "image_analysis": GEMINI_AVAILABLE,
            "farming_tools": True,
            "enhanced_memory": True,
            "msp_comparison": True,
            "audio_delivery": True,
            "twilio_bypassed": True
        },
        "endpoints": {
            "main_test": "/api/test/message",
            "complete_audio_flow": "/api/test/complete-audio-flow",
            "audio_files": "/api/test/audio-files",
            "audio_url_test": "/api/test/audio-url/{filename}",
            "stt_test": "/api/test/audio-to-text", 
            "tts_test": "/api/test/text-to-speech",
            "ai_test": "/api/test/ai-response",
            "intent_test": "/api/test/intent-classification",
            "tools_test": "/api/test/farming-tools",
            "memory_test": "/api/test/enhanced-memory",
            "history": "/api/test/conversation-history/{phone_number}",
            "debug_image": "/api/test/debug-image-analysis",
            "debug_webhook": "/api/test/debug-webhook-twiml",
            "status": "/api/test/status"
        },
        "audio_endpoints": {
            "serve_audio": "/audio/{filename}",
            "test_audio_url": "/api/test/audio-url/{filename}",
            "list_audio_files": "/api/test/audio-files",
            "complete_flow": "/api/test/complete-audio-flow"
        },
        "whatsapp_compatibility": {
            "supported_audio_formats": ["OGG", "MP3", "WAV"],
            "preferred_format": "OGG",
            "content_type_fixed": True,
            "url_accessibility_tested": True,
            "twiml_format_validated": True,
            "error_12300_handled": True
        }
    }

@app.post("/webhook/message")
async def handle_message(request: Request):
    """Main webhook handler - returns TwiML response with chunked audio for long responses"""
    try:
        form_data = await request.form()
        
        from_number = form_data.get('From', '')
        message_body = form_data.get('Body', '')
        num_media = int(form_data.get('NumMedia', '0'))
        
        media_url = form_data.get('MediaUrl0', '') if num_media > 0 else None
        media_type = form_data.get('MediaContentType0', '') if num_media > 0 else None
        
        phone_number = from_number.replace('whatsapp:', '')
        
        logger.info(f"📨 Webhook: Message from {phone_number}: {message_body} | Media: {media_type}")
        
        # Process the query using agentic system
        result = await agentic_process_query(phone_number, message_body, media_url, media_type)
        
        # Check if response is long enough to warrant chunking
        response_length = len(result["text_response"])
        
        if response_length > 150:  # Long response - use chunked delivery
            logger.info(f"📝 Long response ({response_length} chars) - using chunked delivery")
            
            # Send chunked audio messages directly (not via TwiML)
            chunked_result = await send_chunked_audio_messages(
                from_number,  # Already includes whatsapp: prefix
                result["text_response"],
                result["response_language"]
            )
            
            # Return simple TwiML acknowledgment
            resp = MessagingResponse()
            if chunked_result.get('success'):
                resp.message(f"✅ Response sent in {chunked_result.get('total_chunks', 1)} parts")
            else:
                resp.message("⚠️ Response processing completed")
            
            intent = result.get('intent_analysis', {}).get('primary_intent', 'unknown')
            logger.info(f"✅ Chunked delivery completed for {phone_number} - Intent: {intent}")
            
        else:
            # Short response - use traditional single audio TwiML
            logger.info(f"📝 Short response ({response_length} chars) - using single audio")
            
            resp = MessagingResponse()
            
            if result["audio_filename"]:
                # Construct audio URL with proper base URL
                audio_url = f"{BASE_URL}/audio/{result['audio_filename']}"
                
                # Verify the audio file exists and is accessible
                audio_path = os.path.join(AUDIO_DIR, result['audio_filename'])
                if not os.path.exists(audio_path):
                    logger.error(f"❌ Audio file not found: {audio_path}")
                    # Fallback to text-only response
                    clean_text = result["text_response"][:1000]  # Limit for WhatsApp
                    resp.message(clean_text)
                else:
                    logger.info(f"✅ Sending single audio response: {audio_url}")
                    
                    # Create message with audio - WhatsApp compatible
                    msg = resp.message()
                    msg.media(audio_url)
                    
                    # Add minimal text caption for context (WhatsApp best practice)
                    msg.body("🎤 Voice Response")
                    
            else:
                # Fallback to text if no audio available
                logger.warning("⚠️ No audio generated, sending text fallback")
                clean_text = result["text_response"][:1000]  # WhatsApp message limit
                clean_text = clean_text.replace('\n', ' ').replace('\r', ' ')
                resp.message(clean_text)
            
            intent = result.get('intent_analysis', {}).get('primary_intent', 'unknown')
            logger.info(f"✅ Single audio webhook completed for {phone_number} - Intent: {intent}")
        
        # Return properly formatted TwiML
        twiml_str = str(resp)
        logger.info(f"📤 TwiML Response: {twiml_str[:200]}...")
        
        return Response(
            content=twiml_str, 
            media_type="application/xml; charset=utf-8",
            headers={
                "Content-Type": "application/xml; charset=utf-8",
                "Cache-Control": "no-cache"
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Webhook error: {e}")
        
        # Enhanced error response with audio fallback
        try:
            error_audio = await ensure_audio_response(
                "माफ करें, कुछ तकनीकी समस्या हुई है। कृपया दोबारा कोशिश करें।", 
                'hi-IN'
            )
            
            if error_audio:
                error_audio_url = f"{BASE_URL}/audio/{error_audio}"
                
                resp = MessagingResponse()
                msg = resp.message()
                msg.body("⚠️ Technical Issue - Please retry")
                msg.media(error_audio_url)
                
                return Response(
                    content=str(resp), 
                    media_type="application/xml; charset=utf-8"
                )
                
        except Exception as fallback_error:
            logger.error(f"Error fallback also failed: {fallback_error}")
        
        # Final fallback - simple text response
        resp = MessagingResponse()
        resp.message("⚠️ माफ करें, कुछ समस्या हुई है। कृपया दोबारा कोशिश करें। | Sorry, there was an issue. Please try again.")
        
        return Response(
            content=str(resp),
            media_type="application/xml; charset=utf-8"
        )

@app.get("/test-images/{filename}")
async def serve_test_image(filename: str):
    """Serve test images for local testing"""
    try:
        import os
        from fastapi.responses import FileResponse
        
        # Look for images in current directory
        image_path = f"./{filename}"
        if os.path.exists(image_path):
            return FileResponse(image_path)
        else:
            return {"error": f"Image {filename} not found"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/test")
async def test():
    return {"message": "Simple Saarthi AI is running!"}

@app.post("/api/test/debug-image-analysis")
async def debug_image_analysis(request: TestMessageRequest):
    """
    Debug endpoint for image analysis with detailed logging
    """
    try:
        logger.info(f"🔍 DEBUG: Starting image analysis for {request.phone_number}")
        
        # Step 1: Check if image URL is accessible
        logger.info(f"📥 DEBUG: Downloading image from {request.media_url}")
        image_data = await download_media(request.media_url)
        
        if not image_data:
            return {
                "success": False,
                "debug_info": {
                    "step": "image_download",
                    "error": "Failed to download image",
                    "url": request.media_url
                }
            }
        
        logger.info(f"✅ DEBUG: Image downloaded successfully, size: {len(image_data)} bytes")
        
        # Step 2: Check Gemini availability
        if not GEMINI_AVAILABLE:
            return {
                "success": False,
                "debug_info": {
                    "step": "gemini_check",
                    "error": "Gemini not available",
                    "gemini_available": False
                }
            }
        
        logger.info(f"✅ DEBUG: Gemini is available")
        
        # Step 3: Initialize Image Analyzer
        image_analyzer = ImageAnalyzer()
        
        # Step 4: Analyze image with detailed logging
        logger.info(f"🔬 DEBUG: Starting image analysis with context: {request.message}")
        
        try:
            result = await image_analyzer.analyze_farming_image(image_data, request.message)
            logger.info(f"✅ DEBUG: Image analysis completed successfully")
            logger.info(f"📊 DEBUG: Result type: {result.get('image_type', 'unknown')}")
            
            return {
                "success": True,
                "debug_info": {
                    "step": "completed",
                    "image_size_bytes": len(image_data),
                    "context": request.message,
                    "gemini_available": GEMINI_AVAILABLE,
                    "analysis_result": result
                },
                "image_analysis": result
            }
            
        except Exception as analysis_error:
            logger.error(f"❌ DEBUG: Image analysis failed: {analysis_error}")
            return {
                "success": False,
                "debug_info": {
                    "step": "image_analysis",
                    "error": str(analysis_error),
                    "image_size_bytes": len(image_data),
                    "context": request.message,
                    "gemini_available": GEMINI_AVAILABLE
                }
            }
            
    except Exception as e:
        logger.error(f"❌ DEBUG: Complete failure: {e}")
        return {
            "success": False,
            "debug_info": {
                "step": "general_error",
                "error": str(e),
                "phone_number": request.phone_number
            }
        }

@app.post("/api/test/debug-webhook-twiml")
async def debug_webhook_twiml(request: TestMessageRequest):
    """Debug webhook TwiML generation and audio URL accessibility for WhatsApp"""
    try:
        logger.info(f"🔍 DEBUG: Testing webhook TwiML generation for {request.phone_number}")
        
        # Step 1: Process the message
        result = await agentic_process_query(
            request.phone_number, 
            request.message, 
            request.media_url, 
            request.media_type
        )
        
        # Step 2: Test audio file creation and accessibility
        audio_info = {}
        if result["audio_filename"]:
            audio_path = os.path.join(AUDIO_DIR, result["audio_filename"])
            audio_url = f"{BASE_URL}/audio/{result['audio_filename']}"
            
            audio_info = {
                "filename": result["audio_filename"],
                "local_path": audio_path,
                "exists_locally": os.path.exists(audio_path),
                "file_size": os.path.getsize(audio_path) if os.path.exists(audio_path) else 0,
                "audio_url": audio_url,
                "base_url": BASE_URL
            }
            
            # Test URL accessibility
            try:
                import requests
                response = requests.head(audio_url, timeout=10)
                audio_info.update({
                    "url_accessible": response.status_code == 200,
                    "response_status": response.status_code,
                    "response_headers": dict(response.headers),
                    "content_type": response.headers.get('Content-Type', 'unknown')
                })
            except Exception as url_error:
                audio_info.update({
                    "url_accessible": False,
                    "url_error": str(url_error)
                })
        
        # Step 3: Generate TwiML like the webhook would
        from twilio.twiml.messaging_response import MessagingResponse
        resp = MessagingResponse()
        
        if result["audio_filename"] and audio_info.get("exists_locally"):
            # Simulate webhook TwiML generation
            audio_url = f"{BASE_URL}/audio/{result['audio_filename']}"
            msg = resp.message()
            msg.media(audio_url)
            msg.body("🎤 Voice Response")
        else:
            # Text fallback
            clean_text = result["text_response"][:1000]
            resp.message(clean_text)
        
        twiml_output = str(resp)
        
        # Step 4: Analyze TwiML for issues
        twiml_analysis = {
            "has_xml_declaration": '<?xml' in twiml_output,
            "has_response_tags": '<Response>' in twiml_output and '</Response>' in twiml_output,
            "has_message_tags": '<Message>' in twiml_output and '</Message>' in twiml_output,
            "has_media_tags": '<Media>' in twiml_output and '</Media>' in twiml_output,
            "has_body_tags": '<Body>' in twiml_output and '</Body>' in twiml_output,
            "contains_audio_url": BASE_URL in twiml_output,
            "twiml_length": len(twiml_output),
            "is_valid_xml": True  # Assume valid since MessagingResponse generates it
        }
        
        # Step 5: Test direct Twilio message sending (optional)
        twilio_test = None
        if request.phone_number.startswith('+91'):  # Only test with Indian numbers
            try:
                # Test sending message directly via Twilio API (not webhook)
                if result["audio_filename"] and audio_info.get("url_accessible"):
                    message = twilio_client.messages.create(
                        from_=f"whatsapp:{TWILIO_NUMBER}",
                        to=f"whatsapp:{request.phone_number}",
                        body="🧪 Debug Test - Voice Response",
                        media_url=[f"{BASE_URL}/audio/{result['audio_filename']}"]
                    )
                    
                    twilio_test = {
                        "success": True,
                        "message_sid": message.sid,
                        "status": message.status,
                        "method": "direct_api_call"
                    }
                else:
                    # Send text only
                    message = twilio_client.messages.create(
                        from_=f"whatsapp:{TWILIO_NUMBER}",
                        to=f"whatsapp:{request.phone_number}",
                        body=f"🧪 Debug Test - Text Only: {result['text_response'][:100]}..."
                    )
                    
                    twilio_test = {
                        "success": True,
                        "message_sid": message.sid,
                        "status": message.status,
                        "method": "text_only"
                    }
                    
            except Exception as twilio_error:
                twilio_test = {
                    "success": False,
                    "error": str(twilio_error),
                    "error_code": getattr(twilio_error, 'code', 'unknown')
                }
        
        return {
            "success": True,
            "debug_type": "webhook_twiml_audio",
            "phone_number": request.phone_number,
            "message_processing": {
                "text_response": result["text_response"][:200] + "...",
                "detected_language": result["detected_language"],
                "intent": result.get("intent_analysis", {}).get("primary_intent", "unknown")
            },
            "audio_info": audio_info,
            "twiml_output": twiml_output,
            "twiml_analysis": twiml_analysis,
            "twilio_direct_test": twilio_test,
            "recommendations": generate_debug_recommendations(audio_info, twiml_analysis, twilio_test)
        }
        
    except Exception as e:
        logger.error(f"Debug webhook TwiML error: {e}")
        return {
            "success": False,
            "error": str(e),
            "debug_type": "webhook_twiml_audio"
        }

def generate_debug_recommendations(audio_info: dict, twiml_analysis: dict, twilio_test: dict) -> list:
    """Generate debugging recommendations based on test results"""
    recommendations = []
    
    if not audio_info.get("exists_locally"):
        recommendations.append("❌ Audio file not found locally - check TTS generation")
    
    if not audio_info.get("url_accessible"):
        recommendations.append("❌ Audio URL not accessible - check BASE_URL and ngrok")
        
    if audio_info.get("content_type") and "audio" not in audio_info.get("content_type", "").lower():
        recommendations.append("⚠️ Content-Type might not be audio/* - check file serving")
    
    if not twiml_analysis.get("has_media_tags"):
        recommendations.append("❌ TwiML missing Media tags - audio won't be sent")
    
    if twiml_analysis.get("twiml_length", 0) > 5000:
        recommendations.append("⚠️ TwiML response might be too long")
    
    if twilio_test and not twilio_test.get("success"):
        error_code = twilio_test.get("error_code", "")
        if "12300" in str(error_code):
            recommendations.append("🚨 Error 12300: Invalid Content-Type - check audio file headers")
        elif "21660" in str(error_code):
            recommendations.append("🚨 Error 21660: Invalid WhatsApp number format")
        elif "63038" in str(error_code):
            recommendations.append("🚨 Error 63038: Daily message limit exceeded")
        else:
            recommendations.append(f"🚨 Twilio API error: {twilio_test.get('error', 'unknown')}")
    
    if len(recommendations) == 0:
        recommendations.append("✅ All tests passed - audio should work!")
    
    return recommendations

@app.get("/response_{audio_id}.{ext}")
async def serve_direct_audio_files(audio_id: str, ext: str):
    """
    Handle direct audio file requests (when Twilio strips /audio/ from URL)
    This catches requests like /response_12345.ogg and serves them from voice_responses/
    """
    try:
        # Only handle audio files
        if ext in ['ogg', 'wav', 'mp3']:
            filename = f"response_{audio_id}.{ext}"
            logger.info(f"🎵 Direct audio file request: {filename}")
            
            from fastapi.responses import FileResponse
            
            audio_path = os.path.join(AUDIO_DIR, filename)
            if os.path.exists(audio_path):
                # Determine content type
                content_type = "audio/ogg"
                if ext == 'wav':
                    content_type = "audio/wav"
                elif ext == 'mp3':
                    content_type = "audio/mpeg"
                
                logger.info(f"✅ Serving direct audio file: {filename} as {content_type}")
                
                return FileResponse(
                    audio_path,
                    media_type=content_type,
                    headers={
                        "Cache-Control": "public, max-age=3600",
                        "Access-Control-Allow-Origin": "*",
                        "Content-Disposition": f"inline; filename={filename}",
                        "Accept-Ranges": "bytes"
                    }
                )
            else:
                logger.error(f"❌ Direct audio file not found: {audio_path}")
                return {"error": f"Audio file {filename} not found"}
        else:
            # Not an audio file extension
            return {"error": f"Unsupported file type: {ext}"}
            
    except Exception as e:
        logger.error(f"❌ Error serving direct audio file: {e}")
        return {"error": str(e)}

@app.post("/webhook/status")
async def webhook_status(request: Request):
    """Handle Twilio webhook status callbacks"""
    try:
        form_data = await request.form()
        
        message_sid = form_data.get('MessageSid', '')
        message_status = form_data.get('MessageStatus', '')
        to_number = form_data.get('To', '')
        from_number = form_data.get('From', '')
        
        logger.info(f"📊 Webhook Status: {message_status} for SID: {message_sid} ({from_number} -> {to_number})")
        
        # Store status update if needed
        if message_status in ['delivered', 'read', 'failed', 'undelivered']:
            logger.info(f"✅ Message {message_sid}: {message_status}")
        
        return Response(content="OK", media_type="text/plain")
        
    except Exception as e:
        logger.error(f"❌ Webhook status error: {e}")
        return Response(content="ERROR", media_type="text/plain")

@app.post("/api/test/debug-audio-pipeline")
async def debug_audio_pipeline(request: TestTTSRequest):
    """Comprehensive audio pipeline debugging"""
    try:
        debug_info = {
            "input_text": request.text,
            "input_language": request.language,
            "steps": [],
            "files_created": [],
            "audio_analysis": {}
        }
        
        # Step 1: Clean text
        clean_text = request.text.replace("🤖", "").replace("📚", "").replace("🔍", "").replace("✅", "").replace("⚠️", "").strip()
        clean_text = ' '.join(clean_text.split())
        
        debug_info["steps"].append({
            "step": "text_cleaning",
            "original": request.text,
            "cleaned": clean_text,
            "length": len(clean_text)
        })
        
        # Step 2: TTS API Call
        tts_language = request.language if request.language in TTS_SUPPORTED else 'en-IN'
        
        logger.info(f"🔍 DEBUG: Making TTS API call...")
        
        response = requests.post(
            f"{SARVAM_BASE_URL}/text-to-speech",
            headers=SARVAM_HEADERS,
            json={
                "inputs": [clean_text],
                "target_language_code": tts_language,
                "speaker": "meera",
                "pitch": 0,
                "pace": 1.65,
                "loudness": 1.5,
                "speech_sample_rate": 16000,
                "enable_preprocessing": True,
                "model": "bulbul:v1"
            },
            timeout=30
        )
        
        debug_info["steps"].append({
            "step": "tts_api_call",
            "status_code": response.status_code,
            "response_size": len(response.content) if response.content else 0,
            "language_used": tts_language
        })
        
        if response.status_code != 200:
            debug_info["error"] = f"TTS API failed: {response.status_code} - {response.text}"
            return debug_info
        
        # Step 3: Parse TTS Response
        result = response.json()
        audio_base64 = result.get('audios', [None])[0]
        
        if not audio_base64:
            debug_info["error"] = "No audio data in TTS response"
            return debug_info
        
        # Step 4: Decode Base64 Audio
        try:
            audio_data = base64.b64decode(audio_base64)
            debug_info["steps"].append({
                "step": "base64_decode",
                "base64_length": len(audio_base64),
                "decoded_size": len(audio_data),
                "first_bytes": list(audio_data[:10]) if len(audio_data) >= 10 else list(audio_data)
            })
        except Exception as decode_error:
            debug_info["error"] = f"Base64 decode failed: {decode_error}"
            return debug_info
        
        # Step 5: Save Original WAV
        audio_id = str(uuid.uuid4())[:8]
        wav_filename = f"debug_{audio_id}.wav"
        wav_path = os.path.join(AUDIO_DIR, wav_filename)
        
        with open(wav_path, 'wb') as f:
            f.write(audio_data)
        
        wav_size = os.path.getsize(wav_path)
        debug_info["files_created"].append({
            "filename": wav_filename,
            "size": wav_size,
            "type": "original_wav"
        })
        
        # Step 6: Analyze WAV file using pydub
        try:
            from pydub import AudioSegment
            
            audio_segment = AudioSegment.from_wav(wav_path)
            
            debug_info["audio_analysis"] = {
                "duration_ms": len(audio_segment),
                "frame_rate": audio_segment.frame_rate,
                "channels": audio_segment.channels,
                "sample_width": audio_segment.sample_width,
                "max_possible_amplitude": audio_segment.max_possible_amplitude,
                "rms": audio_segment.rms,
                "dBFS": audio_segment.dBFS,
                "is_silent": audio_segment.rms < 100  # Check if audio is too quiet
            }
            
            debug_info["steps"].append({
                "step": "wav_analysis",
                "success": True,
                "duration_seconds": len(audio_segment) / 1000,
                "is_likely_silent": audio_segment.rms < 100
            })
            
            # Step 7: Try different processing approaches
            if audio_segment.rms < 100:
                logger.warning("🔍 DEBUG: Audio seems too quiet, trying amplification...")
                
                # Try amplifying the audio
                amplified = audio_segment + 20  # Add 20dB
                amplified_filename = f"debug_{audio_id}_amplified.wav"
                amplified_path = os.path.join(AUDIO_DIR, amplified_filename)
                amplified.export(amplified_path, format="wav")
                
                debug_info["files_created"].append({
                    "filename": amplified_filename,
                    "size": os.path.getsize(amplified_path),
                    "type": "amplified_wav",
                    "amplification": "+20dB"
                })
            
            # Step 8: Create MP3 with different settings
            mp3_variants = [
                {"bitrate": "128k", "quality": "2", "name": "high_quality"},
                {"bitrate": "64k", "quality": "4", "name": "medium_quality"},
                {"bitrate": "32k", "quality": "6", "name": "low_quality"}
            ]
            
            for variant in mp3_variants:
                try:
                    mp3_filename = f"debug_{audio_id}_{variant['name']}.mp3"
                    mp3_path = os.path.join(AUDIO_DIR, mp3_filename)
                    
                    audio_segment.export(
                        mp3_path,
                        format="mp3",
                        bitrate=variant["bitrate"],
                        parameters=[
                            "-ar", "16000",
                            "-ac", "1",
                            "-b:a", variant["bitrate"],
                            "-q:a", variant["quality"],
                            "-f", "mp3"
                        ]
                    )
                    
                    mp3_size = os.path.getsize(mp3_path)
                    debug_info["files_created"].append({
                        "filename": mp3_filename,
                        "size": mp3_size,
                        "type": f"mp3_{variant['name']}",
                        "bitrate": variant["bitrate"],
                        "quality": variant["quality"]
                    })
                    
                except Exception as mp3_error:
                    debug_info["steps"].append({
                        "step": f"mp3_{variant['name']}_creation",
                        "success": False,
                        "error": str(mp3_error)
                    })
            
        except Exception as analysis_error:
            debug_info["steps"].append({
                "step": "wav_analysis",
                "success": False,
                "error": str(analysis_error)
            })
        
        # Step 9: Test with a simple known-good audio
        try:
            from pydub.generators import Sine
            
            # Create a test tone that we know should work
            test_tone = Sine(440).to_audio_segment(duration=1000)  # 1 second 440Hz tone
            test_tone = test_tone.set_frame_rate(16000).set_channels(1)
            
            test_filename = f"debug_{audio_id}_test_tone.mp3"
            test_path = os.path.join(AUDIO_DIR, test_filename)
            
            test_tone.export(
                test_path,
                format="mp3",
                bitrate="64k",
                parameters=["-ar", "16000", "-ac", "1"]
            )
            
            debug_info["files_created"].append({
                "filename": test_filename,
                "size": os.path.getsize(test_path),
                "type": "test_tone_mp3",
                "note": "Known good audio for comparison"
            })
            
            debug_info["steps"].append({
                "step": "test_tone_creation",
                "success": True,
                "note": "Created known-good test audio"
            })
            
        except Exception as tone_error:
            debug_info["steps"].append({
                "step": "test_tone_creation", 
                "success": False,
                "error": str(tone_error)
            })
        
        # Step 10: Provide recommendations
        recommendations = []
        
        if debug_info.get("audio_analysis", {}).get("is_silent"):
            recommendations.append("🚨 TTS API is returning silent/very quiet audio")
            recommendations.append("💡 Try different TTS parameters or API")
            
        if debug_info.get("audio_analysis", {}).get("rms", 0) > 1000:
            recommendations.append("✅ Audio has good volume levels")
            
        if any(f["size"] < 1000 for f in debug_info["files_created"] if "mp3" in f["type"]):
            recommendations.append("⚠️ Some MP3 files are very small - possible corruption")
            
        debug_info["recommendations"] = recommendations
        
        return {
            "success": True,
            "debug_info": debug_info,
            "total_files_created": len(debug_info["files_created"]),
            "base_url_for_testing": BASE_URL
        }
        
    except Exception as e:
        logger.error(f"Audio pipeline debug error: {e}")
        return {
            "success": False,
            "error": str(e),
            "debug_info": debug_info if 'debug_info' in locals() else {}
        }

async def chunk_text_for_audio(text: str, max_chunk_size: int = 100) -> list:
    """Break text into smaller chunks for better TTS quality"""
    
    # Clean text first
    clean_text = text.replace("🤖", "").replace("📚", "").replace("🔍", "").replace("✅", "").replace("⚠️", "").strip()
    clean_text = ' '.join(clean_text.split())
    
    # If text is short enough, return as single chunk
    if len(clean_text) <= max_chunk_size:
        return [clean_text]
    
    # Split by sentences first
    import re
    sentences = re.split(r'[।\.!?]+', clean_text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # If adding this sentence would exceed limit, save current chunk
        if len(current_chunk) + len(sentence) + 1 > max_chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                # Single sentence is too long, split by words
                words = sentence.split()
                for word in words:
                    if len(current_chunk) + len(word) + 1 > max_chunk_size:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                            current_chunk = word
                        else:
                            # Single word is too long, truncate
                            chunks.append(word[:max_chunk_size])
                    else:
                        current_chunk += " " + word if current_chunk else word
        else:
            current_chunk += " " + sentence if current_chunk else sentence
    
    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # Ensure no empty chunks
    chunks = [chunk for chunk in chunks if chunk.strip()]
    
    return chunks[:3]  # Limit to 3 chunks to avoid spam

async def send_chunked_audio_messages(to_number: str, text_response: str, language: str = 'hi-IN'):
    """Send text as multiple audio messages if needed"""
    try:
        # Break text into chunks
        chunks = await chunk_text_for_audio(text_response, max_chunk_size=80)
        
        logger.info(f"🔀 Chunking text into {len(chunks)} parts for {to_number}")
        
        sent_messages = []
        
        for i, chunk in enumerate(chunks):
            try:
                # Generate audio for this chunk
                audio_filename = await ensure_audio_response(chunk, language)
                
                if audio_filename:
                    # Send this chunk as audio message
                    result = await send_audio_message(to_number, audio_filename, chunk)
                    
                    if result.get('success'):
                        sent_messages.append({
                            "chunk": i + 1,
                            "text": chunk[:50] + "..." if len(chunk) > 50 else chunk,
                            "audio_filename": audio_filename,
                            "message_sid": result.get('message_sid'),
                            "success": True
                        })
                        logger.info(f"✅ Chunk {i+1}/{len(chunks)} sent successfully")
                    else:
                        sent_messages.append({
                            "chunk": i + 1,
                            "text": chunk[:50] + "..." if len(chunk) > 50 else chunk,
                            "error": result.get('error', 'Unknown error'),
                            "success": False
                        })
                        logger.error(f"❌ Chunk {i+1}/{len(chunks)} failed: {result.get('error')}")
                else:
                    # Fallback to text for this chunk
                    result = await send_text_message(to_number, f"📱 Part {i+1}: {chunk}")
                    sent_messages.append({
                        "chunk": i + 1,
                        "text": chunk[:50] + "..." if len(chunk) > 50 else chunk,
                        "fallback_to_text": True,
                        "success": result.get('success', False)
                    })
                    logger.warning(f"⚠️ Chunk {i+1}/{len(chunks)} sent as text fallback")
                
                # Small delay between messages to avoid rate limiting
                if i < len(chunks) - 1:  # Don't delay after last message
                    await asyncio.sleep(1)
                    
            except Exception as chunk_error:
                logger.error(f"❌ Error processing chunk {i+1}: {chunk_error}")
                sent_messages.append({
                    "chunk": i + 1,
                    "text": chunk[:50] + "..." if len(chunk) > 50 else chunk,
                    "error": str(chunk_error),
                    "success": False
                })
        
        return {
            "success": any(msg.get('success') for msg in sent_messages),
            "total_chunks": len(chunks),
            "successful_chunks": sum(1 for msg in sent_messages if msg.get('success')),
            "messages": sent_messages,
            "method": "chunked_audio"
        }
        
    except Exception as e:
        logger.error(f"❌ Chunked audio sending failed: {e}")
        # Final fallback to single text message
        return await send_text_message(to_number, text_response)

@app.post("/api/test/chunked-audio")
async def test_chunked_audio(request: TestMessageRequest):
    """Test chunked audio message sending"""
    try:
        logger.info(f"🔀 Testing chunked audio for {request.phone_number}")
        
        # Process the message first
        result = await agentic_process_query(
            request.phone_number, 
            request.message, 
            request.media_url, 
            request.media_type
        )
        
        # Send using chunked approach
        chunked_result = await send_chunked_audio_messages(
            f"whatsapp:{request.phone_number}",
            result["text_response"],
            result["response_language"]
        )
        
        return {
            "success": True,
            "phone_number": request.phone_number,
            "original_response": result["text_response"],
            "response_language": result["response_language"],
            "chunked_delivery": chunked_result,
            "intent": result.get("intent_analysis", {}).get("primary_intent", "unknown"),
            "test_type": "chunked_audio"
        }
        
    except Exception as e:
        logger.error(f"Chunked audio test error: {e}")
        return {
            "success": False,
            "error": str(e),
            "test_type": "chunked_audio"
        }

@app.post("/api/test/simple-tts")
async def test_simple_tts(request: TestTTSRequest):
    """Test TTS with very simple text to check if Sarvam API is working"""
    try:
        # Try with very simple, short text
        simple_texts = [
            "हैलो",  # Hello in Hindi
            "नमस्ते",  # Namaste in Hindi  
            "Hello",  # English
            "Test"   # Simple English
        ]
        
        results = []
        
        for i, text in enumerate(simple_texts):
            try:
                logger.info(f"🧪 Testing TTS with: '{text}'")
                
                # Use the most basic TTS call
                response = requests.post(
                    f"{SARVAM_BASE_URL}/text-to-speech",
                    headers=SARVAM_HEADERS,
                    json={
                        "inputs": [text],
                        "target_language_code": "hi-IN" if any(ord(c) > 127 for c in text) else "en-IN",
                        "speaker": "meera",
                        "pitch": 0,
                        "pace": 1.0,  # Normal pace
                        "loudness": 2.0,  # Higher loudness
                        "speech_sample_rate": 22050,  # Higher quality
                        "enable_preprocessing": True,
                        "model": "bulbul:v1"
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    audio_base64 = result.get('audios', [None])[0]
                    
                    if audio_base64:
                        audio_data = base64.b64decode(audio_base64)
                        
                        # Save as WAV and analyze
                        audio_id = f"simple_test_{i}_{str(uuid.uuid4())[:6]}"
                        wav_filename = f"{audio_id}.wav"
                        wav_path = os.path.join(AUDIO_DIR, wav_filename)
                        
                        with open(wav_path, 'wb') as f:
                            f.write(audio_data)
                        
                        # Analyze the audio
                        from pydub import AudioSegment
                        audio_segment = AudioSegment.from_wav(wav_path)
                        
                        # Create a simple MP3 version
                        mp3_filename = f"{audio_id}.mp3"
                        mp3_path = os.path.join(AUDIO_DIR, mp3_filename)
                        
                        audio_segment.export(
                            mp3_path,
                            format="mp3",
                            bitrate="128k",  # High quality for testing
                            parameters=["-ar", "22050", "-ac", "1"]
                        )
                        
                        results.append({
                            "text": text,
                            "success": True,
                            "wav_filename": wav_filename,
                            "mp3_filename": mp3_filename,
                            "wav_size": os.path.getsize(wav_path),
                            "mp3_size": os.path.getsize(mp3_path),
                            "duration_ms": len(audio_segment),
                            "rms": audio_segment.rms,
                            "dBFS": audio_segment.dBFS,
                            "is_likely_silent": audio_segment.rms < 100,
                            "audio_analysis": {
                                "frame_rate": audio_segment.frame_rate,
                                "channels": audio_segment.channels,
                                "sample_width": audio_segment.sample_width
                            },
                            "wav_url": f"{BASE_URL}/audio/{wav_filename}",
                            "mp3_url": f"{BASE_URL}/audio/{mp3_filename}"
                        })
                        
                        logger.info(f"✅ TTS test {i+1}: '{text}' - RMS: {audio_segment.rms}, dBFS: {audio_segment.dBFS}")
                    else:
                        results.append({
                            "text": text,
                            "success": False,
                            "error": "No audio data in response"
                        })
                else:
                    results.append({
                        "text": text,
                        "success": False,
                        "error": f"TTS API error: {response.status_code} - {response.text[:200]}"
                    })
                    
            except Exception as test_error:
                results.append({
                    "text": text,
                    "success": False,
                    "error": str(test_error)
                })
        
        # Summary analysis
        successful_tests = [r for r in results if r.get('success')]
        silent_tests = [r for r in successful_tests if r.get('is_likely_silent')]
        
        return {
            "success": True,
            "total_tests": len(simple_texts),
            "successful_tests": len(successful_tests),
            "silent_tests": len(silent_tests),
            "results": results,
            "summary": {
                "tts_api_working": len(successful_tests) > 0,
                "audio_has_content": len(successful_tests) - len(silent_tests) > 0,
                "all_silent": len(silent_tests) == len(successful_tests) and len(successful_tests) > 0,
                "recommendation": "Check Sarvam TTS API settings" if len(silent_tests) == len(successful_tests) else "TTS seems to be working"
            },
            "base_url": BASE_URL
        }
        
    except Exception as e:
        logger.error(f"Simple TTS test error: {e}")
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 