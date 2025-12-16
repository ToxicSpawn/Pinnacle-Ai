from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import os
from loguru import logger

try:
    from fastapi.templating import Jinja2Templates
    templates = None  # Will be set after creating directory
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False
    logger.warning("Jinja2 not available. Web interface will be limited.")

app = FastAPI()

# Create templates directory
os.makedirs("templates", exist_ok=True)

# Create index.html
INDEX_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pinnacle-AI</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #fff;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
        }
        header {
            text-align: center;
            padding: 40px 0;
        }
        h1 {
            font-size: 3em;
            background: linear-gradient(90deg, #00d9ff, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        .subtitle {
            color: #888;
            font-size: 1.2em;
        }
        .chat-container {
            background: rgba(255,255,255,0.05);
            border-radius: 20px;
            padding: 20px;
            margin: 20px 0;
            min-height: 400px;
            max-height: 500px;
            overflow-y: auto;
        }
        .message {
            margin: 15px 0;
            padding: 15px;
            border-radius: 15px;
            max-width: 80%;
        }
        .user-message {
            background: #00d9ff;
            color: #000;
            margin-left: auto;
            text-align: right;
        }
        .ai-message {
            background: rgba(255,255,255,0.1);
            color: #fff;
        }
        .input-container {
            display: flex;
            gap: 10px;
            margin-top: 20px;
        }
        #userInput {
            flex: 1;
            padding: 15px 20px;
            border: none;
            border-radius: 25px;
            font-size: 16px;
            background: rgba(255,255,255,0.1);
            color: #fff;
        }
        #userInput::placeholder {
            color: #888;
        }
        #sendBtn {
            padding: 15px 30px;
            border: none;
            border-radius: 25px;
            background: linear-gradient(90deg, #00d9ff, #00ff88);
            color: #000;
            font-weight: bold;
            cursor: pointer;
            transition: transform 0.2s;
        }
        #sendBtn:hover {
            transform: scale(1.05);
        }
        .status {
            text-align: center;
            padding: 20px;
            background: rgba(0,217,255,0.1);
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .features {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 30px;
        }
        .feature {
            background: rgba(255,255,255,0.05);
            padding: 20px;
            border-radius: 15px;
            text-align: center;
        }
        .feature-icon {
            font-size: 2em;
            margin-bottom: 10px;
        }
        .loading {
            display: none;
            text-align: center;
            padding: 10px;
        }
        .loading.show {
            display: block;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🚀 Pinnacle-AI</h1>
            <p class="subtitle">The Ultimate AGI System</p>
        </header>
        
        <div class="status" id="status">
            <span id="statusText">Connecting...</span>
        </div>
        
        <div class="chat-container" id="chatContainer">
            <div class="message ai-message">
                Hello! I'm Pinnacle-AI, an advanced artificial general intelligence. 
                I have infinite memory, emotional awareness, causal reasoning, and self-improvement capabilities. 
                How can I help you today?
            </div>
        </div>
        
        <div class="loading" id="loading">
            <span>Thinking...</span>
        </div>
        
        <div class="input-container">
            <input type="text" id="userInput" placeholder="Ask me anything..." onkeypress="handleKeyPress(event)">
            <button id="sendBtn" onclick="sendMessage()">Send</button>
        </div>
        
        <div class="features">
            <div class="feature">
                <div class="feature-icon">🧠</div>
                <div>Infinite Memory</div>
            </div>
            <div class="feature">
                <div class="feature-icon">❤️</div>
                <div>Emotional AI</div>
            </div>
            <div class="feature">
                <div class="feature-icon">⚡</div>
                <div>Causal Reasoning</div>
            </div>
            <div class="feature">
                <div class="feature-icon">🔄</div>
                <div>Self-Evolution</div>
            </div>
        </div>
    </div>
    
    <script>
        const API_URL = 'http://localhost:8000';
        
        async function checkHealth() {
            try {
                const response = await fetch(`${API_URL}/health`);
                const data = await response.json();
                document.getElementById('statusText').textContent = 
                    data.ai_loaded ? '✅ AI Ready' : '⏳ Loading AI...';
            } catch (error) {
                document.getElementById('statusText').textContent = '❌ Connection Error';
            }
        }
        
        async function sendMessage() {
            const input = document.getElementById('userInput');
            const message = input.value.trim();
            
            if (!message) return;
            
            // Add user message
            addMessage(message, 'user');
            input.value = '';
            
            // Show loading
            document.getElementById('loading').classList.add('show');
            
            try {
                const response = await fetch(`${API_URL}/generate`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({prompt: message, max_tokens: 500})
                });
                
                const data = await response.json();
                addMessage(data.response, 'ai');
            } catch (error) {
                addMessage('Sorry, I encountered an error. Please try again.', 'ai');
            }
            
            document.getElementById('loading').classList.remove('show');
        }
        
        function addMessage(text, sender) {
            const container = document.getElementById('chatContainer');
            const div = document.createElement('div');
            div.className = `message ${sender}-message`;
            div.textContent = text;
            container.appendChild(div);
            container.scrollTop = container.scrollHeight;
        }
        
        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }
        
        // Check health on load
        checkHealth();
        setInterval(checkHealth, 30000);
    </script>
</body>
</html>
'''

# Save the HTML file
with open("templates/index.html", "w") as f:
    f.write(INDEX_HTML)

if JINJA2_AVAILABLE:
    try:
        templates = Jinja2Templates(directory="templates")
    except:
        templates = None
else:
    templates = None

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    if templates:
        return templates.TemplateResponse("index.html", {"request": request})
    else:
        # Return HTML directly
        return HTMLResponse(content=INDEX_HTML)

