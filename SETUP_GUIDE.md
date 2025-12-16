# 🚀 Pinnacle AI Setup Guide

## Quick Start

### 1. Create `.env` File

Create a `.env` file in the root directory with your API keys:

```bash
# Copy the example file
cp .env.example .env

# Edit .env with your actual API keys
# NEVER commit .env to git!
```

### 2. Configure Settings

The `config/settings.yaml` file is already configured to use environment variables. It will automatically load values from your `.env` file.

### 3. Test Configuration

```bash
python test_config.py
```

This will verify:
- Configuration file loads correctly
- Environment variables are resolved
- API keys are present
- API connectivity (optional)

### 4. Run Pinnacle AI

```bash
# Interactive mode
python src/main.py --interactive

# Web interface
python src/main.py --web

# API server
python src/main.py --api

# Benchmark
python src/main.py --benchmark
```

## 📋 Required API Keys

### Essential (Required)
- **OpenAI API Key**: For LLM functionality
  - Get it at: https://platform.openai.com/api-keys
  - Format: `sk-proj-...`

- **Serper API Key**: For web search
  - Get it at: https://serper.dev/api-keys
  - Format: Alphanumeric string

### Optional (Recommended)
- **Stability AI Key**: For image generation
  - Get it at: https://platform.stability.ai/account/keys
  - Format: `sk-...`

- **ElevenLabs API Key**: For audio generation
  - Get it at: https://elevenlabs.io/app/settings/api-keys
  - Format: `sk_...`

- **Anthropic API Key**: Alternative LLM provider
  - Get it at: https://console.anthropic.com/settings/keys
  - Format: `sk-ant-api03-...`

- **Google API Key**: For Google services
  - Get it at: https://console.cloud.google.com/apis/credentials

- **IBM Quantum API Key**: For quantum computing features
  - Get it at: https://quantum-computing.ibm.com/

## 🔒 Security Best Practices

1. **Never commit `.env` to git** - It's already in `.gitignore`
2. **Never commit `config/settings.yaml` with real keys** - Use environment variables
3. **Rotate keys regularly** - Every 90 days recommended
4. **Use different keys for different environments** - Development, staging, production
5. **Monitor API usage** - Set up alerts for unusual activity

See `SECURITY_NOTICE.md` for detailed security guidelines.

## 📝 Example `.env` File

```bash
# .env
# OpenAI API Key (https://platform.openai.com)
OPENAI_API_KEY="sk-proj-your-actual-key-here"

# Serper API Key (https://serper.dev)
SERPER_API_KEY="your-actual-serper-key-here"

# Stability AI Key (https://platform.stability.ai)
STABILITY_API_KEY="sk-your-actual-stability-key-here"

# ElevenLabs API Key (https://elevenlabs.io)
ELEVENLABS_API_KEY="sk_your-actual-elevenlabs-key-here"

# Anthropic API Key (https://console.anthropic.com)
ANTHROPIC_API_KEY="sk-ant-api03-your-actual-key-here"

# Google API Key (https://console.cloud.google.com)
GOOGLE_API_KEY="AIzaSyYour-actual-google-key-here"

# IBM Quantum API Key (https://quantum-computing.ibm.com)
IBM_QUANTUM_API_KEY="your-actual-ibm-quantum-key-here"
```

## ✅ Verification Checklist

- [ ] `.env` file created with API keys
- [ ] `config/settings.yaml` exists (uses environment variables)
- [ ] `test_config.py` runs successfully
- [ ] Required API keys are set
- [ ] Optional API keys configured (if needed)
- [ ] Security settings reviewed
- [ ] Ready to run Pinnacle AI!

## 🐛 Troubleshooting

### Configuration Not Loading

1. Check that `.env` file exists in root directory
2. Verify environment variable names match exactly
3. Ensure `python-dotenv` is installed: `pip install python-dotenv`

### API Keys Not Working

1. Verify keys are correct (no extra spaces, quotes)
2. Check API key format matches expected format
3. Test API keys directly with their respective services
4. Check API key permissions and quotas

### Environment Variables Not Resolved

1. Check that variables use `${VARIABLE_NAME}` syntax in `config/settings.yaml`
2. Verify variable names match between `.env` and `config/settings.yaml`
3. Run `python test_config.py` to see which variables are missing

## 📚 Next Steps

1. **Read the documentation**: See `README.md` and `docs/` folder
2. **Try interactive mode**: `python src/main.py --interactive`
3. **Explore the web UI**: `python src/main.py --web`
4. **Test the API**: `python src/main.py --api`
5. **Run benchmarks**: `python src/main.py --benchmark`

## 🎉 You're Ready!

Once configuration is complete, you can start using Pinnacle AI!

For more information, see:
- `README.md` - Main documentation
- `SECURITY_NOTICE.md` - Security guidelines
- `PRODUCTION_READY_IMPLEMENTATION.md` - Implementation details

