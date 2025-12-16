# 🔒 Security Notice

## ⚠️ IMPORTANT: API Keys Exposed

**If you shared API keys in this conversation, they are now publicly visible and should be considered compromised.**

## 🚨 Immediate Actions Required

### 1. Rotate All API Keys Immediately

Generate new API keys for all services:

- **OpenAI**: https://platform.openai.com/api-keys
- **Google Cloud**: https://console.cloud.google.com/apis/credentials
- **Anthropic**: https://console.anthropic.com/settings/keys
- **Serper**: https://serper.dev/api-keys
- **Stability AI**: https://platform.stability.ai/account/keys
- **ElevenLabs**: https://elevenlabs.io/app/settings/api-keys
- **IBM Quantum**: https://quantum-computing.ibm.com/

### 2. Never Commit API Keys

- ✅ `.env` is in `.gitignore` - **DO NOT REMOVE IT**
- ✅ `config/settings.yaml` is in `.gitignore` - **DO NOT REMOVE IT**
- ❌ Never commit files with real API keys
- ❌ Never share API keys in conversations, issues, or pull requests

### 3. Use Environment Variables

Always use environment variables for sensitive data:

```bash
# .env file (NOT committed to git)
OPENAI_API_KEY="your-actual-key-here"
```

```yaml
# config/settings.yaml (can be committed with placeholders)
core:
  api_keys:
    openai: "${OPENAI_API_KEY}"  # Resolved from .env
```

### 4. Set Up Rate Limits

Enable rate limiting in your configuration:

```yaml
security:
  rate_limiting:
    enabled: true
    requests_per_minute: 60
    burst_limit: 30
```

### 5. Enable Authentication

For production deployments, enable authentication:

```yaml
security:
  authentication:
    enabled: true
    jwt_secret: "generate-a-strong-secret-key-here"
```

### 6. Monitor API Usage

- Set up alerts for unusual activity
- Monitor your API dashboards regularly
- Review usage logs frequently

### 7. Use Different Keys for Different Environments

Create separate `.env` files:
- `.env.development` - For local development
- `.env.staging` - For staging environment
- `.env.production` - For production (use secrets management)

## 📋 Best Practices

### Environment Variables

1. **Never hardcode API keys in source code**
2. **Use `.env` files for local development**
3. **Use secrets management for production** (AWS Secrets Manager, Azure Key Vault, etc.)
4. **Rotate keys regularly** (every 90 days recommended)
5. **Use different keys for different environments**

### Configuration Files

1. **Use `.env.example` as a template** (without real keys)
2. **Document required environment variables** in README
3. **Validate configuration on startup**
4. **Fail fast if required keys are missing**

### Git Security

1. **Review `.gitignore` regularly**
2. **Use `git-secrets` or similar tools** to prevent accidental commits
3. **Scan repository history** for exposed keys
4. **Use pre-commit hooks** to check for secrets

### API Security

1. **Set up rate limits** to prevent abuse
2. **Monitor API usage** for unusual patterns
3. **Use IP whitelisting** when possible
4. **Enable authentication** for all production deployments
5. **Use HTTPS** for all API communications

## 🔍 Checking for Exposed Keys

### Scan Your Repository

```bash
# Use git-secrets or similar tools
git-secrets --scan

# Or use truffleHog
trufflehog git file://. --json
```

### Check Git History

```bash
# Search for API keys in git history
git log -p -S "sk-" --all
git log -p -S "api_key" --all
```

### If Keys Are Found

1. **Rotate the keys immediately**
2. **Remove keys from git history** (if possible)
3. **Notify your team**
4. **Review access logs** for unauthorized usage

## 📝 Example Secure Setup

### 1. Create `.env` file (NOT in git)

```bash
# .env
OPENAI_API_KEY="sk-proj-actual-key-here"
SERPER_API_KEY="actual-serper-key-here"
```

### 2. Create `.env.example` (in git, no real keys)

```bash
# .env.example
OPENAI_API_KEY="sk-proj-your-key-here"
SERPER_API_KEY="your-serper-key-here"
```

### 3. Use in configuration

```yaml
# config/settings.yaml
core:
  api_keys:
    openai: "${OPENAI_API_KEY}"
```

### 4. Load in code

```python
from src.tools.config_loader import load_config

config = load_config()  # Automatically resolves ${VARIABLE_NAME}
api_key = config["core"]["api_keys"]["openai"]
```

## 🛡️ Additional Security Measures

### For Production

1. **Use secrets management services**
   - AWS Secrets Manager
   - Azure Key Vault
   - Google Secret Manager
   - HashiCorp Vault

2. **Enable authentication**
   - JWT tokens
   - API keys with roles
   - OAuth2

3. **Set up monitoring**
   - API usage tracking
   - Error logging
   - Security alerts

4. **Use HTTPS**
   - SSL/TLS certificates
   - Certificate pinning

5. **Implement rate limiting**
   - Per-user limits
   - Per-IP limits
   - Burst protection

## 📞 Support

If you discover exposed keys:
1. Rotate them immediately
2. Review access logs
3. Contact support if unauthorized usage is detected

## ✅ Security Checklist

- [ ] All API keys rotated
- [ ] `.env` file created and in `.gitignore`
- [ ] `.env.example` created (without real keys)
- [ ] `config/settings.yaml` uses environment variables
- [ ] Rate limiting enabled
- [ ] Authentication enabled (for production)
- [ ] Monitoring set up
- [ ] Repository scanned for exposed keys
- [ ] Team notified of security practices

---

**Remember: Security is an ongoing process, not a one-time setup!**

