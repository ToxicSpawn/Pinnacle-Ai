# Deployment Guide - Pinnacle AI

This guide covers various deployment options for Pinnacle AI.

## Local Deployment

### Basic Setup

```bash
# Clone repository
git clone https://github.com/ToxicSpawn/Pinnacle-AI.git
cd Pinnacle-AI

# Setup environment
python scripts/setup_environment.ps1  # Windows
# or
./scripts/setup_environment.sh        # Linux/Mac

# Run
python main.py --interactive
```

### Production Setup

```bash
# Install production dependencies only
pip install -r requirements.txt

# Configure
cp config/settings.yaml.example config/settings.yaml
# Edit config/settings.yaml with your settings

# Run as service (Linux)
sudo systemctl enable pinnacle-ai
sudo systemctl start pinnacle-ai
```

## Docker Deployment

### Build and Run

```bash
# Build image
docker build -f Dockerfile.pinnacle -t pinnacle-ai .

# Run container
docker run -it -p 8000:8000 \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/data:/app/data \
  pinnacle-ai
```

### Docker Compose

```bash
# Start services
docker-compose -f docker-compose.pinnacle.yml up -d

# View logs
docker-compose -f docker-compose.pinnacle.yml logs -f

# Stop services
docker-compose -f docker-compose.pinnacle.yml down
```

## Cloud Deployment

### AWS (EC2/ECS)

1. **EC2 Deployment**
   ```bash
   # SSH into EC2 instance
   ssh -i key.pem ec2-user@your-instance
   
   # Install dependencies
   sudo yum install python3 git -y
   git clone https://github.com/ToxicSpawn/Pinnacle-AI.git
   cd Pinnacle-AI
   pip3 install -r requirements.txt
   
   # Run with systemd
   sudo cp deploy/pinnacle-ai.service /etc/systemd/system/
   sudo systemctl enable pinnacle-ai
   sudo systemctl start pinnacle-ai
   ```

2. **ECS Deployment**
   - Build Docker image
   - Push to ECR
   - Create ECS task definition
   - Deploy service

### Google Cloud Platform

1. **Cloud Run**
   ```bash
   # Build and deploy
   gcloud builds submit --tag gcr.io/PROJECT_ID/pinnacle-ai
   gcloud run deploy pinnacle-ai --image gcr.io/PROJECT_ID/pinnacle-ai
   ```

2. **Compute Engine**
   - Similar to AWS EC2 deployment

### Azure

1. **Container Instances**
   ```bash
   az container create \
     --resource-group myResourceGroup \
     --name pinnacle-ai \
     --image pinnacle-ai:latest \
     --ports 8000
   ```

2. **App Service**
   - Deploy via Azure CLI or Portal

## Kubernetes Deployment

### Basic Deployment

```yaml
# k8s/deployment.yaml (already exists)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pinnacle-ai
spec:
  replicas: 3
  selector:
    matchLabels:
      app: pinnacle-ai
  template:
    metadata:
      labels:
        app: pinnacle-ai
    spec:
      containers:
      - name: pinnacle-ai
        image: pinnacle-ai:latest
        ports:
        - containerPort: 8000
```

```bash
# Deploy
kubectl apply -f k8s/deployment.yaml

# Expose service
kubectl expose deployment pinnacle-ai --type=LoadBalancer
```

## Environment Variables

Set these environment variables for configuration:

```bash
export PINNACLE_AI_CONFIG_PATH=/path/to/config.yaml
export PINNACLE_AI_LOG_LEVEL=INFO
export PINNACLE_AI_API_KEY=your-api-key
```

## Monitoring

### Health Checks

```bash
# Check health
curl http://localhost:8000/health

# Check metrics
curl http://localhost:8000/metrics
```

### Logging

Logs are written to:
- `logs/pinnacle_ai.log` (file)
- stdout (console)

Configure in `config/settings.yaml`:
```yaml
logging:
  level: INFO
  file: logs/pinnacle_ai.log
```

## Scaling

### Horizontal Scaling

- Use load balancer
- Deploy multiple instances
- Use container orchestration (K8s, ECS)

### Vertical Scaling

- Increase instance size
- Add more memory/CPU
- Optimize configuration

## Security

1. **API Keys**: Store in environment variables or secrets manager
2. **Network**: Use HTTPS in production
3. **Authentication**: Add API authentication if exposing publicly
4. **Secrets**: Use secret management (AWS Secrets Manager, etc.)

## Backup

Backup important data:
- Configuration files
- Model files (if any)
- Logs
- Data directory

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Check Python version (3.9+)
   - Reinstall dependencies

2. **Configuration Errors**
   - Verify YAML syntax
   - Check file paths

3. **Port Conflicts**
   - Change port in configuration
   - Check if port is in use

## Support

For deployment issues:
- Check logs: `logs/pinnacle_ai.log`
- Open an issue on GitHub
- Review documentation

