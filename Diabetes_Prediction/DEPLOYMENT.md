# 🚀 Deployment Guide

This guide provides step-by-step instructions for deploying your Diabetes ML Prediction System.

## 📋 Pre-Deployment Checklist

- [ ] All dependencies listed in `requirements.txt`
- [ ] App runs locally without errors (`streamlit run app.py`)
- [ ] All pages are accessible and functional
- [ ] No hardcoded secrets or API keys
- [ ] `.gitignore` configured properly
- [ ] README.md updated with accurate information

## 🌐 Option 1: Streamlit Cloud (Recommended - Free & Easy)

### Prerequisites
- GitHub account
- Your code pushed to a GitHub repository

### Steps

1. **Prepare Your Repository**
   ```bash
   # Initialize git if not already done
   git init
   
   # Add all files
   git add .
   
   # Commit
   git commit -m "Initial commit: Diabetes ML Prediction System"
   
   # Push to GitHub
   git remote add origin <your-repo-url>
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select:
     - Repository: Your GitHub repo
     - Branch: `main`
     - Main file path: `app.py`
   - Click "Deploy!"

3. **Wait for Deployment**
   - Streamlit Cloud will install dependencies and start your app
   - You'll get a public URL like `https://your-app-name.streamlit.app`
   - Deployment usually takes 2-5 minutes

### Updating Your App
```bash
# Make changes locally
# Commit and push
git add .
git commit -m "Update features"
git push

# Streamlit Cloud auto-redeploys on push!
```

## 🐳 Option 2: Docker Deployment

### Create Dockerfile

Create a file named `Dockerfile` in your project root:

```dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Build and Run

```bash
# Build image
docker build -t diabetes-ml-app .

# Run container
docker run -p 8501:8501 diabetes-ml-app

# Access at http://localhost:8501
```

### Deploy to Docker Hub

```bash
# Tag image
docker tag diabetes-ml-app your-dockerhub-username/diabetes-ml-app:latest

# Push to Docker Hub
docker push your-dockerhub-username/diabetes-ml-app:latest

# Others can now run:
docker pull your-dockerhub-username/diabetes-ml-app:latest
docker run -p 8501:8501 your-dockerhub-username/diabetes-ml-app:latest
```

## ☁️ Option 3: Heroku Deployment

### Prerequisites
- Heroku account
- Heroku CLI installed

### Setup Files

1. **Create `setup.sh`**:
   ```bash
   mkdir -p ~/.streamlit/
   
   echo "\
   [general]\n\
   email = \"your-email@example.com\"\n\
   " > ~/.streamlit/credentials.toml
   
   echo "\
   [server]\n\
   headless = true\n\
   enableCORS=false\n\
   port = $PORT\n\
   " > ~/.streamlit/config.toml
   ```

2. **Create `Procfile`**:
   ```
   web: sh setup.sh && streamlit run app.py
   ```

### Deploy

```bash
# Login to Heroku
heroku login

# Create app
heroku create your-app-name

# Push code
git push heroku main

# Open app
heroku open
```

## 🔧 Option 4: AWS EC2 Deployment

### Launch EC2 Instance

1. Launch Ubuntu Server 22.04 LTS
2. Configure security group:
   - Allow SSH (port 22)
   - Allow Custom TCP (port 8501)

### Setup on EC2

```bash
# SSH into instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and pip
sudo apt install python3-pip python3-venv -y

# Clone your repository
git clone https://github.com/your-username/your-repo.git
cd your-repo

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run with nohup (keeps running after logout)
nohup streamlit run app.py --server.port=8501 --server.address=0.0.0.0 &

# Access at http://your-ec2-ip:8501
```

### Keep Running with systemd

Create `/etc/systemd/system/streamlit-app.service`:

```ini
[Unit]
Description=Streamlit Diabetes ML App
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/your-repo
ExecStart=/home/ubuntu/your-repo/venv/bin/streamlit run app.py --server.port=8501 --server.address=0.0.0.0
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable streamlit-app
sudo systemctl start streamlit-app
sudo systemctl status streamlit-app
```

## 🌍 Option 5: Google Cloud Platform (Cloud Run)

### Prerequisites
- Google Cloud account
- gcloud CLI installed

### Deploy

```bash
# Authenticate
gcloud auth login

# Set project
gcloud config set project YOUR_PROJECT_ID

# Deploy
gcloud run deploy diabetes-ml-app \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated

# You'll get a URL like: https://diabetes-ml-app-xxxxx-uc.a.run.app
```

## 📊 Post-Deployment

### Monitor Your App

1. **Check Logs**:
   - Streamlit Cloud: View logs in dashboard
   - Heroku: `heroku logs --tail`
   - Docker: `docker logs container-id`

2. **Performance Monitoring**:
   - Monitor response times
   - Check memory usage
   - Track error rates

3. **User Analytics**:
   - Add Google Analytics (optional)
   - Track page views
   - Monitor user engagement

### Update Strategy

```bash
# Development workflow
1. Make changes locally
2. Test thoroughly: streamlit run app.py
3. Commit: git add . && git commit -m "description"
4. Push: git push origin main
5. Deployment happens automatically (Streamlit Cloud)
```

## 🔐 Security Best Practices

1. **Environment Variables**: Store secrets in environment variables, not code
   ```python
   import os
   api_key = os.environ.get('API_KEY')
   ```

2. **HTTPS**: Always use HTTPS in production
   - Streamlit Cloud: Automatic HTTPS
   - Others: Use Let's Encrypt or cloud provider SSL

3. **Authentication**: Add authentication for sensitive apps
   ```python
   import streamlit_authenticator as stauth
   ```

4. **Rate Limiting**: Prevent abuse
   - Use cloud provider rate limiting
   - Implement app-level throttling

## 🐛 Troubleshooting

### Common Issues

**Issue**: App won't start
```bash
# Check logs
streamlit run app.py --logger.level=debug
```

**Issue**: Import errors
```bash
# Verify all dependencies
pip install -r requirements.txt
# Check Python version
python --version  # Should be 3.9+
```

**Issue**: Memory errors
- Reduce dataset size in app
- Use efficient data structures
- Clear cache: Add `@st.cache_data` decorators

**Issue**: Slow performance
- Enable caching
- Optimize data loading
- Reduce visualization complexity

## 📞 Support

- **Streamlit Docs**: https://docs.streamlit.io
- **Community Forum**: https://discuss.streamlit.io
- **GitHub Issues**: Report bugs in your repo

## ✅ Success Checklist

After deployment, verify:

- [ ] App loads without errors
- [ ] All pages are accessible
- [ ] Interactive features work (buttons, sliders)
- [ ] Visualizations render correctly
- [ ] Models train successfully
- [ ] Predictions are accurate
- [ ] Mobile responsiveness (test on phone)
- [ ] Acceptable loading times (<3 seconds)
- [ ] No console errors (check browser developer tools)
- [ ] SSL certificate valid (HTTPS)

## 🎉 You're Live!

Congratulations! Your Diabetes ML Prediction System is now deployed and accessible worldwide.

Share your app URL:
- With classmates and professors
- On LinkedIn/portfolio
- In your resume/CV

Keep iterating and improving based on user feedback!
