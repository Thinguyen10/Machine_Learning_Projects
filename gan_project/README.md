# GAN Image Generator - Full Stack Web App

## 🚀 Quick Start

```bash
# Backend (Terminal 1)
cd backend
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --port 8000

# Frontend (Terminal 2)
cd frontend
npm install
npm run dev
```

Open **http://localhost:3000**

## 📁 Project Structure

```
gan_project/
├── backend/              # FastAPI + TensorFlow
│   ├── app/
│   │   ├── main.py      # REST API
│   │   ├── model.py     # GAN models
│   │   └── processing.py
│   └── requirements.txt
├── frontend/             # Next.js + React
│   ├── app/
│   ├── components/
│   └── package.json
└── vercel.json          # Deployment config
```

## 🎯 Features

- ✅ Train GANs via web interface
- ✅ Real-time training progress
- ✅ Generate images on-demand
- ✅ Monitor metrics (loss, accuracy)
- ✅ REST API with 11 endpoints
- ✅ Deploy to Vercel/Railway

## 🌐 API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /status` | Training status & metrics |
| `POST /train` | Start training |
| `POST /generate` | Generate images |
| `GET /health` | Backend health check |

Docs: `http://localhost:8000/docs`


##  What is a GAN?

**Generator** creates fake images → **Discriminator** spots fakes → They compete → Better images!

- Dataset: MNIST (60,000 handwritten digits)
- Training: ~400 epochs for good results
- Output: Realistic digit images from random noise

### Training Tips for Better Results

**If discriminator accuracy is too high (>85%):**
- Generator is too weak - images will be blurry or black
- Solution: Increase generator training frequency in code

**If discriminator accuracy is too low (<30%):**
- Discriminator is too weak - can't provide useful feedback
- Solution: Strengthen discriminator architecture (add layers/dropout)

**Optimal Range:**
- **Discriminator Accuracy: 50-80%** ✓ Balanced training
- **D Loss: 0.4-0.7** ✓ Good competition
- **G Loss: 0.7-1.5** ✓ Generator improving

**Key Improvements Applied:**

- ✅ **Removed BatchNorm from discriminator** - Was causing mean collapse (outputs stuck at 0.5)
- ✅ **Functional API for GAN** - Ensures discriminator is properly frozen during generator training
- ✅ **Balanced 1:1 training ratio** - Equal training for both networks to prevent discriminator dominance
- ✅ **Reduced pre-training**: 30 epochs × 3 iterations = 90 discriminator updates (prevents over-training)
- ✅ **Hard labels only**: Crystal clear signal (real=1.0, fake=0.0) for stable training
- ✅ **Models persist across sessions**: Training accumulates instead of resetting
- ✅ **Dropout 0.3 in discriminator**: Regularization without BatchNorm
- ✅ **BatchNorm in generator only**: Helps generator stability without affecting discriminator
- ✅ **Fresh samples every iteration**: New noise and real images each epoch
- ✅ Standard learning rate (0.0002) with beta_1=0.5 for both networks


## ⚙️ Configuration

**Current Training Configuration:**
```python
Pre-training: 30 epochs × 3 iterations = 90 discriminator updates  
Main training ratio: 1:1 BALANCED (both train once per epoch)
Labels: Hard labels (real=1.0, fake=0.0)
Discriminator: NO BatchNorm (prevents mean collapse), Dropout=0.3
Generator: WITH BatchNorm (helps stability)
Learning rate: 0.0002 (both networks)
Beta_1: 0.5 (Adam optimizer)
Batch size: 32
Noise dimension: 100
```

**Training Parameters:**
- Epochs: 100-1000 (400-600 recommended)
- Batch Size: 128
- Dataset: MNIST or Fashion-MNIST

**Environment:**
```bash
# frontend/.env.local
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Backend won't start | `pip install --upgrade -r requirements.txt` |
| Can't connect | Check backend runs on :8000 |
| CORS error | Update `allow_origins` in `main.py` |
| Training timeout | Deploy backend to Railway (Vercel has limits) |

## 📊 Tech Stack

**Backend:** 
- **FastAPI** - Modern, high-performance Python web framework with automatic API documentation
- **TensorFlow 2.x** - Deep learning framework for building and training neural networks
- **Keras** - High-level neural networks API integrated with TensorFlow
- **Python 3.11+** - Core programming language with async support

**Frontend:** 
- **Next.js 15** - React framework with server-side rendering and App Router for optimal performance
- **React 19** - UI library with hooks for state management and component composition
- **TypeScript** - Type-safe JavaScript for better code quality and developer experience
- **Tailwind CSS** - Utility-first CSS framework for rapid, responsive UI development

**Deployment:** 
- **Vercel** - Zero-config frontend hosting with automatic HTTPS and global CDN
- **Railway/Render** - Backend hosting platforms with container support and easy scaling
- **Docker** - Optional containerization for consistent deployment environments

## 🎨 How to Use

1. Click "Train GAN" → Set epochs → Start
2. Watch real-time progress bar & metrics
3. Click "Generate Images" → Choose quantity
4. View/download generated digits

## 🔐 Production Setup

- Update CORS to your domain
- Set environment variables
- Enable HTTPS (auto on Vercel/Railway)
- Optional: Add rate limiting

## 📚 Resources

- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Next.js Docs](https://nextjs.org/docs)
- [GAN Tutorial](https://www.tensorflow.org/tutorials/generative/dcgan)

## 🚀 Deployment

### Option 1: Web Interface (Recommended)

The full-stack web application with real-time monitoring and interactive UI.

**Frontend (Vercel)**
```bash
# 1. Push your code to GitHub
git init
git add .
git commit -m "Initial commit"
git push origin main

# 2. Deploy to Vercel
# - Go to vercel.com and import your repository
# - Set Root Directory: "frontend"
# - Add environment variable:
#   NEXT_PUBLIC_API_URL=<your-backend-url>
# - Deploy!
```

**Backend (Railway)**
```bash
# 1. Connect your GitHub repository to Railway
# 2. Configure deployment:
#    - Root Directory: "backend"
#    - Start Command: uvicorn app.main:app --host 0.0.0.0 --port $PORT
# 3. Railway will auto-deploy on git push

# Alternative: Deploy to Render
# - Similar process, use render.yaml for configuration
# - Render offers free tier with some limitations
```

**Post-Deployment:**
- Update CORS origins in `backend/app/main.py` to include your frontend URL
- Test connection using the status button in the UI
- Start training and verify real-time updates work correctly

### Option 2: Standalone CLI (Original)

For local development or running training scripts without the web interface.

```bash
# 1. Navigate to backend directory
cd backend

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run standalone training script
python standalone_train.py

# 4. Or use interactive Python
python
>>> from app.model import Generator, Discriminator, GAN
>>> from app.processing import DataProcessor
>>> # Train your GAN with custom code
```

**Use Case:** Best for:
- Quick local experiments
- Automated training pipelines
- Integration with existing ML workflows
- Research and development without UI overhead
---

**Ready to deploy? See vercel.json for config** 🚀
