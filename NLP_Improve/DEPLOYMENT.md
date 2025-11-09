# 🚀 Deployment Guide

## Deploy to Vercel

### Quick Deploy

1. **Install Vercel CLI** (if not already installed):
```bash
npm install -g vercel
```

2. **Deploy from the project root**:
```bash
vercel
```

3. **Follow the prompts**:
   - Set up and deploy? **Yes**
   - Which scope? Select your account
   - Link to existing project? **No**
   - What's your project's name? `nlp-sentiment-analyzer` (or your choice)
   - In which directory is your code located? `./`
   - Want to override settings? **No**

### Production Deployment

```bash
vercel --prod
```

### Environment Variables

For production, you may need to set the backend API URL:

1. Go to your Vercel dashboard
2. Navigate to Settings → Environment Variables
3. Add: `VITE_API_URL` with your backend URL

### Update API Configuration

If deploying frontend and backend separately, update `frontend/src/services/api.js`:

```javascript
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'
```

## Local Development

### Frontend
```bash
cd frontend
npm install
npm run dev
```

### Backend
```bash
# Activate virtual environment
source .venv/bin/activate

# Run FastAPI server
uvicorn backend.main:app --reload --port 8000
```

## Build for Production

```bash
cd frontend
npm run build
npm run preview
```

## Vercel Deployment Options

### Option 1: GitHub Integration (Recommended)
1. Push your code to GitHub
2. Go to [vercel.com](https://vercel.com)
3. Click "Import Project"
4. Select your repository
5. Vercel will auto-detect settings and deploy

### Option 2: CLI Deployment
```bash
# One-time setup
vercel login

# Deploy
vercel

# Deploy to production
vercel --prod
```

## Post-Deployment

After deployment, your app will be available at:
- **Development**: `https://your-project.vercel.app`
- **Production**: `https://your-project.vercel.app` (with custom domain option)

### Custom Domain
1. Go to Vercel Dashboard
2. Select your project
3. Navigate to Settings → Domains
4. Add your custom domain

## Troubleshooting

### Build Fails
- Check `vercel.json` configuration
- Ensure `package.json` has correct scripts
- Verify all dependencies are in `package.json`

### API Not Working
- Update `VITE_API_URL` environment variable
- Check CORS settings in backend
- Ensure backend is deployed and accessible

### Styling Issues
- Verify Tailwind CSS is properly configured
- Check `tailwind.config.js` and `postcss.config.js`
- Ensure `index.css` imports Tailwind directives
