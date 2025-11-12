#!/bin/bash

# Start NLP Sentiment Analysis Application
# This script starts both the backend and frontend servers

echo "🚀 Starting NLP Sentiment Analysis Application..."
echo ""

# Check if models exist
if [ ! -f "model_sklearn.joblib" ] || [ ! -f "model_keras.keras" ] || [ ! -f "vect.joblib" ]; then
    echo "⚠️  Models not found! Please train them first:"
    echo "   ./train_models.sh"
    exit 1
fi

echo "✓ Models found!"
echo ""

# Start backend
echo "📡 Starting backend server on http://localhost:8000..."
uvicorn backend.main:app --reload --port 8000 > /tmp/nlp_backend.log 2>&1 &
BACKEND_PID=$!
echo "   Backend PID: $BACKEND_PID"

# Wait for backend to start
sleep 3

# Check if backend started successfully
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "❌ Backend failed to start. Check /tmp/nlp_backend.log"
    exit 1
fi

echo "✓ Backend running!"
echo ""

# Start frontend
echo "🎨 Starting frontend dev server..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo "============================================"
echo "✅ Application Started Successfully!"
echo "============================================"
echo ""
echo "📍 Frontend: http://localhost:3000"
echo "📍 Backend:  http://localhost:8000"
echo ""
echo "📝 Logs:"
echo "   Backend: tail -f /tmp/nlp_backend.log"
echo ""
echo "🛑 To stop:"
echo "   kill $BACKEND_PID $FRONTEND_PID"
echo "   or press Ctrl+C in the terminal running frontend"
echo ""
echo "🎉 Happy analyzing!"
