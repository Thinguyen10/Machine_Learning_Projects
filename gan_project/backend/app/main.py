"""
FastAPI Backend for GAN Image Generator
Provides REST API endpoints for training and image generation
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import numpy as np
import os
import json
from datetime import datetime
import base64
from io import BytesIO
from PIL import Image
from tensorflow.keras.optimizers import Adam

# Import GAN components
from .model import Generator, Discriminator, GAN
from .processing import DataProcessor
from .analysis import GANAnalyzer
from .visual import GANVisualizer

app = FastAPI(
    title="GAN Image Generator API",
    description="REST API for training and generating images with GANs",
    version="1.0.0"
)

# CORS configuration for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
training_state = {
    "is_training": False,
    "is_preparing": False,
    "preparation_status": "",
    "current_epoch": 0,
    "total_epochs": 0,
    "d_loss": 0.0,
    "d_accuracy": 0.0,
    "g_loss": 0.0,
    "latest_images": None,
    "training_history": []
}

# Models storage
models = {
    "generator": None,
    "discriminator": None,
    "gan": None,
    "data_processor": None,
    "analyzer": None,
    "visualizer": None
}

# Pydantic models
class TrainingConfig(BaseModel):
    epochs: int = 400
    batch_size: int = 128
    noise_dim: int = 100
    learning_rate: float = 0.0002
    dataset: str = "mnist"

class GenerateRequest(BaseModel):
    num_images: int = 16
    noise_seed: Optional[int] = None


# Helper functions
def initialize_models(config: TrainingConfig):
    """Initialize GAN models (only if not already initialized)"""
    # Only initialize if models don't exist yet
    if models["generator"] is not None:
        print("Models already initialized, skipping re-initialization")
        return
    
    img_shape = (28, 28, 1)
    
    models["data_processor"] = DataProcessor(dataset_name=config.dataset)
    models["generator"] = Generator(config.noise_dim, img_shape)
    models["discriminator"] = Discriminator(img_shape)
    models["analyzer"] = GANAnalyzer()
    models["visualizer"] = GANVisualizer(img_shape)
    
    # Build and compile
    models["generator"].build()
    models["generator"].compile()
    
    models["discriminator"].build()
    models["discriminator"].compile()
    
    models["gan"] = GAN(models["generator"], models["discriminator"])
    models["gan"].build()
    models["gan"].compile()
    
    print("Models initialized successfully")


def images_to_base64(images):
    """Convert numpy array images to base64 strings"""
    base64_images = []
    
    for img in images:
        # Rescale from [-1, 1] to [0, 255]
        img_rescaled = ((img + 1) * 127.5).astype(np.uint8)
        
        # Convert to PIL Image
        if img_rescaled.shape[-1] == 1:
            pil_img = Image.fromarray(img_rescaled[:, :, 0], mode='L')
        else:
            pil_img = Image.fromarray(img_rescaled, mode='RGB')
        
        # Convert to base64
        buffered = BytesIO()
        pil_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        base64_images.append(f"data:image/png;base64,{img_str}")
    
    return base64_images


async def train_gan_background(config: TrainingConfig):
    """Background task for training GAN"""
    global training_state
    
    try:
        import asyncio
        
        training_state["is_preparing"] = True
        training_state["preparation_status"] = "Loading dataset..."
        training_state["total_epochs"] = config.epochs
        training_state["current_epoch"] = 0
        await asyncio.sleep(0.5)  # Give UI time to poll
        
        # Load data
        models["data_processor"].load_data()
        training_state["preparation_status"] = f"✓ Loaded {config.dataset} dataset: {models['data_processor'].x_train.shape[0]} training samples"
        await asyncio.sleep(0.8)
        
        training_state["preparation_status"] = f"✓ Image shape: {models['data_processor'].img_shape}"
        await asyncio.sleep(0.8)
        
        training_state["preparation_status"] = "⚙️ Normalizing data to [-1, 1]..."
        await asyncio.sleep(0.5)
        models["data_processor"].preprocess_data(normalize=True, reshape=True)
        
        training_state["preparation_status"] = f"✓ Reshaped to: {models['data_processor'].img_shape}"
        await asyncio.sleep(0.8)
        
        training_state["preparation_status"] = "🚀 Starting training..."
        await asyncio.sleep(0.5)
        training_state["is_preparing"] = False
        training_state["is_training"] = True
        
        # Training parameters
        batch_size = config.batch_size
        noise_dim = config.noise_dim
        
        # PRE-TRAINING: Train discriminator first to give it a head start
        training_state["preparation_status"] = "🎯 Pre-training discriminator (100 epochs)..."
        training_state["is_preparing"] = True
        
        # CRITICAL: Ensure discriminator is trainable
        models["discriminator"].model.trainable = True
        
        # Reduced pre-training to 30 epochs to prevent discriminator from becoming too strong
        for pre_epoch in range(30):
            # Use standard labels for pre-training (0 and 1)
            # Train multiple times per pre-training epoch
            for iteration in range(3):
                # Train on real images
                real_images = models["data_processor"].get_random_samples(batch_size)
                real_labels = np.ones((batch_size, 1))
                d_loss_real = models["discriminator"].model.train_on_batch(real_images, real_labels)
                
                # Train on fake images
                noise = models["data_processor"].generate_noise(batch_size, noise_dim)
                fake_images = models["generator"].model.predict(noise, verbose=0)
                fake_labels = np.zeros((batch_size, 1))
                d_loss_fake = models["discriminator"].model.train_on_batch(fake_images, fake_labels)
                
                # Track progress every 10 epochs
                if pre_epoch % 10 == 0 and iteration == 2:
                    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                    print(f"Pre-training epoch {pre_epoch+1}/30 - D Loss: {d_loss[0]:.4f}, D Acc: {d_loss[1]:.1%}")
            
            # Update status every 10 epochs so frontend knows we're alive
            if pre_epoch % 10 == 0:
                training_state["preparation_status"] = f"🎯 Pre-training: {pre_epoch+1}/30 epochs..."
                await asyncio.sleep(0.1)
        
        # Check pre-training results
        test_real = models["data_processor"].get_random_samples(batch_size)
        test_noise = models["data_processor"].generate_noise(batch_size, noise_dim)
        test_fake = models["generator"].model.predict(test_noise, verbose=0)
        
        test_real_pred = models["discriminator"].model.predict(test_real, verbose=0)
        test_fake_pred = models["discriminator"].model.predict(test_fake, verbose=0)
        
        pre_train_acc = ((test_real_pred > 0.5).sum() + (test_fake_pred < 0.5).sum()) / (batch_size * 2)
        print(f"Pre-training accuracy check: {pre_train_acc:.1%}")
        print(f"Real predictions: mean={test_real_pred.mean():.3f}, Fake predictions: mean={test_fake_pred.mean():.3f}")
        
        training_state["preparation_status"] = f"✓ Pre-training complete! Accuracy: {pre_train_acc:.1%}"
        training_state["is_preparing"] = False
        await asyncio.sleep(1.0)
        
        # Training loop - Balanced 1:1 training for optimal GAN dynamics
        for epoch in range(config.epochs):
            # Labels
            real_labels = np.ones((batch_size, 1))
            fake_labels = np.zeros((batch_size, 1))
            
            # ===== Train Discriminator once =====
            # Train on real images
            real_images = models["data_processor"].get_random_samples(batch_size)
            d_loss_real = models["discriminator"].model.train_on_batch(real_images, real_labels)
            
            # Train on fake images
            noise = models["data_processor"].generate_noise(batch_size, noise_dim)
            fake_images = models["generator"].model.predict(noise, verbose=0)
            d_loss_fake = models["discriminator"].model.train_on_batch(fake_images, fake_labels)
            
            # Average discriminator metrics
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # ===== Train Generator once =====
            # Train through GAN model (discriminator frozen in GAN by Functional API)
            noise = models["data_processor"].generate_noise(batch_size, noise_dim)
            g_loss = models["gan"].model.train_on_batch(noise, real_labels)
            
            # No trainable toggling needed - discriminator is ALWAYS trainable when called directly,
            # and ALWAYS frozen when accessed through the GAN model (set during GAN.build())
            
            # Diagnostic logging every 100 epochs
            if epoch % 100 == 0:
                # Check what discriminator actually predicts
                test_real = models["data_processor"].get_random_samples(32)
                test_noise = models["data_processor"].generate_noise(32, noise_dim)
                test_fake = models["generator"].model.predict(test_noise, verbose=0)
                
                pred_real = models["discriminator"].model.predict(test_real, verbose=0)
                pred_fake = models["discriminator"].model.predict(test_fake, verbose=0)
                
                print(f"\n{'='*60}")
                print(f"Epoch {epoch+1} Diagnostic:")
                print(f"D Loss: {d_loss[0]:.4f}, D Acc: {d_loss[1]:.1%}, G Loss: {g_loss:.4f}")
                print(f"Real predictions - mean: {pred_real.mean():.3f}, std: {pred_real.std():.3f}")
                print(f"Fake predictions - mean: {pred_fake.mean():.3f}, std: {pred_fake.std():.3f}")
                print(f"Real > 0.5: {(pred_real > 0.5).sum()}/32, Fake < 0.5: {(pred_fake < 0.5).sum()}/32")
                print(f"{'='*60}\n")
            
            # Update state
            training_state["current_epoch"] = epoch + 1
            training_state["d_loss"] = float(d_loss[0])
            training_state["d_accuracy"] = float(d_loss[1])
            training_state["g_loss"] = float(g_loss)
            
            # Update analyzer
            models["analyzer"].update_metrics(d_loss[0], d_loss[1], g_loss, epoch)
            
            # Save history periodically
            if epoch % 10 == 0:
                training_state["training_history"].append({
                    "epoch": epoch,
                    "d_loss": float(d_loss[0]),
                    "d_accuracy": float(d_loss[1]),
                    "g_loss": float(g_loss)
                })
            
            # Generate sample images at milestones: epoch 1, 30, 100, 400
            if epoch % 50 == 0 or epoch in [1, 30, 100, 400]:
                sample_noise = models["data_processor"].generate_noise(16, noise_dim)
                sample_images = models["generator"].model.predict(sample_noise, verbose=0)
                training_state["latest_images"] = images_to_base64(sample_images)
        
        training_state["is_training"] = False
        training_state["is_preparing"] = False
        
    except Exception as e:
        training_state["is_training"] = False
        training_state["is_preparing"] = False
        training_state["error"] = str(e)
        print(f"Training error: {e}")


# API Endpoints

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "GAN Image Generator API",
        "version": "1.0.0",
        "endpoints": {
            "status": "/status",
            "train": "/train",
            "generate": "/generate",
            "history": "/history",
            "models": "/models"
        }
    }


@app.get("/status")
async def get_status():
    """Get current training status"""
    return {
        "is_training": training_state["is_training"],
        "is_preparing": training_state["is_preparing"],
        "preparation_status": training_state["preparation_status"],
        "current_epoch": training_state["current_epoch"],
        "total_epochs": training_state["total_epochs"],
        "metrics": {
            "discriminator_loss": training_state["d_loss"],
            "discriminator_accuracy": training_state["d_accuracy"],
            "generator_loss": training_state["g_loss"]
        },
        "progress": (training_state["current_epoch"] / training_state["total_epochs"] * 100) 
                    if training_state["total_epochs"] > 0 else 0
    }


@app.post("/train")
async def start_training(config: TrainingConfig, background_tasks: BackgroundTasks):
    """Start GAN training"""
    if training_state["is_training"]:
        raise HTTPException(status_code=400, detail="Training already in progress")
    
    # Initialize models
    initialize_models(config)
    
    # Start training in background
    background_tasks.add_task(train_gan_background, config)
    
    return {
        "message": "Training started",
        "config": config.dict()
    }


@app.post("/train/stop")
async def stop_training():
    """Stop current training"""
    if not training_state["is_training"]:
        raise HTTPException(status_code=400, detail="No training in progress")
    
    training_state["is_training"] = False
    return {"message": "Training stopped"}


@app.post("/train/reset")
async def reset_models():
    """Reset models to start fresh training"""
    if training_state["is_training"]:
        raise HTTPException(status_code=400, detail="Cannot reset while training is in progress")
    
    # Clear all models
    models["generator"] = None
    models["discriminator"] = None
    models["gan"] = None
    models["data_processor"] = None
    models["analyzer"] = None
    models["visualizer"] = None
    
    # Reset training state
    training_state["current_epoch"] = 0
    training_state["total_epochs"] = 0
    training_state["d_loss"] = 0.0
    training_state["d_accuracy"] = 0.0
    training_state["g_loss"] = 0.0
    training_state["latest_images"] = None
    training_state["training_history"] = []
    
    return {"message": "Models reset successfully. Ready for fresh training."}


@app.post("/generate")
async def generate_images(request: GenerateRequest):
    """Generate images using trained generator"""
    if models["generator"] is None:
        raise HTTPException(status_code=400, detail="Model not trained yet")
    
    # Set random seed if provided
    if request.noise_seed is not None:
        np.random.seed(request.noise_seed)
    
    # Generate images
    noise_dim = 100  # Default
    noise = np.random.normal(0, 1, (request.num_images, noise_dim))
    generated_images = models["generator"].model.predict(noise, verbose=0)
    
    # Convert to base64
    base64_images = images_to_base64(generated_images)
    
    return {
        "num_images": request.num_images,
        "images": base64_images
    }


@app.get("/latest-images")
async def get_latest_images():
    """Get latest generated images from training"""
    if training_state["latest_images"] is None:
        raise HTTPException(status_code=404, detail="No images available yet")
    
    return {
        "images": training_state["latest_images"],
        "epoch": training_state["current_epoch"]
    }


@app.get("/history")
async def get_training_history():
    """Get training history"""
    return {
        "history": training_state["training_history"],
        "total_epochs": training_state["current_epoch"]
    }


@app.get("/summary")
async def get_summary():
    """Get training summary report"""
    if models["analyzer"] is None:
        raise HTTPException(status_code=400, detail="No training data available")
    
    stats = models["analyzer"].calculate_statistics()
    
    if stats is None:
        return {"message": "No statistics available yet"}
    
    return {
        "discriminator": {
            "loss": {
                "mean": stats["d_loss_mean"],
                "std": stats["d_loss_std"],
                "min": stats["d_loss_min"],
                "max": stats["d_loss_max"]
            },
            "accuracy": {
                "mean": stats["d_acc_mean"],
                "std": stats["d_acc_std"],
                "min": stats["d_acc_min"],
                "max": stats["d_acc_max"]
            }
        },
        "generator": {
            "loss": {
                "mean": stats["g_loss_mean"],
                "std": stats["g_loss_std"],
                "min": stats["g_loss_min"],
                "max": stats["g_loss_max"]
            }
        },
        "total_epochs": len(models["analyzer"].history["epochs"])
    }


@app.get("/models/info")
async def get_models_info():
    """Get information about loaded models"""
    return {
        "generator_loaded": models["generator"] is not None,
        "discriminator_loaded": models["discriminator"] is not None,
        "gan_loaded": models["gan"] is not None,
        "data_loaded": models["data_processor"] is not None
    }


@app.post("/models/save")
async def save_models():
    """Save trained models"""
    if models["generator"] is None:
        raise HTTPException(status_code=400, detail="No models to save")
    
    save_dir = "backend/saved_models"
    os.makedirs(save_dir, exist_ok=True)
    
    models["generator"].model.save(f"{save_dir}/generator.h5")
    models["discriminator"].model.save(f"{save_dir}/discriminator.h5")
    models["gan"].model.save(f"{save_dir}/gan.h5")
    
    return {
        "message": "Models saved successfully",
        "location": save_dir
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
