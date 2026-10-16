# services/pipeline_service.py
# Pipeline service for ML model training
import json
import logging
import random
import csv
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sentence_transformers import SentenceTransformer

from models.schemas import IntentsConfig, PipelineStatus
from core.state import pipeline_state, active_domain, models
from ml_utils.helpers import update_pipeline_state, load_domain, MODELS_DIR
from config.settings import EMBED_MODEL

logger = logging.getLogger(__name__)

async def run_pipeline(config: IntentsConfig):
    """Run the complete ML pipeline"""
    try:
        domain = config.domain
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create domain-specific directories
        domain_dir = MODELS_DIR / domain
        domain_dir.mkdir(parents=True, exist_ok=True)
        
        dataset_filename = domain_dir / f"synthetic_intents_{domain}.csv"
        
        update_pipeline_state(
            status=PipelineStatus.GENERATING_DATA,
            domain=domain,
            start_time=datetime.utcnow().isoformat() + "Z",
            current_step="Generating synthetic dataset",
            dataset_path=str(dataset_filename)
        )
        
        # Step 1: Generate synthetic data
        logger.info(f"Starting data generation for domain: {domain}")
        await generate_synthetic_data(config, dataset_filename)
        
        update_pipeline_state(
            status=PipelineStatus.TRAINING_MODEL,
            current_step="Training classification model"
        )
        
        # Step 2: Train the model
        logger.info(f"Starting model training for domain: {domain}")
        metrics = await train_model(dataset_filename, domain_dir)
        
        # Step 3: Auto-reload the model in memory if it's the active domain
        if active_domain == domain:
            try:
                if domain in models:
                    del models[domain]
                load_domain(domain)
                logger.info(f"Auto-reloaded model for active domain: {domain}")
            except Exception as e:
                logger.warning(f"Could not auto-reload domain {domain}: {e}")
        
        logger.info(f"Model training completed for domain: {domain}")
        
        update_pipeline_state(
            status=PipelineStatus.COMPLETED,
            completion_time=datetime.utcnow().isoformat() + "Z",
            current_step="Pipeline completed successfully",
            ml_model_path=str(domain_dir / "classifier.pkl"),
            metrics=metrics
        )
        
        logger.info(f"Pipeline completed successfully for domain: {domain}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        update_pipeline_state(
            status=PipelineStatus.FAILED,
            completion_time=datetime.utcnow().isoformat() + "Z",
            current_step="Pipeline failed",
            error=str(e)
        )

async def generate_synthetic_data(config: IntentsConfig, output_file: Path):
    """Generate synthetic training data using the configuration"""
    from ml_utils.synthetic_data_utils import (
        random_email, random_phone, random_name, add_typos,
        POLITENESS_PREFIXES, PUNCTUATION
    )
    
    # Use provided roles/skills or defaults
    roles = config.roles or [
        "backend engineer", "frontend developer", "data scientist",
        "site reliability engineer", "product manager", "devops engineer"
    ]
    skills = config.skills or [
        "react", "python", "golang", "spark", "kubernetes", "java"
    ]
    
    rows = []
    idc = 1
    
    for intent_name, intent_def in config.intents.items():
        templates = intent_def.templates
        
        for _ in range(config.samples_per_intent):
            t = random.choice(templates)
            
            # Replace placeholders
            t = t.replace("{role}", random.choice(roles))
            t = t.replace("{skill}", random.choice(skills))
            t = t.replace("{email}", random_email())
            t = t.replace("{phone}", random_phone())
            t = t.replace("{name}", random_name())
            
            # Add variations
            if random.random() < 0.3:
                t = random.choice(POLITENESS_PREFIXES) + t
            
            if random.random() < 0.5:
                t = t.capitalize()
            elif random.random() < 0.3:
                t = t.upper()
            
            if random.random() < 0.6:
                t = t + random.choice(PUNCTUATION)
            
            # Add noise
            if config.add_noise and random.random() < 0.1:
                t = add_typos(t)
            
            # Add whitespace variations
            if random.random() < 0.05:
                t = "  " + t + "  "
            
            rows.append({
                "id": idc,
                "utterance": t,
                "intent": intent_name,
                "annotator": "synthetic",
                "created_at": (datetime.utcnow() - timedelta(days=random.randint(0, 90))).isoformat() + "Z",
                "domain": config.domain,
                "split": "train" if random.random() < config.train_split else "validation"
            })
            idc += 1
    
    random.shuffle(rows)
    
    # Write to CSV
    with open(output_file, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "utterance", "intent", "annotator", 
                                                "created_at", "domain", "split"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    
    logger.info(f"Generated {len(rows)} samples and saved to {output_file}")
    
    # Upload to S3
    try:
        from ml_utils.s3_handler import S3Handler
        s3 = S3Handler()
        s3_key = f"models/{config.domain}/{output_file.name}"
        s3.upload_file(str(output_file), s3_key)
        logger.info(f"Uploaded synthetic dataset to S3: {s3_key}")
    except Exception as e:
        logger.warning(f"Failed to upload synthetic dataset to S3: {e}")

async def train_model(dataset_path: Path, output_dir: Path) -> Dict:
    """Train the classification model"""
    # Load data
    df = pd.read_csv(dataset_path)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # Create label mappings
    labels = sorted(df['intent'].unique().tolist())
    label2idx = {l: i for i, l in enumerate(labels)}
    idx2label = {str(i): l for l, i in label2idx.items()}
    df['label_idx'] = df['intent'].map(label2idx)
    
    # Split data
    train_df, test_df = train_test_split(
        df, test_size=0.15, random_state=42, stratify=df['label_idx']
    )
    
    logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")
    
    # Load embedder and generate embeddings
    embedder = SentenceTransformer(EMBED_MODEL)
    
    X_train = embedder.encode(
        train_df['utterance'].tolist(), 
        show_progress_bar=False, 
        convert_to_numpy=True
    )
    X_test = embedder.encode(
        test_df['utterance'].tolist(), 
        show_progress_bar=False, 
        convert_to_numpy=True
    )
    y_train = train_df['label_idx'].values
    y_test = test_df['label_idx'].values
    
    # Train classifier
    clf = LogisticRegression(max_iter=2000, multi_class='ovr')
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    logger.info(f"Model accuracy: {accuracy}")
    
    # Save model and metadata
    joblib.dump(clf, output_dir / "classifier.pkl")
    
    meta = {
        "embed_model": EMBED_MODEL,
        "label2idx": label2idx,
        "idx2label": idx2label,
        "training_date": datetime.utcnow().isoformat() + "Z",
        "accuracy": float(accuracy),
        "num_intents": len(labels)
    }
    
    with open(output_dir / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    # Calculate metrics
    intent_counts = df['intent'].value_counts().to_dict()
    
    metrics = {
        "accuracy": float(accuracy),
        "intent_distribution": intent_counts,
        "train_samples": len(train_df),
        "validation_samples": len(test_df)
    }
    
    logger.info(f"Model saved to {output_dir}")
    
    # Upload to S3
    try:
        from ml_utils.s3_handler import S3Handler
        s3 = S3Handler()
        
        # Upload classifier
        classifier_path = output_dir / "classifier.pkl"
        s3_key_clf = f"models/{output_dir.name}/classifier.pkl"
        s3.upload_file(str(classifier_path), s3_key_clf)
        
        # Upload metadata
        meta_path = output_dir / "model_meta.json"
        s3_key_meta = f"models/{output_dir.name}/model_meta.json"
        s3.upload_file(str(meta_path), s3_key_meta)
        
        logger.info(f"Uploaded model artifacts to S3: {s3_key_clf}, {s3_key_meta}")
    except Exception as e:
        logger.warning(f"Failed to upload model artifacts to S3: {e}")
    
    return metrics

