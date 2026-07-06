import torch
import torch.nn as nn
import torch.optim as optim
import logging
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

class BiLSTMAttnTrainer:
    def __init__(self, model, config, device, vocab, class_weights=None):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.vocab = vocab
        
        weight_tensor = None
        if class_weights is not None:
            # Reorder weights to match vocab.id2label
            weights = [class_weights.get(vocab.id2label[i], 1.0) for i in range(len(vocab.id2label))]
            weights[0] = 0.0 # Weight for <PAD> is 0
            weight_tensor = torch.tensor(weights, dtype=torch.float).to(device)
            logger.info(f"Using class weights: {weights}")
            
        self.criterion = nn.CrossEntropyLoss(ignore_index=0, weight=weight_tensor)
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=config.get("learning_rate", 1e-3),
            weight_decay=config.get("weight_decay", 1e-4)
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=2
        )
        
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        
        for batch_idx, (seqs, labels, lengths) in enumerate(dataloader):
            seqs, labels = seqs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            logits, _ = self.model(seqs, lengths)
            
            logits = logits.view(-1, logits.shape[-1])
            labels = labels.view(-1)
            
            loss = self.criterion(logits, labels)
            loss.backward()
            
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(dataloader)
        
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for seqs, labels, lengths in dataloader:
                seqs, labels = seqs.to(self.device), labels.to(self.device)
                logits, _ = self.model(seqs, lengths)
                
                logits = logits.view(-1, logits.shape[-1])
                labels = labels.view(-1)
                
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                
        return total_loss / len(dataloader)
        
    def train(self, train_loader, val_loader, epochs, save_dir: Path):
        best_val_loss = float('inf')
        patience_counter = 0
        patience_limit = self.config.get("early_stopping_patience", 3)
        
        train_losses, val_losses = [], []
        
        for epoch in range(1, epochs + 1):
            logger.info(f"Epoch {epoch}/{epochs}")
            
            train_loss = self.train_epoch(train_loader)
            val_loss = self.evaluate(val_loader)
            
            self.scheduler.step(val_loss)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            logger.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), save_dir / "best_model.pt")
                logger.info(f"Saved new best model with Val Loss: {val_loss:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience_limit:
                    logger.info("Early stopping triggered.")
                    break
                    
        return train_losses, val_losses
