"""
Training utilities: EarlyStopping, training loop, evaluation loop.
"""
import os
import time
import numpy as np
import torch


class EarlyStopping:
    """
    Stops training when validation loss doesn't improve for `patience` epochs.
    Saves the best model checkpoint automatically.
    """

    def __init__(self, patience=10, delta=1e-6, path="models/best_model.pth", verbose=True):
        self.patience  = patience
        self.delta     = delta
        self.path      = path
        self.verbose   = verbose
        self.counter   = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self._save(model)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def _save(self, model):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        torch.save(model.state_dict(), self.path)
        if self.verbose:
            print(f"  ✓ Best model saved → {self.path}  (val_loss={self.best_loss:.6f})")


def train_one_epoch(model, loader, optimizer, criterion, device, clip_grad=1.0):
    """Run one training epoch, return average loss."""
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss  = criterion(preds, y_batch)
        loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model on a DataLoader, return average loss."""
    model.eval()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        preds   = model(X_batch)
        loss    = criterion(preds, y_batch)
        total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict_all(model, loader, device):
    """Return concatenated predictions and labels as numpy arrays."""
    model.eval()
    all_preds, all_labels = [], []
    for X_batch, y_batch in loader:
        preds = model(X_batch.to(device)).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(y_batch.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


def full_train(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    scheduler,
    early_stopping,
    device,
    max_epochs=100,
    model_name="Model",
):
    """
    Full training loop with early stopping and scheduler.
    Returns (train_losses, val_losses) lists.
    """
    train_losses, val_losses = [], []
    print(f"\n{'━'*50}")
    print(f"  Training {model_name} on {device}")
    print(f"{'━'*50}")
    t0 = time.time()

    for epoch in range(1, max_epochs + 1):
        tr_loss  = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        train_losses.append(tr_loss)
        val_losses.append(val_loss)

        if scheduler is not None:
            scheduler.step(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{max_epochs} | train={tr_loss:.6f} | val={val_loss:.6f}")

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"  ⛔ Early stopping at epoch {epoch}")
            break

    elapsed = time.time() - t0
    print(f"\n  ✅ Training complete in {elapsed:.1f}s")
    # reload best weights
    model.load_state_dict(torch.load(early_stopping.path, weights_only=True))
    return train_losses, val_losses


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
