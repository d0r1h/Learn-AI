def accuracy(preds, targets):
    return (preds == targets).float().mean().item() * 100.0