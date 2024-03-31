import torch

from torchmetrics.regression import R2Score

from tqdm import tqdm


def train(model, dataloader, criterion, optimizer, device):
    model.train()

    r2score = R2Score().cpu()
    train_loss = 0.0
    r2_train = 0.0
    
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, 
                     leave=False, position=0, desc="Train")

    for i, batch in enumerate(dataloader):
        inputs = batch.to(device) # (batch_size, 1, 64, 1000)

        outputs = model(inputs)
        loss = criterion(outputs, inputs)

        train_loss += loss.item()
        r2_train += r2score(outputs.flatten().cpu(), inputs.flatten().cpu())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        batch_bar.set_postfix(
            loss = f"{train_loss/(i+1):.4f}",
            r2_loss = f"{r2_train/(i+1):.4f}", 
            lr = f"{optimizer.param_groups[0]['lr']:.4f}"
        )
        batch_bar.update()             
        
        torch.cuda.empty_cache()
        del inputs, outputs
    
    batch_bar.close()
    train_loss /= len(dataloader)
    r2_train /= len(dataloader)

    return train_loss, r2_train


def validate(model, dataloader, criterion, optimizer, device):
    model.eval()

    r2score = R2Score().cpu()
    valid_loss = 0.0
    r2_valid = 0.0
    
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True,
                     leave=False, position=0, desc="Validation")
    
    for i, batch in enumerate(dataloader):
        inputs = batch.to(device)
        
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
        
        valid_loss += loss.item()
        r2_valid += r2score(outputs.flatten().cpu(), inputs.flatten().cpu())
        
        batch_bar.set_postfix(
            loss = f"{valid_loss/(i+1):.4f}",
            r2_loss = f"{r2_valid/(i+1):.4f}", 
            lr = f"{optimizer.param_groups[0]['lr']:.4f}"
        )
        batch_bar.update()   

        torch.cuda.empty_cache()
        del inputs, outputs
    
    batch_bar.close()
    valid_loss /= len(dataloader)
    r2_valid /= len(dataloader)
    
    return valid_loss, r2_valid
