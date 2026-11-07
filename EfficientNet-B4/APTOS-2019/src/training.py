import torch
import os
import logging
from sklearn.metrics import f1_score
from tqdm.notebook import tqdm

def validation(model, valid_loader, criterion, device):
    model.eval()
    total_valid_loss = 0.0
    valid_corrects = 0
    all_valid_preds = []
    all_valid_labels = []

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(valid_loader), total=len(valid_loader), desc="Validation:" ):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_valid_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, dim=1)
            valid_corrects += torch.sum(preds == labels.data)
            all_valid_preds.extend(preds.cpu().numpy())
            all_valid_labels.extend(labels.cpu().numpy())

    validation_loss = total_valid_loss / len(valid_loader.dataset)
    validation_accuracy = valid_corrects.double() / len(valid_loader.dataset)
    validation_f1 = f1_score(all_valid_labels, all_valid_preds, average='macro')

    return validation_loss, validation_accuracy, validation_f1

def training(
        model,
        train_loader,
        valid_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        chkpt_dir,
        training_record,
        num_epochs,
        best_validation_f1,
        start_epoch=0
):
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_train_loss = 0.0
        train_corrects = 0
        all_train_preds = []
        all_train_labels = []

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Training: Epoch {epoch + 1}/{num_epochs}')

        for i, (images, labels) in progress_bar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, dim=1)
            train_corrects += torch.sum(preds == labels.data)
            all_train_preds.extend(preds.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())

        training_loss = total_train_loss / len(train_loader.dataset)
        training_accuracy = train_corrects.double() / len(train_loader.dataset)
        training_f1 = f1_score(all_train_labels, all_train_preds, average='macro')

        validation_loss, validation_accuracy, validation_f1 = validation(model, valid_loader, criterion, device)
        scheduler.step(validation_f1)

        training_record["train_loss"].append(training_loss)
        training_record["valid_loss"].append(validation_loss)
        training_record["train_f1"].append(training_f1)
        training_record["valid_f1"].append(validation_f1)

        print(
            f'Epoch {epoch + 1}/{num_epochs}:\n'
            f'Train Loss: {training_loss:.4f}, Train Accuracy: {training_accuracy:.4f}, Train F1: {training_f1:.4f}\n'
            f'Valid Loss: {validation_loss:.4f}, Valid Accuracy: {validation_accuracy:.4f}, Valid F1: {validation_f1:.4f}')

        if validation_f1 > best_validation_f1:
            best_validation_f1 = validation_f1
            checkpoint_path = os.path.join(chkpt_dir, f"epoch{epoch+1}_validF1_{best_validation_f1:.4f}.pth")
            torch.save({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_valid_f1": best_validation_f1,
            }, checkpoint_path)
            print("best model saved to: ", checkpoint_path)
            best_checkpoint_path = os.path.join(chkpt_dir, "best_checkpoint.pth")
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_valid_f1": best_validation_f1,
            }, best_checkpoint_path)

        logging.info(
            f'Epoch {epoch + 1}/{num_epochs}'
            f'Train Loss: {training_loss:.4f}, Train Accuracy: {training_accuracy:.4f}, Train F1: {training_f1:.4f},'
            f'Valid Loss: {validation_loss:.4f}, Valid Accuracy: {validation_accuracy:.4f}, Valid F1: {validation_f1:.4f}')

    return training_record

