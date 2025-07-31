from collections import defaultdict
import tqdm
import torch
import wandb
import numpy as np
from sksurv.metrics import concordance_index_censored
from pycox.models.loss import NLLLogistiHazardLoss

from morpheus.utils.data import compute_clf_metrics


def loop_pretrain(model, omics_modalities, train_dataloader, val_dataloader, optimizer, lr_scheduler, epochs, device, base_path_save, wandb_logging):
    metrics = defaultdict(list)
    for epoch in range(epochs):
        metrics['lr'].append(optimizer.param_groups[0]['lr'])
        print(f"Epoch {epoch + 1}/{epochs}")
        print("-" * 10)
        model.train()
        epoch_loss = 0.
        total_modality_loss = {modality: 0. for modality in omics_modalities}
        for _, inputs in enumerate(tqdm.tqdm(train_dataloader, desc='Training')):
            # put wsi to device
            wsi = inputs['wsi'].to(device)
            bs = wsi.shape[0]
            omics_data = {modality: inputs[modality] for modality in omics_modalities}
            for modality in omics_data.keys():
                omics_data[modality] = {k: v.to(device) for k, v in omics_data[modality].items()}
            
            optimizer.zero_grad()
            outputs, task_masks = model(wsi, omics_data)
            modality_loss = {}
            for modality in omics_data.keys():
                temp_loss = []
                for k in outputs[modality].keys():
                    temp_loss_k = torch.abs(outputs[modality][k] - omics_data[modality][k])
                    temp_loss.append(temp_loss_k.mean(-1))

                temp_loss = torch.stack(temp_loss, dim=1)
                temp_loss = (temp_loss * task_masks[modality]).sum() / task_masks[modality].sum()
                modality_loss[modality] = temp_loss
                
            loss = sum(modality_loss.values())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * bs
        
            for modality in omics_data.keys():
                total_modality_loss[modality] += modality_loss[modality].item() * bs

        lr_scheduler.step()
        epoch_loss /= len(train_dataloader.dataset)
        print(f"Epoch Loss: {epoch_loss}")
        for modality in omics_data.keys():
            total_modality_loss[modality] /= len(train_dataloader.dataset)
            print(f"{modality} loss: {total_modality_loss[modality]:.4f}")
            metrics[f"train/{modality}_loss"].append(total_modality_loss[modality])
            metrics["train/loss"].append(epoch_loss)

        # validation loop every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            val_loss = 0.
            total_modality_loss = {modality: 0. for modality in omics_modalities}
            model.eval()
            with torch.no_grad():
                for _, inputs in enumerate(tqdm.tqdm(val_dataloader, desc='Validation')):
                    wsi = inputs['wsi'].to(device)
                    bs = wsi.shape[0]
                    omics_data = {modality: inputs[modality] for modality in omics_modalities}
                    for modality in omics_data.keys():
                        omics_data[modality] = {k: v.to(device) for k, v in omics_data[modality].items()}

                    outputs, task_masks = model(wsi, omics_data)
                    modality_loss = {}
                    for modality in omics_data.keys():
                        temp_loss = []
                        for k in outputs[modality].keys():
                            temp_loss_k = torch.abs(outputs[modality][k] - omics_data[modality][k])
                            temp_loss.append(temp_loss_k.mean(-1))
                        temp_loss = torch.stack(temp_loss, dim=1)
                        temp_loss = (temp_loss * task_masks[modality]).sum() / task_masks[modality].sum()
                        modality_loss[modality] = temp_loss

                    loss = sum(modality_loss.values())
                    val_loss += loss.item() * bs
                    for modality in omics_data.keys():
                        total_modality_loss[modality] += modality_loss[modality].item() * bs
                
            val_loss /= len(val_dataloader.dataset)
            print(f"Validation Loss: {val_loss}")
            for modality in total_modality_loss:
                total_modality_loss[modality] /= len(val_dataloader.dataset)
                print(f"{modality} loss: {total_modality_loss[modality]:.4f}")
                metrics[f"val/{modality}_loss"].append(total_modality_loss[modality])
                metrics["val/loss"].append(val_loss)

        # save model every 100 epochs
        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(), f"{base_path_save}{epoch + 1}.pth")
            print(f"Model saved at epoch {epoch + 1}")
        
            
        if wandb_logging:
            wandb.log({k: v[-1] for k, v in metrics.items()})
            
    return model, metrics


def loop_survival(model, omics_modalities, train_dataloader, val_dataloader, optimizer, lr_scheduler, epochs, device, wandb_logging, accum_iter=None):
    loss_fn = NLLLogistiHazardLoss(reduction='sum')
    metrics = defaultdict(list)
    for epoch in range(epochs):
        # add lr to metrics
        metrics["lr"].append(optimizer.param_groups[0]["lr"])
        print(f"Epoch {epoch + 1}/{epochs}")
        print("-" * 10)
        model.train()
        epoch_loss = 0.0
        for i, inputs in enumerate(tqdm.tqdm(train_dataloader, desc="Training...")):
            wsi = inputs['wsi'].to(device)
            time = inputs['time'].to(device)
            censorship = inputs['censorship'].to(device)
            surv_label = inputs['label'].to(device)
            if not accum_iter:
                optimizer.zero_grad()
                
            if len(omics_modalities) == 0:
                outputs = model(wsi)
            else:
                # create omic data
                omics_data = {modality: inputs[modality] for modality in omics_modalities}
                for modality in omics_data.keys():
                    if isinstance(omics_data[modality], dict):
                        omics_data[modality] = {k: v.to(device) for k, v in omics_data[modality].items()}
                    else:
                        omics_data[modality] = omics_data[modality].to(device)
                outputs = model(wsi, omics_data)

            loss = loss_fn(outputs, surv_label, torch.abs(1 - censorship).float())
            loss.backward()
            if accum_iter:
                if ((i + 1) % accum_iter == 0) or (i + 1 == len(train_dataloader)):
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                optimizer.step()

            epoch_loss += loss.item() * surv_label.shape[0]

        lr_scheduler.step()
        epoch_loss /= len(train_dataloader.dataset)
        print(f"Training loss: {epoch_loss:.4f}")
        metrics["train/loss"].append(epoch_loss)
        
        if epoch == epochs - 1:
            val_loss = 0.0
            model.eval()
            times = []
            censorships = []
            risks = []
            with torch.no_grad():
                for _, inputs in enumerate(tqdm.tqdm(val_dataloader, desc="Validation...")):
                    wsi = inputs['wsi'].to(device)
                    time = inputs['time'].to(device)
                    censorship = inputs['censorship'].to(device)
                    surv_label = inputs['label'].to(device)
                    if len(omics_modalities) == 0:
                        outputs = model(wsi)
                    else:
                    # create omic data
                        omics_data = {modality: inputs[modality] for modality in omics_modalities}
                        for modality in omics_data.keys():
                            if isinstance(omics_data[modality], dict):
                                omics_data[modality] = {k: v.to(device) for k, v in omics_data[modality].items()}
                            else:
                                omics_data[modality] = omics_data[modality].to(device)
                            
                        outputs = model(wsi, omics_data)
                        
                    loss = loss_fn(outputs, surv_label, torch.abs(1 - censorship).float())
                    val_loss += loss.item() * surv_label.size(0)
                    times.append(time.cpu().numpy())
                    censorships.append(censorship.cpu().numpy())
                    risks.append(outputs.cpu().numpy().sum(axis=1)) 

                val_loss /= len(val_dataloader.dataset)
                print(f"Validation loss: {val_loss:.4f}")
                metrics["val/loss"].append(val_loss)
                # roc_auc and ap
                times = np.concatenate(times)
                censorships = np.concatenate(censorships)
                risks = np.concatenate(risks)
                c_index = concordance_index_censored(np.abs((censorships - 1)).astype(bool), times, risks)[0]
                metrics["val/c_index"].append(c_index)
                print(f"c_index: {c_index:.4f}")

        if wandb_logging:
            wandb.log({k: v[-1] for k, v in metrics.items()})

    return model, metrics



def loop_classification(model, omics_modalities, train_dataloader, val_dataloader, optimizer, lr_scheduler, epochs, device, wandb_logging, accum_iter=None):
    loss_fn = torch.nn.BCEWithLogitsLoss()    
    metrics = defaultdict(list)
    for epoch in range(epochs):
        # add lr to metrics
        metrics["lr"].append(optimizer.param_groups[0]["lr"])
        print(f"Epoch {epoch + 1}/{epochs}")
        print("-" * 10)
        model.train()
        epoch_loss = 0.0
        for i, inputs in enumerate(tqdm.tqdm(train_dataloader, desc="Training...")):
            wsi = inputs['wsi'].to(device)
            label = inputs['label'].to(device)
            if not accum_iter:
                optimizer.zero_grad()
            if len(omics_modalities) == 0:
                outputs = model(wsi)
            else:
                # create omic data
                omics_data = {modality: inputs[modality] for modality in omics_modalities}
                for modality in omics_data.keys():
                        omics_data[modality] = {k: v.to(device) for k, v in omics_data[modality].items()}
                        
                outputs = model(wsi, omics_data)

            
            loss = loss_fn(outputs.squeeze(-1), label.float())
            
            loss.backward()
            if accum_iter:
                if ((i + 1) % accum_iter == 0) or (i + 1 == len(train_dataloader)):
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                optimizer.step()
            # optimizer.step()
            epoch_loss += loss.item() * label.shape[0]
        
        if lr_scheduler:
            lr_scheduler.step()
        epoch_loss /= len(train_dataloader.dataset)
        print(f"Training loss: {epoch_loss:.4f}")
        metrics["train/loss"].append(epoch_loss)

        if epoch == epochs - 1:
            # do validation
            val_loss = 0.0
            model.eval()
            predictions = []
            labels = []
            with torch.no_grad():
                for _, inputs in enumerate(tqdm.tqdm(val_dataloader, desc="Validation...")):
                    wsi = inputs['wsi'].to(device)
                    label = inputs['label'].to(device)
                    if len(omics_modalities) == 0:
                        outputs = model(wsi)
                    else:
                    # create omic data
                        omics_data = {modality: inputs[modality] for modality in omics_modalities}
                        for modality in omics_data.keys():
                            omics_data[modality] = {k: v.to(device) for k, v in omics_data[modality].items()}
                            
                        outputs = model(wsi, omics_data)
                    
                    loss = loss_fn(outputs.squeeze(-1), label.float())
                    predictions.append(outputs.sigmoid().cpu().numpy())
                    
                    val_loss += loss.item() * label.size(0)
                    labels.append(label.cpu().numpy())

                val_loss /= len(val_dataloader.dataset)
                print(f"Validation loss: {val_loss:.4f}")
                metrics["val/loss"].append(val_loss)
                labels = np.concatenate(labels)
                predictions = np.concatenate(predictions)
                roc_auc, balanced_accuracy = compute_clf_metrics(labels, predictions)
                metrics["val/auc"].append(roc_auc)
                metrics["val/balanced_accuracy"].append(balanced_accuracy)
                print(f"ROC AUC: {roc_auc:.4f}")
                print(f"Balanced Accuracy: {balanced_accuracy:.4f}")

        if wandb_logging:
            wandb.log({k: v[-1] for k, v in metrics.items()})

    return model, metrics

