import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau 

from tqdm import tqdm
from pathlib import Path

from .utils import true_positive, false_positive, false_negative, f1_score

class EarlyStopping:
    def __init__(self, tol, consecutive = 0):
        self.tol = tol
        self.consecutive = consecutive
        
        self.counter = 0
        self.early_stop = False
    
    def __call__(self, train_score, validation_score):
        if validation_score -train_score >= self.tol:
            self.counter += 1
            if self.counter >= self.consecutive:
                self.early_stop = True

class GanClassifier:
    def __init__(self, input_size, latent_size, device = "cpu"):
        self.latent_size = latent_size

        self.generator = nn.Sequential(
            nn.Linear(latent_size, latent_size*2),
            nn.Tanh(),
            nn.Linear(latent_size*2, latent_size*4),
            nn.Tanh(),
            nn.Linear(latent_size*4, input_size)
        )
        self.discriminator = nn.Sequential(
            nn.Linear(input_size, latent_size),
            nn.Tanh(),
            nn.Linear(latent_size, 1),
            nn.Sigmoid()
        )
        self.criterion = nn.BCELoss()
        self.optimiserG = optim.Adam(self.generator.parameters(), lr=0.0001)
        self.optimiserD = optim.Adam(self.discriminator.parameters(), lr=0.0001)
        self.schedulerG = ReduceLROnPlateau(self.optimiserG, "min", patience=5)
        self.schedulerD = ReduceLROnPlateau(self.optimiserD, "min", patience=5)

        self.device = device
        self.real_label = 0.0
        self.fake_label = 1.0
    
    def fit(self, X, y, X_validation, y_validation, epochs, batch_size=32, make_checkpoint=False, from_checkponit=False, tol=0.0, consecutive = 0): 
        self.generator.to(self.device)
        self.discriminator.to(self.device)

        early_stopping = EarlyStopping(tol=tol, consecutive=consecutive)

        start_epoch = 0
        history_score_legal_train = []
        history_score_fraud_train = []
        history_score_legal_validation = []
        history_score_fraud_validation = []
        history_loss_train_D = []   
        history_loss_train_G = []   
        history_loss_validation_D = []   
        history_loss_validation_G = []   
        if from_checkponit:
            Path("./checkpoint").mkdir(parents=True, exist_ok=True)
            try:
                checkpoint = torch.load("./checkpoint/gan.pt")
                start_epoch = checkpoint["epoch"]

                self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
                self.generator.load_state_dict(checkpoint["generator_state_dict"])
                self.optimiserD.load_state_dict(checkpoint["optimiserD_state_dict"])
                self.optimiserG.load_state_dict(checkpoint["optimiserG_state_dict"])
                self.schedulerD.load_state_dict(checkpoint["schedulerD_state_dict"])
                self.schedulerG.load_state_dict(checkpoint["schedulerG_state_dict"])

                history_score_legal_train = checkpoint["history_score_legal_train"]
                history_score_fraud_train = checkpoint["history_score_fraud_train"]
                history_score_legal_validation = checkpoint["history_score_legal_validation"]
                history_score_fraud_validation = checkpoint["history_score_fraud_validation"]
                history_loss_train_D = checkpoint["history_loss_train_D"]
                history_loss_train_G = checkpoint["history_loss_train_G"]
                history_loss_validation_D = checkpoint["history_loss_validation_D"]
                history_loss_validation_G = checkpoint["history_loss_validation_G"]
            except FileNotFoundError:
                print("No checkpoint has been created yet. Training starts from scratch.")
                
        loader_train = DataLoader(list(zip(X, y.reshape(y.size(0), 1).float())), shuffle=True, batch_size=batch_size)
        loader_validation = DataLoader(list(zip(X_validation, y_validation.reshape(y_validation.size(0), 1).float())), shuffle=True, batch_size=batch_size)
        
        for epoch in tqdm(range(start_epoch, epochs)):
            for X_batch, y_batch in loader_train:
                size = X_batch.size(0)

                X_real = X_batch.to(self.device)
                label_real = torch.full((size, 1), self.real_label, device=self.device)
                # Train discriminator
                self.discriminator.train()
                self.generator.eval()
                # update Discriminator on all real items
                self.optimiserD.zero_grad()
                out_real = self.discriminator(X_real)
                errorD_real = self.criterion(out_real, label_real)
                # update discriminator on fake items
                noise = torch.randn((size, self.latent_size), device=self.device) # generate batch of latent vectors
                X_fake = self.generator(noise)
                label_fake = torch.full((size, 1), self.fake_label, device=self.device)
                out_fake = self.discriminator(X_fake.detach())
                errorD_fake = self.criterion(out_fake, label_fake)
                errorD = errorD_real + errorD_fake
                errorD.backward()
                self.optimiserD.step()
                
                # update Generator
                self.discriminator.eval()
                self.generator.train()
                self.optimiserG.zero_grad()
                label = torch.full((size, 1), self.real_label, device=self.device) # maybe problem here
                out = self.discriminator(X_fake)
                errorG = -self.criterion(out, label)
                errorG.backward()
                self.optimiserG.step()  

            (loss_train_D, loss_train_G), (f1_score_legal_train, f1_score_fraud_train) = self.score(X, y)
            (loss_validation_D, loss_validation_G), (f1_score_legal_validation, f1_score_fraud_validation) = self.score(X_validation, y_validation)

            self.schedulerD.step(loss_validation_D)
            self.schedulerG.step(loss_validation_G)
            print(f1_score_legal_train, f1_score_fraud_train)
            print(f"Epoch {epoch} TRAIN: loss =", round(loss_train_D, 3),", F1 score legal=", int(100*f1_score_legal_train), "%, F1 score Fraud=", int(100*f1_score_fraud_train), "%")
            print("\tVALIDATION: loss =", round(loss_validation_G, 3),", F1 score legal=", int(100*f1_score_legal_validation), "%, F1 score Fraud=", int(100*f1_score_fraud_validation), "%")
            
            history_score_legal_train.append(f1_score_legal_train)
            history_score_fraud_train.append(f1_score_fraud_train)
            history_score_legal_validation.append(f1_score_legal_validation)
            history_score_fraud_validation.append(f1_score_fraud_validation)
            history_loss_train_D.append(loss_train_D)
            history_loss_train_G.append(loss_train_G)
            history_loss_validation_D.append(loss_validation_D)
            history_loss_validation_G.append(loss_validation_G)
            
            if make_checkpoint:
                torch.save({'epoch': epoch, 
                                'discriminator_state_dict': self.discriminator.state_dict(),
                                'generator_state_dict': self.generator.state_dict(),
                                'optimiserD_state_dict': self.optimiserD.state_dict(),
                                'optimiserG_state_dict': self.optimiserG.state_dict(),
                                "schedulerD_state_dict": self.schedulerD.state_dict(),
                                "schedulerG_state_dict": self.schedulerG.state_dict(),
                                "history_score_legal_train": history_score_legal_train, "history_score_fraud_train": history_score_fraud_train,  
                                "history_loss_train_D": history_loss_train_D, "history_loss_train_G": history_loss_train_G,
                                "history_score_legal_validation": history_score_legal_validation, "history_score_fraud_validation": history_score_fraud_validation,  
                                "history_loss_validation_D": history_loss_validation_D, "history_loss_validation_G": history_loss_validation_G,
                                },
                                "./checkpoint/gan.pt")
            early_stopping(train_score=f1_score_fraud_train, validation_score=f1_score_fraud_validation)
            
            if early_stopping.early_stop:
                print(f"Training completed within {epoch} epochs")
                return
        
        print(f"Maximum number of epochs reached. Consider increasing the number of epochs.")

    def score(self, X, y, batch_size=128):
        self.discriminator.eval()
        self.generator.eval()

        mask = torch.rand((X.size(0), )) <= 0.3
        X = X[mask,:]
        y = y[mask]
        size = X.size(0)
        noise = torch.randn((size, self.latent_size), device=self.device) # generate batch of latent vectors
        X_fake = self.generator(noise).cpu()
        label_fake = torch.full((size,), self.fake_label)
        X = torch.cat((X, X_fake), dim=0)
        y = torch.cat((y, label_fake), dim=0)

        loader = DataLoader(list(zip(X, y.reshape(y.size(0), 1).float())), shuffle=True, batch_size=batch_size)
        (L_discriminator, L_generator), (f1_score_legal, f1_score_fraud) = self._score_loader(loader)
        return (L_discriminator, L_generator), (f1_score_legal, f1_score_fraud)

    def _score_loader(self, loader):
        L_discriminator = 0.0
        L_generator = 0.0

        self.discriminator.to(self.device)
        self.generator.to(self.device)
        tp_fraud = 0
        fp_fraud = 0
        fn_fraud = 0
        tp_legal = 0
        fp_legal = 0
        fn_legal = 0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            y_pred = self.discriminator(X_batch)
            L_discriminator += self.criterion(y_pred, y_batch).item()

            noise = torch.randn((X_batch.size(0), self.latent_size), device=self.device) # generate batch of latent vectors
            fake = self.generator(noise)
            out_fake = self.discriminator(fake)
            label_fake = torch.full((X_batch.size(0), 1), self.fake_label, device=self.device)
            L_generator += self.criterion(out_fake, label_fake).item()

            y_pred = (y_pred >= 0.5).long()
            y_batch = y_batch.cpu().long()
            y_pred = y_pred.cpu().long()

            tp_legal += true_positive(y_batch, y_pred, label=0)
            fp_legal += false_positive(y_batch, y_pred, label=0)
            fn_legal += false_negative(y_batch, y_pred, label=0)

            tp_fraud += true_positive(y_batch, y_pred, label=1)
            fp_fraud += false_positive(y_batch, y_pred, label=1)
            fn_fraud += false_negative(y_batch, y_pred, label=1)
            
        f1_score_legal = f1_score(tp_legal, fp_legal, fn_legal)
        f1_score_fraud = f1_score(tp_fraud, fp_fraud, fn_fraud)
        
        L_discriminator /= len(loader)
        L_generator /= len(loader)
        
        return (L_discriminator, L_generator), (f1_score_legal, f1_score_fraud)
    
    def predict(self, X):
        X = X.to(self.device)
        y_pred = self.discriminator(X).cpu() 
        _, y_pred = torch.max(y_pred, 1)
        return y_pred

