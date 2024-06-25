from src.utils import load_data
from src.model import GanClassifier
import torch
import matplotlib.pyplot as plt

(X_train, y_train), (X_validation, y_validation) = load_data(proportion_validation=0.2)

# CNN Classifiar
classifier = GanClassifier(X_train.size(1), latent_size=32, device="cpu")
classifier.fit(
    X_train,
    y_train,
    X_validation,
    y_validation,
    epochs=50,
    from_checkponit=True,
    make_checkpoint=True,
    tol=0.001,
    consecutive=10,
)

(L_discriminator, L_generator), (f1_score_legal, f1_score_fraud) = classifier.score(X_validation, y_validation)

print(
    "Gan Classifier F1 score, Legal:",
    int(100 * f1_score_legal),
    "%, Fraud:",
    int(100 * f1_score_fraud), "%"
)


checkpoint = torch.load("./checkpoint/gan.pt")
history_score_legal_train = checkpoint["history_score_legal_train"]
history_score_fraud_train = checkpoint["history_score_fraud_train"]
history_score_legal_validation = checkpoint["history_score_legal_validation"]
history_score_fraud_validation = checkpoint["history_score_fraud_validation"]
history_loss_train_D = checkpoint["history_loss_train_D"]
history_loss_train_G = checkpoint["history_loss_train_G"]
history_loss_validation_D = checkpoint["history_loss_validation_D"]
history_loss_validation_G = checkpoint["history_loss_validation_G"]

fig, ax = plt.subplots(2, 2, sharex=True)
ax[0, 0].plot(history_loss_train_G, label="train")
ax[0, 0].plot(history_loss_validation_G, label="valid")
ax[0, 0].set_title("Generator loss")
ax[0, 1].plot(history_loss_train_D, label="train")
ax[0, 1].plot(history_loss_validation_G, label="valid")
ax[0, 1].set_title("Discriminator loss")

ax[1, 0].plot(history_score_legal_train, label="train")
ax[1, 0].plot(history_score_legal_validation, label="valid")
ax[1, 0].set_title("F1-score legal")
ax[1, 0].set_xlabel("epoch")
ax[1, 1].plot(history_score_fraud_train, label="train")
ax[1, 1].plot(history_score_fraud_validation, label="valid")
ax[1, 1].set_title("F1-score fraud")
ax[1, 1].set_xlabel("epoch")
handles, labels = ax[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper left")
fig.tight_layout()
fig.savefig("./figures/loss_score.png")
