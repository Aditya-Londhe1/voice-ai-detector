import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from custom_dataset import CustomDeepfakeDataset

from model import RawNetLite


# -------- CONFIG --------

DEVICE = "cpu"
EPOCHS = 20
BATCH = 16
LR = 1e-3


MODEL_PATH = "models/rawnet_custom.pth"



# -------- TRAIN --------

def evaluate(model, loader):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for x, y in loader:

            out = model(x).squeeze()

            pred = (torch.sigmoid(out) > 0.5)

            correct += (pred == y.bool()).sum().item()

            total += y.size(0)

    return correct / total


def train():

    dataset = CustomDeepfakeDataset(
    r"C:\Users\sai\Downloads\archive"
)


    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_set, val_set = random_split(
        dataset,
        [train_size, val_size]
    )

    train_loader = DataLoader(
    train_set,
    batch_size=BATCH,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)


    val_loader = DataLoader(
        val_set,
        batch_size=BATCH
    )

    model = RawNetLite().to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )

    loss_fn = nn.BCEWithLogitsLoss()


    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0

        for x, y in tqdm(train_loader):

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            out = model(x).squeeze()

            loss = loss_fn(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()


        acc = evaluate(model, val_loader)

        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"Loss: {total_loss:.3f}")
        print(f"Val Acc: {acc:.3f}")
        print("-" * 30)


    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)


    print("Model saved to:", MODEL_PATH)


if __name__ == "__main__":
    train()
