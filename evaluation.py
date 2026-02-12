import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

import umap

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LinearSegmentationProbe(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.head = nn.Conv2d(in_dim, num_classes, kernel_size=1)

    def forward(self, x):
        return self.head(x)

def train_linear_segmentation_probe(
    backbone,
    dataloader,
    num_classes,
    feature_dim,
    epochs=20,
    lr=1e-3
):
    backbone.eval()
    probe = LinearSegmentationProbe(feature_dim, num_classes).to(DEVICE)

    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(epochs):
        probe.train()
        total_loss = 0.0

        for x, y in tqdm(dataloader, desc=f"Seg Probe Epoch {epoch}"):
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            with torch.no_grad():
                feats = backbone.encode_patches(x)  # (N, D, H, W)

            logits = probe(feats)
            logits = F.interpolate(
                logits,
                size=y.shape[-2:],   
                mode="bilinear",
                align_corners=False
            )

            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Seg Probe] Epoch {epoch} | Loss {total_loss / len(dataloader):.4f}")

    return probe


@torch.no_grad()
def extract_representations(model, dataloader):
    """
    Extract frozen image-level representations.
    """
    model.eval()
    feats, labels = [], []

    for x, y in tqdm(dataloader, desc="Extracting representations"):
        x = x.to(DEVICE)
        z = model.encode_image(x)  # (N, D)

        feats.append(z.cpu())
        labels.append(y.cpu())

    return torch.cat(feats), torch.cat(labels)

def pixel_to_image_labels(pixel_labels, ignore_index=0):
    """
    Convert pixel-wise labels (N, H, W) → image-level labels (N,)
    using majority vote.
    """
    if not torch.is_tensor(pixel_labels):
        raise TypeError("pixel_to_image_labels expects a torch.Tensor")

    img_labels = []

    for y in pixel_labels:
        y = y.reshape(-1)

        if ignore_index is not None:
            y = y[y != ignore_index]

        if y.numel() == 0:
            img_labels.append(torch.tensor(ignore_index, device=y.device))
        else:
            img_labels.append(torch.mode(y).values)

    return torch.stack(img_labels)



def evaluate_classification(
    representations,
    labels,
    task_type="single",     # "single" or "multi"
    method="linear",        # "linear", "mlp", "knn", "kmeans"
    num_classes=None,
    k=20,
    epochs=100
):
    """
    Frozen representation evaluation following CROMA protocol.
    """

    X = representations.cpu().numpy()

    # Pixel → image labels if needed
    if labels.ndim > 1:
        y = pixel_to_image_labels(labels).cpu().numpy()
    else:
        y = labels.cpu().numpy()

    # ---------------- Linear probe ----------------
    if method == "linear":
        clf = nn.Linear(X.shape[1], num_classes).to(DEVICE)
        optimizer = torch.optim.Adam(clf.parameters(), lr=1e-3)

        X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        y_t = torch.tensor(y, dtype=torch.long).to(DEVICE)

        for _ in range(epochs):
            optimizer.zero_grad()
            logits = clf(X_t)

            if task_type == "multi":
                loss = F.binary_cross_entropy_with_logits(
                    logits, y_t.float()
                )
            else:
                loss = F.cross_entropy(logits, y_t)

            loss.backward()
            optimizer.step()

        with torch.no_grad():
            logits = clf(X_t).cpu()

        if task_type == "multi":
            return average_precision_score(
                y, logits.sigmoid().numpy(), average="macro"
            )
        else:
            preds = logits.argmax(1).numpy()
            return accuracy_score(y, preds)

    # ---------------- MLP probe ----------------
    if method == "mlp":
        clf = nn.Sequential(
            nn.Linear(X.shape[1], 2048),
            nn.ReLU(),
            nn.Linear(2048, num_classes)
        ).to(DEVICE)

        optimizer = torch.optim.Adam(clf.parameters(), lr=1e-3)

        X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        y_t = torch.tensor(y, dtype=torch.long).to(DEVICE)

        for _ in range(epochs):
            optimizer.zero_grad()
            logits = clf(X_t)

            if task_type == "multi":
                loss = F.binary_cross_entropy_with_logits(
                    logits, y_t.float()
                )
            else:
                loss = F.cross_entropy(logits, y_t)

            loss.backward()
            optimizer.step()

        with torch.no_grad():
            logits = clf(X_t).cpu()

        if task_type == "multi":
            return average_precision_score(
                y, logits.sigmoid().numpy(), average="macro"
            )
        else:
            preds = logits.argmax(1).numpy()
            return accuracy_score(y, preds)

    # ---------------- kNN ----------------
    if method == "knn":
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X, y)
        preds = knn.predict(X)
        return accuracy_score(y, preds)

    # ---------------- KMeans (Hungarian matched) ----------------
    if method == "kmeans":
        kmeans = KMeans(n_clusters=num_classes, n_init=10)
        clusters = kmeans.fit_predict(X)

        conf = np.zeros((num_classes, num_classes), dtype=np.int64)
        for c, t in zip(clusters, y):
            conf[c, t] += 1

        row_ind, col_ind = linear_sum_assignment(-conf)
        acc = conf[row_ind, col_ind].sum() / len(y)
        return acc

    raise ValueError(f"Unknown method: {method}")

def compute_iou(pred, target, num_classes):
    ious = []

    for cls in range(num_classes):
        pred_i = pred == cls
        tgt_i = target == cls

        inter = (pred_i & tgt_i).sum()
        union = (pred_i | tgt_i).sum()

        if union == 0:
            ious.append(np.nan)
        else:
            ious.append(inter / union)

    return np.nanmean(ious)

@torch.no_grad()
def evaluate_segmentation_model(model, dataloader, num_classes):
    """
    Linear probe on frozen patch encodings (DFC2020 protocol)
    """
    model.eval()
    miou_scores = []

    for x, y in tqdm(dataloader, desc="Evaluating segmentation"):
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        logits = model(x)        
        preds = logits.argmax(1)

        for p, t in zip(preds, y):
            miou_scores.append(
                compute_iou(
                    p.cpu().numpy(),
                    t.cpu().numpy(),
                    num_classes
                )
            )

    return float(np.mean(miou_scores))
@torch.no_grad()
def compute_umap_embeddings(
    representations,
    labels,
    n_neighbors=15,
    min_dist=0.1
):

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=42
    )

    emb_2d = reducer.fit_transform(
        representations.cpu().numpy()
    )

    return emb_2d, labels.cpu().numpy()

@torch.no_grad()
def evaluate_segmentation_probe(
    backbone,
    probe,
    dataloader,
    num_classes
):
    backbone.eval()
    probe.eval()

    miou_scores = []

    for x, y in tqdm(dataloader, desc="Evaluating segmentation"):
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        feats = backbone.encode_patches(x)
        logits = probe(feats)
        logits = F.interpolate(
            logits,
            size=y.shape[-2:],     
            mode="bilinear",
            align_corners=False
        )
        preds = logits.argmax(1)

        for p, t in zip(preds, y):
            miou_scores.append(
                compute_iou(
                    p.cpu().numpy(),
                    t.cpu().numpy(),
                    num_classes
                )
            )

    return float(np.mean(miou_scores))
