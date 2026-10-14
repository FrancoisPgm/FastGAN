import os
import torch
import threading
import queue
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext
from PIL import Image
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms


# ==========================================
# 1. Custom Dataset Class (No Changes)
# ==========================================
class FolderDataset(Dataset):
    def __init__(self, data_dict, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(data_dict.keys())}
        for cls_name, folder_path in data_dict.items():
            label = self.class_to_idx[cls_name]
            if not os.path.exists(folder_path):
                continue
            for img_name in os.listdir(folder_path):
                if img_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    self.images.append(os.path.join(folder_path, img_name))
                    self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


# ==========================================
# 2. Model Initialization (No Changes)
# ==========================================
def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def get_resnet18(num_classes):
    model = models.resnet18(weights=None)
    model.apply(init_weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ==========================================
# 3. Training Logic (No Changes)
# ==========================================
def train_advanced_resnet(
    data_dict,
    epochs,
    batch_size,
    lr,
    warmup_epochs,
    output_folder,
    output_name,
    log_queue,
):
    def log(msg):
        log_queue.put(msg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Starting training on device: {device}")

    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    full_dataset = FolderDataset(data_dict, transform=train_transform)
    if len(full_dataset) == 0:
        return "Error: Dataset is empty."

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    model = get_resnet18(len(data_dict)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    warmup_sch = LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_sch = CosineAnnealingLR(optimizer, T_max=(epochs - warmup_epochs))
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup_sch, cosine_sch], milestones=[warmup_epochs]
    )

    for epoch in range(epochs):
        log_queue.put(("PROGRESS", (epoch / epochs) * 100))
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        log(
            f"Epoch [{epoch + 1}/{epochs}] - Loss: {running_loss / len(train_loader):.4f} - Val Acc: {100 * correct / total:.2f}%"
        )

    os.makedirs(output_folder, exist_ok=True)
    model_path = os.path.join(output_folder, output_name)
    torch.save(model.state_dict(), model_path)
    with open(os.path.join(output_folder, "labels.txt"), "w") as f:
        f.writelines(f"{idx}: {lbl}\n" for idx, lbl in enumerate(data_dict.keys()))

    log_queue.put(("PROGRESS", 100))
    return f"Success! Model saved to {model_path}"


# ==========================================
# 4. GUI Interface
# ==========================================


class LabelInputDialog(tk.Toplevel):
    def __init__(self, parent, title):
        super().__init__(parent)
        self.title(title)
        self.geometry("250x150")
        self.result = None
        tk.Label(self, text="Enter Label (e.g. 'cats'):").pack(pady=10)
        self.ent_label = tk.Entry(self)
        self.ent_label.pack(pady=5)
        self.ent_label.focus_set()
        btn_frame = tk.Frame(self)
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="OK", width=8, command=self.ok).pack(
            side="left", padx=5
        )
        tk.Button(btn_frame, text="Cancel", width=8, command=self.destroy).pack(
            side="left", padx=5
        )

    def ok(self):
        self.result = self.ent_label.get().strip()
        self.destroy()


class TrainingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Trainer")
        self.root.geometry("550x680")  # Fixed width/height to ensure button visibility
        self.data_folders = []
        self.log_queue = queue.Queue()

        # --- Hyperparameters (Compact 2x2 Grid) ---
        hyper_frame = tk.LabelFrame(root, text="Hyperparameters", padx=10, pady=5)
        hyper_frame.pack(fill="x", padx=15, pady=5)

        # Row 0
        tk.Label(hyper_frame, text="Epochs:").grid(row=0, column=0, sticky="e")
        self.ent_epochs = self._create_entry(hyper_frame, "50", 0, 1)
        tk.Label(hyper_frame, text="Batch:").grid(row=0, column=2, sticky="e")
        self.ent_batch = self._create_entry(hyper_frame, "32", 0, 3)

        # Row 1
        tk.Label(hyper_frame, text="LR:").grid(row=1, column=0, sticky="e")
        self.ent_lr = self._create_entry(hyper_frame, "0.001", 1, 1)
        tk.Label(hyper_frame, text="Warmup:").grid(row=1, column=2, sticky="e")
        self.ent_warmup = self._create_entry(hyper_frame, "5", 1, 3)

        # --- Data Selection (Fixed Height) ---
        data_frame = tk.LabelFrame(
            root, text="Dataset (Add folders & specify labels)", padx=10, pady=5
        )
        data_frame.pack(fill="x", padx=15, pady=5)

        self.canvas = tk.Canvas(data_frame, height=120)  # Reduced height
        self.scrollbar = ttk.Scrollbar(
            data_frame, orient="vertical", command=self.canvas.yview
        )
        self.scroll_frame = tk.Frame(self.canvas)
        self.scroll_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="x", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        tk.Button(data_frame, text="+ Add Folder", command=self.add_folder_dialog).pack(
            pady=2
        )

        # --- Output Settings ---
        out_frame = tk.LabelFrame(root, text="Output", padx=10, pady=5)
        out_frame.pack(fill="x", padx=15, pady=5)
        self.out_dir_var = tk.StringVar(value="output")
        tk.Button(
            out_frame,
            text="Select Output Folder",
            command=self.select_output_dir,
            font=("Arial", 8),
        ).pack(anchor="w")
        tk.Label(
            out_frame, textvariable=self.out_dir_var, fg="blue", font=("Arial", 8)
        ).pack(anchor="w")
        self.ent_model_name = tk.Entry(out_frame)
        self.ent_model_name.insert(0, "resnet18_model.pth")
        self.ent_model_name.pack(fill="x")

        # --- Progress and Logs ---
        self.progress_bar = ttk.Progressbar(
            root, orient="horizontal", mode="determinate"
        )
        self.progress_bar.pack(fill="x", padx=15, pady=2)

        self.log_widget = scrolledtext.ScrolledText(
            root, height=5, state="disabled", font=("Consolas", 9), bg="#f0f0f0"
        )
        self.log_widget.pack(fill="both", padx=15, pady=2)

        # --- Bottom Button ---
        self.btn_train = tk.Button(
            root,
            text="🚀 START TRAINING",
            bg="#28a745",
            fg="white",
            font=("Arial", 12, "bold"),
            command=self.start_training_thread,
        )
        self.btn_train.pack(pady=10, fill="x", padx=15)

        self.root.after(100, self.process_queue)

    def _create_entry(self, parent, default, row, col):
        ent = tk.Entry(parent, width=8)
        ent.insert(0, default)
        ent.grid(row=row, column=col, padx=5, pady=2)
        return ent

    def add_folder_dialog(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            dialog = LabelInputDialog(self.root, "Enter Label")
            self.root.wait_window(dialog)
            if dialog.result:
                self.register_folder_ui(dialog.result, folder_path)

    def register_folder_ui(self, label, path):
        row_frame = tk.Frame(self.scroll_frame)
        row_frame.pack(fill="x", pady=1)
        tk.Label(
            row_frame,
            text=f"[{label}]",
            width=10,
            anchor="w",
            font=("Arial", 8, "bold"),
        ).pack(side="left")
        tk.Label(row_frame, text=path, fg="gray", font=("Arial", 7)).pack(
            side="left", fill="x", expand=True
        )
        tk.Button(
            row_frame,
            text="X",
            fg="red",
            command=lambda: self.remove_folder(row_frame, label, path),
        ).pack(side="right")
        self.data_folders.append({"label": label, "path": path})

    def remove_folder(self, row_widget, label, path):
        row_widget.destroy()
        self.data_folders = [
            d
            for d in self.data_folders
            if not (d["label"] == label and d["path"] == path)
        ]

    def select_output_dir(self):
        folder = filedialog.askdirectory()
        if folder:
            self.out_dir_var.set(folder)

    def process_queue(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                if isinstance(msg, tuple) and msg[0] == "PROGRESS":
                    self.progress_bar["value"] = msg[1]
                else:
                    self.update_log(str(msg))
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.process_queue)

    def update_log(self, msg):
        self.log_widget.config(state="normal")
        self.log_widget.insert(tk.END, msg + "\n")
        self.log_widget.see(tk.END)
        self.log_widget.config(state="disabled")

    def start_training_thread(self):
        if len(self.data_folders) < 2:
            messagebox.showwarning("Error", "Select at least 2 folders.")
            return
        data_dict = {item["label"]: item["path"] for item in self.data_folders}
        try:
            args = (
                data_dict,
                int(self.ent_epochs.get()),
                int(self.ent_batch.get()),
                float(self.ent_lr.get()),
                int(self.ent_warmup.get()),
                self.out_dir_var.get(),
                self.ent_model_name.get(),
                self.log_queue,
            )
            self.btn_train.config(state="disabled", text="RUNNING...")
            self.log_widget.config(state="normal")
            self.log_widget.delete("1.0", tk.END)
            self.log_widget.config(state="disabled")
            threading.Thread(target=self.run_training, args=args, daemon=True).start()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def run_training(self, *args):
        try:
            res = train_advanced_resnet(*args)
            self.log_queue.put(res)
        except Exception as e:
            self.log_queue.put(f"ERROR: {e}")
        finally:
            self.btn_train.config(state="normal", text="🚀 START TRAINING")


if __name__ == "__main__":
    root = tk.Tk()
    app = TrainingApp(root)
    root.mainloop()
