import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms, utils as vutils

import os
import random
import threading
import queue
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext

# Import FastGAN components
from FastGAN.models import weights_init, Discriminator, Generator
from FastGAN.operation import copy_G_params, load_params
from FastGAN.operation import ImageFolder, InfiniteSamplerWrapper
from FastGAN.diffaug import DiffAugment
import FastGAN.lpips as lpips

# ==========================================
# 1. Training Logic (Refactored)
# ==========================================


def crop_image_by_part(image, part):
    hw = image.shape[2] // 2
    if part == 0:
        return image[:, :, :hw, :hw]
    if part == 1:
        return image[:, :, :hw, hw:]
    if part == 2:
        return image[:, :, hw:, :hw]
    if part == 3:
        return image[:, :, hw:, hw:]


def make_train_d_func(percept):
    def train_d(net, data, label="real"):
        if label == "real":
            part = random.randint(0, 3)
            pred, [rec_all, rec_small, rec_part] = net(data, label, part=part)
            err = (
                F.relu(torch.rand_like(pred) * 0.2 + 0.8 - pred).mean()
                + percept(rec_all, F.interpolate(data, rec_all.shape[2])).sum()
                + percept(rec_small, F.interpolate(data, rec_small.shape[2])).sum()
                + percept(
                    rec_part,
                    F.interpolate(crop_image_by_part(data, part), rec_part.shape[2]),
                ).sum()
            )
            err.backward()
            return pred.mean().item(), rec_all, rec_small, rec_part
        else:
            pred = net(data, label)
            err = F.relu(torch.rand_like(pred) * 0.2 + 0.8 + pred).mean()
            err.backward()
            return pred.mean().item()

    return train_d


def train_gan_logic(params, log_queue):
    """
    This function contains the core training loop,
    now accepting a dictionary of parameters and a log queue.
    """

    def log(msg):
        log_queue.put(msg)

    def log_progress(val):
        log_queue.put(("PROGRESS", val))

    # Unpack params
    data_root = params["path"]
    total_iterations = params["iter"]
    batch_size = params["batch_size"]
    im_size = params["im_size"]
    dataloader_workers = params["workers"]
    save_interval = params["save_interval"]
    output_root = params["output_path"]
    exp_name = params["name"]
    device = torch.device(params["device"])
    checkpoint_path = params["ckpt"]
    start_iter = params["start_iter"]

    # Setup directories
    saved_model_folder = os.path.join(output_root, exp_name, "models")
    saved_image_folder = os.path.join(output_root, exp_name, "images")
    os.makedirs(saved_model_folder, exist_ok=True)
    os.makedirs(saved_image_folder, exist_ok=True)

    ndf, ngf, nz = 64, 64, 256
    nlr, nbeta1 = 0.0002, 0.5
    policy = "color,translation"

    percept = lpips.PerceptualLoss(
        model="net-lin", net="vgg", use_gpu="cuda" in params["device"]
    )
    train_d = make_train_d_func(percept)

    transform_list = [
        transforms.Resize((int(im_size), int(im_size))),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
    trans = transforms.Compose(transform_list)
    dataset = ImageFolder(root=data_root, transform=trans)

    dataloader = iter(
        DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=InfiniteSamplerWrapper(dataset),
            num_workers=dataloader_workers,
            pin_memory=True,
        )
    )

    netG = Generator(ngf=ngf, nz=nz, im_size=im_size)
    netD = Discriminator(ndf=ndf, im_size=im_size)
    netG.apply(weights_init)
    netD.apply(weights_init)
    netG.to(device)
    netD.to(device)

    avg_param_G = copy_G_params(netG)
    fixed_noise = torch.FloatTensor(8, nz).normal_(0, 1).to(device)

    optimizerG = optim.Adam(netG.parameters(), lr=nlr, betas=(nbeta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=nlr, betas=(nbeta1, 0.999))

    current_iteration = start_iter
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path)
        netG.load_state_dict(
            {k.replace("module.", ""): v for k, v in ckpt["g"].items()}
        )
        netD.load_state_dict(
            {k.replace("module.", ""): v for k, v in ckpt["d"].items()}
        )
        avg_param_G = ckpt["g_ema"]
        optimizerG.load_state_dict(ckpt["opt_g"])
        optimizerD.load_state_dict(ckpt["opt_d"])
        current_iteration = int(checkpoint_path.split("_")[-1].split(".")[0])
        log(f"Loaded checkpoint: {checkpoint_path}")

    for iteration in range(current_iteration, total_iterations + 1):
        # Update progress bar
        progress = (iteration / total_iterations) * 100
        log_progress(progress)

        real_image = next(dataloader)
        real_image = real_image.to(device)
        current_batch_size = real_image.size(0)
        noise = torch.Tensor(current_batch_size, nz).normal_(0, 1).to(device)

        fake_images = netG(noise)
        real_image = DiffAugment(real_image, policy=policy)
        fake_images = [DiffAugment(fake, policy=policy) for fake in fake_images]

        # 2. train Discriminator
        netD.zero_grad()
        err_dr, rec_img_all, rec_img_small, rec_img_part = train_d(
            netD, real_image, label="real"
        )
        train_d(netD, [fi.detach() for fi in fake_images], label="fake")
        optimizerD.step()

        # 3. train Generator
        netG.zero_grad()
        pred_g = netD(fake_images, "fake")
        err_g = -pred_g.mean()
        err_g.backward()
        optimizerG.step()

        for p, avg_p in zip(netG.parameters(), avg_param_G):
            avg_p.mul_(0.999).add_(0.001 * p.data)

        if iteration % 100 == 0:
            log(
                f"Iter {iteration} | Loss D: {err_dr:.4f} | Loss G: {-err_g.item():.4f}"
            )

        if iteration % save_interval == 0:
            backup_para = copy_G_params(netG)
            load_params(netG, avg_param_G)
            with torch.no_grad():
                vutils.save_image(
                    netG(fixed_noise)[0].add(1).mul(0.5),
                    os.path.join(saved_image_folder, f"{iteration}.jpg"),
                    nrow=4,
                )
                vutils.save_image(
                    torch.cat(
                        [
                            F.interpolate(real_image, 128),
                            rec_img_all,
                            rec_img_small,
                            rec_img_part,
                        ]
                    )
                    .add(1)
                    .mul(0.5),
                    os.path.join(saved_image_folder, f"rec_{iteration}.jpg"),
                )
            load_params(netG, backup_para)

            # Save checkpoint
            torch_backup = copy_G_params(netG)
            load_params(netG, avg_param_G)
            torch.save(
                {"g": netG.state_dict(), "d": netD.state_dict()},
                os.path.join(saved_model_folder, f"{iteration}.pth"),
            )
            torch.save(
                {
                    "g": netG.state_dict(),
                    "d": netD.state_dict(),
                    "g_ema": avg_param_G,
                    "opt_g": optimizerG.state_dict(),
                    "opt_d": optimizerD.state_dict(),
                },
                os.path.join(saved_model_folder, f"all_{iteration}.pth"),
            )
            load_params(netG, torch_backup)

    log_progress(100)
    return "Training Complete!"


# ==========================================
# 2. GUI Interface
# ==========================================


class TrainingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FastGAN GUI Trainer")
        self.root.geometry("550x650")
        self.log_queue = queue.Queue()

        # --- Hyperparameters Frame ---
        hp_frame = tk.LabelFrame(root, text="Hyperparameters", padx=10, pady=5)
        hp_frame.pack(fill="x", padx=15, pady=5)

        # Row 1
        tk.Label(hp_frame, text="Iter:").grid(row=0, column=0, sticky="w")
        self.ent_iter = self._create_entry(hp_frame, "50000", 0, 1)
        tk.Label(hp_frame, text="Start:").grid(row=0, column=2, sticky="w")
        self.ent_start = self._create_entry(hp_frame, "0", 0, 3)

        # Row 2
        tk.Label(hp_frame, text="Batch:").grid(row=1, column=0, sticky="w")
        self.ent_batch = self._create_entry(hp_frame, "8", 1, 1)
        tk.Label(hp_frame, text="Img Size:").grid(row=1, column=2, sticky="w")
        self.ent_size = self._create_entry(hp_frame, "1024", 1, 3)

        # Row 3
        tk.Label(hp_frame, text="Save Int:").grid(row=2, column=0, sticky="w")
        self.ent_save = self._create_entry(hp_frame, "1000", 2, 1)
        tk.Label(hp_frame, text="Workers:").grid(row=2, column=2, sticky="w")
        self.ent_workers = self._create_entry(hp_frame, "2", 2, 3)

        # Row 4
        tk.Label(hp_frame, text="Device:").grid(row=3, column=0, sticky="w")
        self.cb_device = ttk.Combobox(
            hp_frame, values=["cuda", "cpu"], state="readonly", width=10
        )
        self.cb_device.set("cuda")
        self.cb_device.grid(row=3, column=1, padx=5, pady=2)

        # --- Data/Output Frame ---
        io_frame = tk.LabelFrame(root, text="File Settings", padx=10, pady=5)
        io_frame.pack(fill="x", padx=15, pady=5)

        self.path_var = tk.StringVar()
        tk.Label(io_frame, text="Dataset:").grid(row=0, column=0, sticky="w")
        tk.Entry(io_frame, textvariable=self.path_var, width=30).grid(
            row=0, column=1, columnspan=2
        )
        tk.Button(
            io_frame, text="Browse", command=lambda: self.browse_folder(self.path_var)
        ).grid(row=0, column=3)

        self.out_var = tk.StringVar(value="output")
        tk.Label(io_frame, text="Output:").grid(row=1, column=0, sticky="w")
        tk.Entry(io_frame, textvariable=self.out_var, width=30).grid(
            row=1, column=1, columnspan=2
        )
        tk.Button(
            io_frame, text="Browse", command=lambda: self.browse_folder(self.out_var)
        ).grid(row=1, column=3)

        self.name_var = tk.StringVar(value="experiment_1")
        tk.Label(io_frame, text="Exp Name:").grid(row=2, column=0, sticky="w")
        tk.Entry(io_frame, textvariable=self.name_var, width=30).grid(
            row=2, column=1, columnspan=2
        )

        self.ckpt_var = tk.StringVar()
        tk.Label(io_frame, text="Checkpoint:").grid(row=3, column=0, sticky="w")
        tk.Entry(io_frame, textvariable=self.ckpt_var, width=30).grid(
            row=3, column=1, columnspan=2
        )
        tk.Button(io_frame, text="Browse", command=self.browse_file).grid(
            row=3, column=3
        )

        # --- Progress and Log ---
        prog_frame = tk.Frame(root, padx=15, pady=5)
        prog_frame.pack(fill="x")
        self.progress_bar = ttk.Progressbar(
            prog_frame, orient="horizontal", mode="determinate"
        )
        self.progress_bar.pack(fill="x")

        self.log_widget = scrolledtext.ScrolledText(
            root, height=10, state="disabled", font=("Consolas", 9), bg="#f0f0f0"
        )
        self.log_widget.pack(fill="both", expand=True, padx=15, pady=5)

        # --- Action Button ---
        self.btn_train = tk.Button(
            root,
            text="🚀 START TRAINING",
            bg="#81d995",
            fg="black",
            font=("Arial", 16, "bold"),
            command=self.start_training_thread,
        )
        self.btn_train.pack(pady=30, fill="x", padx=15)

        self.root.after(100, self.process_queue)

    def _create_entry(self, parent, default, row, col):
        ent = tk.Entry(parent, width=10)
        ent.insert(0, default)
        ent.grid(row=row, column=col, padx=5, pady=2)
        return ent

    def browse_folder(self, var):
        folder = filedialog.askdirectory()
        if folder:
            var.set(folder)

    def browse_file(self):
        file = filedialog.askopenfilename()
        if file:
            self.ckpt_var.set(file)

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
        params = {
            "path": self.path_var.get(),
            "iter": int(self.ent_iter.get()),
            "batch_size": int(self.ent_batch.get()),
            "im_size": int(self.ent_size.get()),
            "workers": int(self.ent_workers.get()),
            "save_interval": int(self.ent_save.get()),
            "output_path": self.out_var.get(),
            "name": self.name_var.get(),
            "device": self.cb_device.get(),
            "ckpt": self.ckpt_var.get(),
            "start_iter": int(self.ent_start.get()),
        }

        if not params["path"]:
            messagebox.showerror("Error", "Dataset path is required!")
            return

        self.btn_train.config(state="disabled", text="RUNNING...")
        self.log_widget.config(state="normal")
        self.log_widget.delete("1.0", tk.END)
        self.log_widget.config(state="disabled")

        t = threading.Thread(target=self.run_worker, args=(params,), daemon=True)
        t.start()

    def run_worker(self, params):
        try:
            res = train_gan_logic(params, self.log_queue)
            self.log_queue.put(res)
        except Exception as e:
            self.log_queue.put(f"CRITICAL ERROR: {e}")
        finally:
            self.btn_train.config(state="normal", text="🚀 START TRAINING")


if __name__ == "__main__":
    root = tk.Tk()
    app = TrainingApp(root)
    root.mainloop()
