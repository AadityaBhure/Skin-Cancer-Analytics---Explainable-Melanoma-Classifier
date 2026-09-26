"""Small local desktop UI; launch with Start Model.cmd."""
import json
import os
from pathlib import Path
import queue
import shutil
import threading

ROOT = Path(__file__).resolve().parents[1]
# The bundled Windows Python ships Tk DLLs but may omit their support scripts.
for variable, folder in [("TCL_LIBRARY", "tcl8.6"), ("TK_LIBRARY", "tk8.6")]:
    support = ROOT / ".venv/tcl" / folder
    if support.is_dir():
        os.environ[variable] = str(support)
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

class ModelUI:
    def __init__(self, window):
        self.window = window
        self.events = queue.Queue()
        self.model = None
        self.names = None
        self.busy = False
        self.photos = {}
        window.title("Skin lesion model — educational prototype")
        window.geometry("1040x760")
        window.minsize(800, 620)
        outer = ttk.Frame(window, padding=16)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="Skin lesion classifier", font=("Segoe UI", 19, "bold")).pack(anchor="w")
        ttk.Label(outer, text="Educational prototype. Not a diagnosis. Non-melanoma does not mean benign.").pack(anchor="w", pady=(4, 12))
        toolbar = ttk.Frame(outer)
        toolbar.pack(fill="x")
        self.upload = ttk.Button(toolbar, text="Choose image and predict", command=self.choose_image)
        self.upload.pack(side="left")
        self.status = tk.StringVar(value="Choose a dermoscopic JPG or PNG image. The model loads on first use.")
        ttk.Label(outer, textvariable=self.status, wraplength=960).pack(anchor="w", pady=8)
        self.tabs = ttk.Notebook(outer)
        self.tabs.pack(fill="both", expand=True)
        self.prediction_tab = ttk.Frame(self.tabs, padding=12)
        self.tabs.add(self.prediction_tab, text="Image prediction")
        self.result = tk.StringVar(value="No image selected")
        ttk.Label(self.prediction_tab, textvariable=self.result, font=("Segoe UI", 15, "bold")).pack(anchor="w", pady=8)
        self.score = tk.StringVar(value="Melanoma and non-melanoma scores will appear here.")
        ttk.Label(self.prediction_tab, textvariable=self.score).pack(anchor="w")
        image_row = ttk.Frame(self.prediction_tab)
        image_row.pack(fill="both", expand=True, pady=12)
        self.image_labels = []
        for title in ["Original (resized)", "Grad-CAM overlay — melanoma score"]:
            frame = ttk.LabelFrame(image_row, text=title, padding=8)
            frame.pack(side="left", fill="both", expand=True, padx=4)
            label = ttk.Label(frame, text="No image", anchor="center")
            label.pack(fill="both", expand=True)
            self.image_labels.append(label)
        ttk.Label(self.prediction_tab, text="Scores are model outputs, not calibrated medical risk. Heatmaps are not lesion boundaries.", wraplength=920).pack(anchor="w")
        self.metrics_tab = ttk.Frame(self.tabs, padding=16)
        self.tabs.add(self.metrics_tab, text="Test metrics")
        self.show_metrics()
        self.figure_tabs = {}
        for number, title, filename, note in [
            (4, "Training history", "training_curves.png", "Training versus validation accuracy and loss across completed epochs."),
            (5, "Confusion matrix", "confusion_matrix.png", "Two model classes: melanoma and non-melanoma. A seven-class matrix requires a separately trained seven-class model."),
            (6, "Explainability", "gradcam_example.png", "Initially shows the saved example. Choosing an image updates this comparison with its prediction and Grad-CAM."),
        ]:
            tab = ttk.Frame(self.tabs, padding=10)
            self.tabs.add(tab, text=f"Figure {number}")
            ttk.Label(tab, text=f"Figure {number}. {title}", font=("Segoe UI", 14, "bold")).pack(anchor="w")
            ttk.Label(tab, text=note, wraplength=920).pack(anchor="w", pady=6)
            path = ROOT / "outputs/figures" / filename
            ttk.Button(tab, text="Save figure as…", command=lambda p=path: self.save_figure(p)).pack(anchor="w")
            label = ttk.Label(tab, anchor="center")
            label.pack(fill="both", expand=True, pady=8)
            self.figure_tabs[number] = (label, path)
            self.show_figure(number)
        self.window.after(100, self.poll)

    def show_metrics(self):
        ttk.Label(self.metrics_tab, text="Held-out test-set performance", font=("Segoe UI", 15, "bold")).pack(anchor="w", pady=8)
        ttk.Label(self.metrics_tab, text="These metrics describe the saved model's test set, not the uploaded image.", wraplength=900).pack(anchor="w")
        try:
            metrics = json.loads((ROOT / "outputs/metrics/test_metrics.json").read_text(encoding="utf-8"))
            for label, key in [("Accuracy", "accuracy"), ("Melanoma precision", "precision"), ("Melanoma recall", "recall"), ("Melanoma F1", "f1_score"), ("ROC-AUC", "roc_auc")]:
                value = metrics.get(key)
                formatted = "Unavailable" if value is None else (f"{value:.4f}" if key == "roc_auc" else f"{value:.2%}")
                ttk.Label(self.metrics_tab, text=f"{label}: {formatted}", font=("Segoe UI", 12)).pack(anchor="w", pady=6)
            ttk.Label(self.metrics_tab, text=f"Test images: {metrics['test_images']}  |  Threshold: {metrics['threshold']:.2f}").pack(anchor="w", pady=10)
        except (OSError, ValueError, KeyError) as error:
            ttk.Label(self.metrics_tab, text=f"Test metrics unavailable. Run evaluate.py.\n{error}", wraplength=900).pack(anchor="w")
        ttk.Label(self.metrics_tab, text="The balanced image-level split may share lesions across sets. These results are not lesion-independent clinical validation.", wraplength=900).pack(anchor="w", pady=12)

    def show_figure(self, number):
        label, path = self.figure_tabs[number]
        try:
            with Image.open(path) as source:
                picture = source.copy()
            picture.thumbnail((740, 400), Image.Resampling.LANCZOS)
            self.photos[f"figure{number}"] = ImageTk.PhotoImage(picture)
            label.configure(image=self.photos[f"figure{number}"], text="")
        except OSError:
            label.configure(image="", text="Figure not generated yet.")

    def save_figure(self, path):
        if not path.is_file():
            messagebox.showinfo("Figure unavailable", "Generate this figure first.", parent=self.window)
            return
        destination = filedialog.asksaveasfilename(parent=self.window, initialfile=path.name, defaultextension=".png", filetypes=[("PNG image", "*.png")])
        if destination:
            try:
                if Path(destination).resolve() != path.resolve():
                    shutil.copyfile(path, destination)
            except OSError as error:
                messagebox.showerror("Could not save", str(error), parent=self.window)

    def choose_image(self):
        path = filedialog.askopenfilename(parent=self.window, title="Choose a dermoscopic image", filetypes=[("Images", "*.jpg *.jpeg *.png"), ("All files", "*.*")])
        if path:
            self.start_prediction(path)

    def start_prediction(self, path):
        if self.busy:
            return
        self.busy = True
        self.upload.configure(state="disabled")
        self.tabs.select(self.prediction_tab)
        self.result.set("Processing…")
        self.score.set("")
        for label in self.image_labels:
            label.configure(image="", text="Processing…")
        self.status.set(f"Loading model and analysing {Path(path).name}… Please wait.")
        threading.Thread(target=self.worker, args=(path,), daemon=True).start()

    def worker(self, path):
        try:
            # Heavy imports and inference stay off Tk's event loop.
            from predict import predict_image
            from train import load_model, load_class_names
            if self.model is None:
                self.model = load_model()
                self.names = load_class_names()
            result = predict_image(path, self.model, self.names)
            self.events.put(("result", (path, result)))
        except (Exception, SystemExit) as error:
            self.events.put(("error", str(error)))

    def poll(self):
        try:
            kind, payload = self.events.get_nowait()
        except queue.Empty:
            pass
        else:
            self.busy = False
            self.upload.configure(state="normal")
            if kind == "error":
                self.result.set("Could not analyse image")
                self.status.set("Choose another image or check the saved model.")
                for label in self.image_labels:
                    label.configure(image="", text="No result")
                messagebox.showerror("Prediction failed", payload, parent=self.window)
            else:
                path, result = payload
                self.result.set(f"Prediction: {result['prediction'].replace('_', ' ')}")
                p = result["probability"]
                self.score.set(f"Melanoma score: {p:.2%}    |    Non-melanoma score: {1-p:.2%}    |    Threshold: 50%")
                for index, key in enumerate(["original", "overlay"]):
                    photo = ImageTk.PhotoImage(Image.fromarray(result[key]).resize((300, 300)))
                    self.photos[key] = photo
                    self.image_labels[index].configure(image=photo, text="")
                self.show_figure(6)
                self.status.set(f"Finished: {Path(path).name}. Figure 6 is ready to save for your report.")
        self.window.after(100, self.poll)


def main():
    window = tk.Tk()
    ModelUI(window)
    window.mainloop()


if __name__ == "__main__":
    main()
