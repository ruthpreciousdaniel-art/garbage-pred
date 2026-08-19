import sys; sys.path.insert(0, ".")
from model import GarbageCNN
import torch
m = GarbageCNN(num_classes=10)
print("Model builds OK:", sum(p.numel() for p in m.parameters()), "params")

!cd /content/garbage_classifier_deploy && python app/check_model.py
