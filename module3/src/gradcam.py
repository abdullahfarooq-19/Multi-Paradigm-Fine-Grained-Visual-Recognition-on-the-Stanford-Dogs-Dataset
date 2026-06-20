"""
Module 3: Grad-CAM and Saliency Map Implementation
Explainability tools for CNN breed classification
"""

import torch
import torch.nn as nn
import numpy as np
import cv2


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.
    Visualizes which regions the CNN focuses on for predictions.
    """
    
    def __init__(self, model, target_layer):
        """
        Initialize Grad-CAM.
        
        Args:
            model: PyTorch model (ResNet50)
            target_layer: Layer to extract activations (e.g., model.layer4[-1])
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks on target layer."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate(self, input_tensor, class_idx=None):
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: Preprocessed image tensor [1, 3, 224, 224]
            class_idx: Target class (None = use predicted class)
        
        Returns:
            cam: Normalized CAM heatmap [H, W]
            class_idx: Predicted/target class index
        """
        self.model.eval()
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(1).item()
        
        # Backward pass
        self.model.zero_grad()
        output[0, class_idx].backward()
        
        # Compute CAM
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.squeeze().cpu().numpy(), class_idx


def compute_saliency(model, input_tensor):
    """
    Compute saliency map using input gradients.
    
    Args:
        model: PyTorch model
        input_tensor: Image tensor with requires_grad=True
    
    Returns:
        saliency: Normalized saliency map [H, W]
    """
    model.eval()
    input_tensor.requires_grad = True
    
    output = model(input_tensor)
    pred_class = output.argmax(1)
    output[0, pred_class].backward()
    
    # Take max absolute gradient across channels
    saliency = input_tensor.grad.abs().squeeze().max(dim=0)[0]
    saliency = saliency.cpu().numpy()
    
    # Normalize
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    
    return saliency


def overlay_heatmap(image, heatmap, alpha=0.5):
    """
    Overlay heatmap on original image.
    
    Args:
        image: Original image [H, W, 3] normalized to [0, 1]
        heatmap: Heatmap [H, W] normalized to [0, 1]
        alpha: Blending factor
    
    Returns:
        overlay: Blended image [H, W, 3]
    """
    # Resize heatmap to match image
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8), 
        cv2.COLORMAP_JET
    )
    heatmap_colored = heatmap_colored[:, :, ::-1] / 255.0  # BGR to RGB
    
    # Blend
    overlay = alpha * image + (1 - alpha) * heatmap_colored
    overlay = np.clip(overlay, 0, 1)
    
    return overlay
