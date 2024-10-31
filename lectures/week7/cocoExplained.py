import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
# Kräver eventuellt pip install torch --user

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        def save_gradient(grad):
            self.gradients = grad
            
        def save_activation(module, input, output):
            self.activations = output
            
        # Attach hooks
        target_layer.register_forward_hook(save_activation)
        target_layer.register_backward_hook(save_gradient)
    
    def generate_cam(self, input_image, target_class):
        # Forward pass
        output = self.model(input_image)
        
        # Zero all gradients
        self.model.zero_grad()
        
        # Target for backprop
        one_hot = torch.zeros(output.size())
        one_hot[0][target_class] = 1
        
        # Backward pass
        output.backward(gradient=one_hot)
        
        # Generate CAM
        gradients = self.gradients.data.numpy()[0]
        activations = self.activations.data.numpy()[0]
        
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
            
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input_image.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        
        return cam
    
def resize_to_fit_screen(image, max_width=1280, max_height=720):
    """Resize image to fit screen while maintaining aspect ratio"""
    height, width = image.shape[:2]
    
    # Calculate scaling factor to fit screen
    scale_width = max_width / width
    scale_height = max_height / height
    scale = min(scale_width, scale_height)
    
    # If image is smaller than max size, keep original size
    if scale >= 1:
        return image
    
    # Calculate new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized

# Modified main detection code
def detect_and_explain(image_path, model, classes, confidence_threshold=0.7):
    # Original image loading and preprocessing
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    
    # Get output layer names
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

    # Get the name of a conv layer before YOLO layer
    conv_layer_name = 'conv_81'  # This is typically one of the last conv layers before YOLO

    # Get detections
    model.setInput(blob)
    outputs = model.forward([conv_layer_name] +output_layers)
    
    # Separate feature maps and detections
    feature_maps = outputs[0]  # First output is from conv layer
    detections = outputs[1:]   # Rest are YOLO detection layers

    # Process detections
    boxes = []
    confidences = []
    class_ids = []
    
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > confidence_threshold:
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                w = int(obj[2] * width)
                h = int(obj[3] * height)
                
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Apply NMS
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)

    
    try:
        # Process feature maps for visualization
        # Average across all channels
        feature_vis = np.mean(feature_maps, axis=1)[0]
        
        # Ensure we have a valid numpy array
        if not isinstance(feature_vis, np.ndarray):
            feature_vis = np.array(feature_vis)
        
        # Normalize to 0-255 range
        feature_vis = ((feature_vis - feature_vis.min()) * 255 / 
                      (feature_vis.max() - feature_vis.min() + 1e-8))
        feature_vis = feature_vis.astype(np.uint8)
        
        # Resize to match original image
        feature_vis = cv2.resize(feature_vis, (width, height))
        
        # Apply colormap
        feature_vis = cv2.applyColorMap(feature_vis, cv2.COLORMAP_JET)
        
        # Create visualization
        alpha = 0.4
        output_image = cv2.addWeighted(image, 1, feature_vis, alpha, 0)
    except Exception as e:
        print(f"Warning: Could not process feature maps: {e}")
        print("Falling back to original image")
        output_image = image.copy()

    # Draw boxes and labels
    if len(indices) > 0:  # Check if indices is not empty
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
            
            # Draw box
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw label
            cv2.putText(output_image, label, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return output_image

# Load model and classes
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Process image
image_path = "sample_image.jpg"
visualization = detect_and_explain(image_path, net, classes)

# Resize to fit screen
resized_viz = resize_to_fit_screen(visualization)

# Display results
cv2.imshow("YOLO Detections with Feature Visualization", resized_viz)
cv2.waitKey(0)
cv2.destroyAllWindows()