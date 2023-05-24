import torch
import torch.nn as nn
import torchvision
import onnx
import onnxruntime
import time



model = torchvision.models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = nn.DataParallel(model)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

device = torch.device('cuda')
model = model.to(device)

model = model.module


# Provide an example input tensor
dummy_input = torch.randn(1, 3, 224, 224).to(device)

# Export the model to ONNX format
onnx_path = 'model.onnx'
torch.onnx.export(model, dummy_input, onnx_path, opset_version=11)

onnx_model = onnx.load(onnx_path)

ort_session = onnxruntime.InferenceSession(onnx_path)

# Generate a random input tensor for inference
input_data = torch.randn(1, 3, 224, 224)

# Run the inference and measure the inference time
start_time = time.time()
ort_inputs = {ort_session.get_inputs()[0].name: input_data.numpy()}
ort_outputs = ort_session.run(None, ort_inputs)
end_time = time.time()
inference_time = end_time - start_time

# Process the outputs if needed
output_tensor = ort_outputs[0]


print(f"Inference Time: {inference_time} seconds")