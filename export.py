import os
import argparse
import torch
from model import LDRN


def get_model_name(args):
    dataset = args.pretrained.lower()
    model_name = f"lap-depth-{dataset}"
    if args.use_kitti_grad:
        model_name += "-grad"
    return model_name


def get_model_dir(model_name):
    return model_name.replace("-", "_") + ".pkl"


def download_model(model_name, file_name):
    from huggingface_hub import hf_hub_download
    downloaded_model_path = hf_hub_download(repo_id=f"ibaiGorordo/{model_name}", filename=file_name)
    args.model_dir = downloaded_model_path


parser = argparse.ArgumentParser(description='Laplacian Depth Residual Network training on KITTI',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Directory setting
parser.add_argument('--input_shape', type=int, nargs='+', default=[352, 1216], help='image size')
parser.add_argument('--pretrained', type=str, default="KITTI", help='KITTI or NYU')
parser.add_argument('--use_kitti_grad', action='store_true')
parser.add_argument('--model_dir', type=str, default="", help='pretrained model directory')
args = parser.parse_args()

if args.pretrained == 'KITTI':
    args.max_depth = 80.0
    # assert (args.input_shape[0] == 352), "KITTI pretrained model only supports 352xN input size"
elif args.pretrained == 'NYU':
    args.max_depth = 10.0
    # assert (args.input_shape[0] == 432), "NYU pretrained model only supports 432xN input size"

model_name = get_model_name(args)

# Download model if not provided
if args.model_dir == "":
    args.model_dir = get_model_dir(model_name)
    download_model(model_name, args.model_dir)

print('=> loading model..')
Model = LDRN(max_depth=args.max_depth)
Model = torch.nn.DataParallel(Model)
Model.load_state_dict(torch.load(args.model_dir, map_location=torch.device('cpu')), strict=False)
Model.eval()

with torch.no_grad():
    img = torch.randn(1, 3, args.input_shape[0], args.input_shape[1])
    _, out = Model(img)

    # Convert to ONNX
    torch.onnx.export(Model.module,
                      img,
                      f'{model_name}.onnx',
                      verbose=True,
                      opset_version=11,
                      input_names=['input'],
                      output_names=['output'])

    # Simplify ONNX
    os.system(f'onnxsim {model_name}.onnx {model_name}.onnx')
