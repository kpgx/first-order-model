import onnx
import torch
import yaml

from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector


def load_and_check(model_path):
    # Load the ONNX model
    model = onnx.load(model_path)

    # Check that the IR is well formed
    onnx.checker.check_model(model)

    # Print a human readable representation of the graph
    onnx.helper.printable_graph(model.graph)


def load_checkpoints(checkpoint_path, config_path):
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    generator.load_state_dict(checkpoint['generator'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])
    generator.eval()
    kp_detector.eval()
    return generator, kp_detector


def export_models(config_path, checkpoint_path):
    generator, kp_detector = load_checkpoints(checkpoint_path, config_path)

    # First export the kp detector
    dummy_source_image = torch.randn(1, 3, 256, 256, device='cpu')
    torch.onnx.export(kp_detector, (dummy_source_image,),
                      'first-order-model-kp_detector.onnx',
                      export_params=True, input_names=['input'], output_names=['output'], opset_version=11, verbose=True)

    load_and_check('first-order-model-kp_detector.onnx')

    kp_driving = kp_source = {
        'value': torch.randn([1, 10, 2], device='cpu'),
        'jacobian': torch.randn(([1, 10, 2, 2]), device='cpu')
    }

    # This will fail because this model is not traceable and grid_sample is not supported
    # Even in the opset version 12 (the latest).
    #torch.onnx.export(generator, (dummy_source_image, kp_driving, kp_source),
    #                  'first-order-model-generator.onnx', export_params=True,
    #                  input_names=['input'], output_names=['output'], verbose=True)

    #load_and_check('first-order-model-generator.onnx')

if __name__ == "__main__":
    config_file = 'config/bair-256.yaml'
    checkpoint_file = 'checkpoints/bair-cpk.pth.tar'
    try:
#        export_models('config/vox-adv-256.yaml',
#                      '/Users/username/Download/vox-adv-cpk.pth.tar')
        export_models(config_file,
                      checkpoint_file)
    except Exception as e:
        raise (e)
