import onnx
import onnxruntime
import torch
import yaml
import numpy as np

from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
    
root_dir = "pc_folder/checkpoints/bair_new_conv2/"
config_file = root_dir+"config.yaml"
checkpoint_file = root_dir+"checkpoint.pth.tar"
onnx_model_name = root_dir+"checkpoint.onnx"


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

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def export_models(config_path, checkpoint_path, out_file_name):
    _, kp_detector = load_checkpoints(checkpoint_path, config_path)
    #torch.save(kp_detector.state_dict(), "kp_detector_state_dict.pt")
    #torch.save(kp_detector, "kp_detector_complete_model.pt")

    # First export the kp detector
    dummy_source_image = torch.randn(1, 3, 256, 256, device='cpu')
    kp_detector.eval()
    torch_out = kp_detector(dummy_source_image)
    torch.onnx.export(kp_detector, (dummy_source_image,),
                      out_file_name,
                      export_params=True, input_names=['input'], output_names=['output'], opset_version=11, verbose=True)

    load_and_check(out_file_name)

    #compare outputs
    ort_session = onnxruntime.InferenceSession(out_file_name, providers=['CUDAExecutionProvider',])
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_source_image)}
    ort_outs = ort_session.run(None, ort_inputs)
    print("torch outs\n", torch_out)
    print("ort outs \n", ort_outs)

    #np.testing.assert_allclose(to_numpy(torch_out[0]), ort_outs[0][0], rtol=1e-03, atol=1e-05)
    #np.testing.assert_allclose(to_numpy(torch_out[1]), ort_outs[0][1], rtol=1e-03, atol=1e-05)


#    kp_driving = kp_source = {
#        'value': torch.randn([1, 10, 2], device='cpu'),
#        'jacobian': torch.randn(([1, 10, 2, 2]), device='cpu')
#    }
#
    # This will fail because this model is not traceable and grid_sample is not supported
    # Even in the opset version 12 (the latest).
    #torch.onnx.export(generator, (dummy_source_image, kp_driving, kp_source),
    #                  'first-order-model-generator.onnx', export_params=True,
    #                  input_names=['input'], output_names=['output'], verbose=True)

    #load_and_check('first-order-model-generator.onnx')

if __name__ == "__main__":
    try:
#        export_models('config/vox-adv-256.yaml',
#                      '/Users/username/Download/vox-adv-cpk.pth.tar')
        export_models(config_file,
                      checkpoint_file, onnx_model_name)
        print("model exported and checked")
    except Exception as e:
        raise (e)
