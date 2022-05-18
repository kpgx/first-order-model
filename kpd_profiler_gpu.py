import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
import yaml
from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector

model_folder = "retrained_models/01/"
ckpt_path = model_folder+"checkpoint.pth.tar"
cfg_path = model_folder+"bair-256.yaml"


def load_checkpoints(checkpoint_path, config_path):
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda'))
    generator.load_state_dict(checkpoint['generator'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])
    generator.eval()
    kp_detector.eval()
    return generator, kp_detector


_, kpd = load_checkpoints(ckpt_path, cfg_path)


model = kpd.cuda()
inputs = torch.randn(1, 3, 256, 256).cuda()

with profile(on_trace_ready=torch.profiler.tensorboard_trace_handler("./log/kpd_gpu"), profile_memory=True, with_stack=True, record_shapes=True, use_cuda=True) as prof:
    with record_function("model_inference"):
        model(inputs)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

