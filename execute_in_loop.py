import subprocess
import time
#command = ["python3", "onnx_sender.py", "--config", "config/bair-256.yaml", "--checkpoint", "checkpoints/bair-cpk.pth.tar", "--driving_video", "h264_long.mp4", "--cpu"]
command = ["python3", "onnx_sender_new.py", "--config", "config/bair-256.yaml", "--driving_video", "h264_long.mp4", "--checkpoint", "fp32_kpd.onnx", "--run_time", "gpu"]
#command = ["python3", "onnx_sender_new.py", "--config", "config/bair-256.yaml", "--checkpoint", "int8_kpd.onnx", "--driving_video", "h264_long.mp4", "--run_time", "gpu"]
#command = ["python3", "onnx_sender_new.py", "--config", "config/bair-256.yaml", "--checkpoint", "fp32_kpd.onnx", "--driving_video", "h264_long.mp4", "--run_time", "cpu"]
#command = ["python3", "sender.py", "--config", "config/bair-256.yaml", "--checkpoint", "checkpoints/bair-cpk.pth.tar", "--driving_video", "h264_long.mp4", "--cpu"]
#command = ["python3", "onnx_sender_cpu.py", "--config", "config/bair-256.yaml", "--checkpoint", "checkpoints/bair-cpk.pth.tar", "--driving_video", "h264_long.mp4", "--cpu"]
#command = ["python3", "sender.py", "--config", "config/bair-256.yaml", "--checkpoint", "checkpoints/bair-cpk.pth.tar", "--driving_video", "h264_long.mp4", "--cpu"]
#command = ["python3", "sender.py", "--config", "config/bair-256.yaml", "--checkpoint", "checkpoints/bair-cpk.pth.tar", "--driving_video", "h264_long.mp4"]
#command_2 =["ffmpeg","-hide_banner", "-loglevel", "error", "-i", "h264_long.mp4", "-c:v", "h264", "-preset", "medium", "-bf", "0", "out_x10_0000000.mp4"]
#command_3 =["ffmpeg","-hide_banner", "-loglevel", "error", "-i", "working/driving/x10_000000.mp4", "-c:v", "hevc", "-preset", "medium", "-x265-params", "bframes=0", "out_x10_0000000.mp4"]
#command = ["python3", "prune_resnet18_cifar10.py", "--mode", "test", "--round"]                                                          
#command =["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", "h264_long.mp4", "-c:v", "hevc", "-preset", "medium","-crf", "28", "-x265-params", "bframes=0", "out_x10_0000000.mp4"]
log_file_name = "ite.csv"

start_round = 0
end_round = 10
sleep_time = 30

def write_log_line(file_name, log_line):
    with open(file_name, 'a') as f:
        f.write(log_line + '\n')

log_line = "ite,type,start,end"
write_log_line(log_file_name, log_line)
for cur_round in range(start_round, end_round):
       # start = (time.time())
       # subprocess.run(command)
       # end = (time.time())
       # log_line = "{}, {}, {}, {}".format(cur_round, i, start, end)
       # write_log_line(log_file_name, log_line)
       # 
       # new_cmd = command_2[:-1]
       # new_cmd.append(str(i)+command_2[-1])
       # start = (time.time())
       # subprocess.run(new_cmd)
       # end = (time.time())
       # log_line = "{}, {}, {}, {}".format(cur_round, i, start, end)
       # write_log_line(log_file_name, log_line)

       # new_cmd = command_3[:-1]
       # new_cmd.append(str(i)+'o'+command_3[-1])
       # start = (time.time())
       # subprocess.run(new_cmd)
       # end = (time.time())
       # log_line = "{}, {}, {}, {}".format(cur_round, i, start, end)
       # write_log_line(log_file_name, log_line)

        #new_cmd = command_3[:-1]
        #new_cmd.append(str(i)+'o'+command_3[-1])

        # fom conversion
    start = (time.time())
    subprocess.run(command)
    end = (time.time())
    log_line = "{},{},{},{}".format(cur_round,"cmd", start, end)
    write_log_line(log_file_name, log_line)
        #idle

    start = (time.time())
    time.sleep(sleep_time)
    end = (time.time())
    log_line = "{},{},{},{}".format(cur_round,"idlw", start, end)
    write_log_line(log_file_name, log_line)
