import torch
from torchaudio.io import StreamReader
from torchvision.transforms import v2 as transforms
import time
print(torch._dynamo.list_backends())

device = torch.device("cuda")
src = "./video2_2.mp4"

def yuv_to_rgb(frames):
    frames = frames.to(torch.float)
    y = frames[..., 0, :, :]
    u = frames[..., 1, :, :]
    v = frames[..., 2, :, :]

    y = y / 255.0
    u = u / 255.0 - 0.5
    v = v / 255.0 - 0.5

    r = y + 1.402 * v
    g = y - 0.344136 * u - 0.714136 * v
    b = y + 1.772 * u

    rgb = torch.stack([r, g, b], -1)
    rgb = rgb.clamp(0, 1)
    return rgb
torch.backends.cudnn.benchmark = True
model = torch.jit.load("giddy_music.pt") #torch.jit.load("trt_model.ep")
model = model.to(memory_format=torch.channels_last)
model = torch.compile(model, mode="max-autotune")

def test_hw_decode_and_resize(src, decoder, decoder_option, hw_accel="cuda", frames_per_chunk=5):
    s = StreamReader(src)
    s.add_video_stream(15, decoder=decoder, decoder_option=decoder_option, hw_accel=hw_accel)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)
    std = torch.tensor([0.229, 0.224, 0.225], device=device)
    frameTransforms = transforms.Compose([
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=mean, std=std),
    ])
    num_frames = 0
    chunk = None
    t0 = time.monotonic()
    i = 0
    for (chunk,) in s.stream():
        chunk = chunk[0].unsqueeze(0)
        i += chunk.shape[0]
        num_frames += chunk.shape[0]
        # Convert to RGB
        chunk = yuv_to_rgb(chunk)
        chunk = chunk.permute(0, 3, 1, 2)
        frame = chunk[0]
        # Normalize
        frame = frameTransforms(frame)

    elapsed = time.monotonic() - t0
    fps = num_frames / elapsed
    print(f" - Shape: {chunk.shape}, torch dtype: {chunk.dtype}")
    print(f" - Processed {num_frames} frames in {elapsed:.2f} seconds. ({fps:.2f} fps)")
    return fps

import decord
decord.bridge.set_bridge('torch')
def preprocess_video_cpu(video_path, frame_interval=15, img_size=512, use_half_precision=True):
    global globalframe1
    global globalframepre1
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0), width=img_size, height=img_size)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device)
    std = torch.tensor([0.229, 0.224, 0.225], device=device)
    #fps = custom_round(vr.get_avg_fps())
    #frame_interval = int(fps * frame_interval)
    frameTransforms = transforms.Compose([
        transforms.ToDtype(torch.float16, scale=True),
        transforms.Normalize(mean=mean, std=std),
    ])
    dummyframe = frameTransforms(vr[0].to(device).permute(2, 0, 1)).unsqueeze(0)
    batch = dummyframe.repeat(16, 1, 1, 1)
    with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True):
                model(batch)
                model(batch)
                model(batch)
                model(batch)
    t0 = time.monotonic()
    num_frames = 0
    batch_size = 16
    frames = torch.empty((batch_size, 3, img_size, img_size), device=device)
    j = 0

    for i in range(0, len(vr), frame_interval):
        # the video reader will handle seeking and skipping in the most efficient manner
        frame = vr[i].to(device)
        frame = frame.permute(2, 0, 1)
        frame = frameTransforms(frame)
        frames[j % batch_size] = frame

        if (j + 1) % batch_size == 0:
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=True):
                    model(frames)
            num_frames += batch_size
        j += 1

    elapsed = time.monotonic() - t0
    fps = num_frames / elapsed
    print(f" - Shape: {frame.shape}, torch dtype: {frame.dtype}")
    print(f" - Processed {num_frames} frames in {elapsed:.2f} seconds. ({fps:.2f} fps)")
#test_hw_decode_and_resize(src, "h264_cuvid", decoder_option={"resize": "512x512"}, hw_accel="cuda")
preprocess_video_cpu(src, frame_interval=15, img_size=512, use_half_precision=False)
preprocess_video_cpu(src, frame_interval=15, img_size=512, use_half_precision=False)

def extract_frames_at_intervals(src, decoder, decoder_option, hw_accel="cuda", interval=0.5):
    s = StreamReader(src)
    s.add_video_stream(1, decoder=decoder, decoder_option=decoder_option, hw_accel=hw_accel, filter_desc="framestep=14")
    
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)
    std = torch.tensor([0.229, 0.224, 0.225], device=device)
    frameTransforms = transforms.Compose([
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=mean, std=std),
    ])
    print("debug")
    num_frames = 0
    chunk = None
    t0 = time.monotonic()
    frames = []
    
    duration = 571
    timestamps = torch.arange(0, duration, interval)
    
    for timestamp in timestamps:
        s.seek(timestamp.item())
        for (chunk,) in s.stream():
            chunk = yuv_to_rgb(chunk)
            chunk = chunk.permute(0, 3, 1, 2)
            frame = chunk[0]
            frame = frameTransforms(frame)
            frames.append(frame)
            break  # We only need the first frame after seeking

    elapsed = time.monotonic() - t0
    fps = len(frames) / elapsed
    print(f" - Extracted {len(frames)} frames in {elapsed:.2f} seconds. ({fps:.2f} fps)")
    return frames
#extract_frames_at_intervals(src, "hevc_cuvid", decoder_option={"resize": "512x512"}, hw_accel="cuda", interval=0.5)