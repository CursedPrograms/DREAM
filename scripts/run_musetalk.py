#!/usr/bin/env python3

import os
import re
import sys
import copy
import glob
import pickle
import shutil
import asyncio
import subprocess
import argparse
from argparse import Namespace

import cv2
import numpy as np
import torch
import imageio
import gradio as gr
from tqdm import tqdm
from moviepy.editor import VideoFileClip, AudioFileClip
from transformers import WhisperModel

# ─── Resolve paths relative to this script's location ───────────────────────
ProjectDir      = os.path.abspath(os.path.dirname(__file__))
CheckpointsDir  = os.path.abspath(os.path.join(ProjectDir, "..", "models"))

UNET_MODEL_PATH = os.path.join(CheckpointsDir, "musetalkV15", "unet.pth")
UNET_CONFIG     = os.path.join(CheckpointsDir, "musetalkV15", "musetalk.json")
VAE_PATH        = os.path.join(CheckpointsDir, "sd-vae")
WHISPER_PATH    = os.path.join(CheckpointsDir, "whisper")

# ─── Model file check ────────────────────────────────────────────────────────
def download_model():
    required_models = {
        "MuseTalk UNet":  UNET_MODEL_PATH,
        "MuseTalk config": UNET_CONFIG,
        "SD VAE":         os.path.join(CheckpointsDir, "sd-vae",              "config.json"),
        "Whisper":        os.path.join(CheckpointsDir, "whisper",             "config.json"),
        "DWPose":         os.path.join(CheckpointsDir, "dwpose",              "dw-ll_ucoco_384.pth"),
        "SyncNet":        os.path.join(CheckpointsDir, "syncnet",             "latentsync_syncnet.pt"),
        "Face Parse":     os.path.join(CheckpointsDir, "face-parse-bisent",   "79999_iter.pth"),
        "ResNet":         os.path.join(CheckpointsDir, "face-parse-bisent",   "resnet18-5c106cde.pth"),
    }
    missing = [name for name, path in required_models.items() if not os.path.exists(path)]
    if missing:
        print("The following required model files are missing:")
        for m in missing:
            print(f"  - {m}")
        print("\nPlease run the download script:")
        print("  Windows : download_weights.bat")
        print("  Linux   : ./download_weights.sh")
        sys.exit(1)
    print("All required model files exist.")

download_model()

# ─── MuseTalk imports (after model check) ────────────────────────────────────
from musetalk.utils.blending      import get_image
from musetalk.utils.face_parsing  import FaceParsing
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.utils         import get_file_type, get_video_fps, datagen, load_all_model
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder, get_bbox_range

# ─── ffmpeg helper ───────────────────────────────────────────────────────────
def fast_check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except Exception:
        return False

# ─── Load models (once, at startup) ─────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vae, unet, pe = load_all_model(
    unet_model_path = UNET_MODEL_PATH,
    vae_type        = VAE_PATH,
    unet_config     = UNET_CONFIG,
    device          = device,
)

# ─── Parse CLI args ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--ffmpeg_path",  type=str,  default=r"ffmpeg-master-latest-win64-gpl-shared\bin")
parser.add_argument("--ip",           type=str,  default="127.0.0.1")
parser.add_argument("--port",         type=int,  default=7860)
parser.add_argument("--share",        action="store_true")
parser.add_argument("--use_float16",  action="store_true")
args = parser.parse_args()

# ─── Precision ───────────────────────────────────────────────────────────────
if args.use_float16:
    pe.half()
    vae.vae   = vae.vae.half()
    unet.model = unet.model.half()
    weight_dtype = torch.float16
else:
    weight_dtype = torch.float32

pe         = pe.to(device)
vae.vae    = vae.vae.to(device)
unet.model = unet.model.to(device)

timesteps = torch.tensor([0], device=device)

# ─── Audio / Whisper ─────────────────────────────────────────────────────────
audio_processor = AudioProcessor(feature_extractor_path=WHISPER_PATH)
whisper = WhisperModel.from_pretrained(WHISPER_PATH)
whisper = whisper.to(device=device, dtype=weight_dtype).eval()
whisper.requires_grad_(False)

# ─── ffmpeg PATH fix ─────────────────────────────────────────────────────────
if not fast_check_ffmpeg():
    sep = ";" if sys.platform == "win32" else ":"
    os.environ["PATH"] = f"{args.ffmpeg_path}{sep}{os.environ['PATH']}"
    if not fast_check_ffmpeg():
        print("Warning: ffmpeg not found — video export will fail.")

# ─── Video normaliser (convert to 25 fps on upload) ──────────────────────────
def check_video(video):
    if not isinstance(video, str):
        return video
    dir_path, file_name = os.path.split(video)
    if file_name.startswith("outputxxx_"):
        return video
    os.makedirs("./results/input", exist_ok=True)
    output_video = os.path.join("./results/input", "outputxxx_" + file_name)
    reader = imageio.get_reader(video)
    fps    = reader.get_meta_data()["fps"]
    frames = [im for im in reader]
    target_fps = 25
    L          = len(frames)
    L_target   = int(L / fps * target_fps)
    original_t = [x / fps for x in range(1, L + 1)]
    t_idx = 0
    target_frames = []
    for target_t in range(1, L_target + 1):
        while target_t / target_fps > original_t[t_idx]:
            t_idx += 1
            if t_idx >= L:
                break
        target_frames.append(frames[t_idx])
    imageio.mimwrite(output_video, target_frames, "FFMPEG",
                     fps=25, codec="libx264", quality=9, pixelformat="yuv420p")
    return output_video

# ─── Debug: test inpainting on first frame only ───────────────────────────────
@torch.no_grad()
def debug_inpainting(video_path, bbox_shift, extra_margin=10, parsing_mode="jaw",
                     left_cheek_width=90, right_cheek_width=90):
    result_dir = "./results/debug"
    os.makedirs(result_dir, exist_ok=True)

    # Read first frame
    if get_file_type(video_path) == "video":
        reader      = imageio.get_reader(video_path)
        first_frame = reader.get_data(0)
        reader.close()
    else:
        first_frame = cv2.cvtColor(cv2.imread(video_path), cv2.COLOR_BGR2RGB)

    debug_frame_path = os.path.join(result_dir, "debug_frame.png")
    cv2.imwrite(debug_frame_path, cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR))

    coord_list, frame_list = get_landmark_and_bbox([debug_frame_path], bbox_shift)
    bbox  = coord_list[0]
    frame = frame_list[0]

    if bbox == coord_placeholder:
        return None, "No face detected — try adjusting bbox_shift."

    fp = FaceParsing(left_cheek_width=left_cheek_width, right_cheek_width=right_cheek_width)

    x1, y1, x2, y2 = bbox
    y2 = min(y2 + extra_margin, frame.shape[0])
    crop_frame = cv2.resize(frame[y1:y2, x1:x2], (256, 256), interpolation=cv2.INTER_LANCZOS4)

    random_audio  = torch.randn(1, 50, 384, device=device, dtype=weight_dtype)
    audio_feature = pe(random_audio)
    latents       = vae.get_latents_for_unet(crop_frame).to(dtype=weight_dtype)
    pred_latents  = unet.model(latents, timesteps, encoder_hidden_states=audio_feature).sample
    recon         = vae.decode_latents(pred_latents)

    res_frame     = cv2.resize(recon[0].astype(np.uint8), (x2 - x1, y2 - y1))
    combine_frame = get_image(frame, res_frame, [x1, y1, x2, y2], mode=parsing_mode, fp=fp)

    cv2.imwrite(os.path.join(result_dir, "debug_result.png"), combine_frame)

    info = (f"bbox_shift: {bbox_shift}\nextra_margin: {extra_margin}\n"
            f"parsing_mode: {parsing_mode}\nleft_cheek_width: {left_cheek_width}\n"
            f"right_cheek_width: {right_cheek_width}\n"
            f"Face bbox: [{x1}, {y1}, {x2}, {y2}]")
    return cv2.cvtColor(combine_frame, cv2.COLOR_RGB2BGR), info

# ─── Main inference ───────────────────────────────────────────────────────────
@torch.no_grad()
def inference(audio_path, video_path, bbox_shift,
              extra_margin=10, parsing_mode="jaw",
              left_cheek_width=90, right_cheek_width=90,
              progress=gr.Progress(track_tqdm=True)):

    result_dir = "./results/output/v15"
    os.makedirs(result_dir, exist_ok=True)

    input_basename  = os.path.basename(video_path).split(".")[0]
    audio_basename  = os.path.basename(audio_path).split(".")[0]
    output_basename = f"{input_basename}_{audio_basename}"

    result_img_save_path = os.path.join(result_dir, output_basename)
    crop_coord_save_path = os.path.join("./results", input_basename + ".pkl")
    os.makedirs(result_img_save_path, exist_ok=True)

    output_vid_name = os.path.join(result_dir, output_basename + ".mp4")

    # ── Extract frames ──────────────────────────────────────────────────────
    if get_file_type(video_path) == "video":
        save_dir_full = os.path.join(result_dir, input_basename)
        os.makedirs(save_dir_full, exist_ok=True)
        reader = imageio.get_reader(video_path)
        for i, im in enumerate(reader):
            imageio.imwrite(f"{save_dir_full}/{i:08d}.png", im)
        input_img_list = sorted(glob.glob(os.path.join(save_dir_full, "*.[jpJP][pnPN]*[gG]")))
        fps = get_video_fps(video_path)
    else:
        input_img_list = sorted(
            glob.glob(os.path.join(video_path, "*.[jpJP][pnPN]*[gG]")),
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
        )
        fps = 25

    # ── Audio features ──────────────────────────────────────────────────────
    whisper_input_features, librosa_length = audio_processor.get_audio_feature(audio_path)
    whisper_chunks = audio_processor.get_whisper_chunk(
        whisper_input_features, device, weight_dtype, whisper, librosa_length,
        fps=fps, audio_padding_length_left=2, audio_padding_length_right=2,
    )

    # ── Face landmarks / bbox ───────────────────────────────────────────────
    if os.path.exists(crop_coord_save_path):
        print("Using saved coordinates")
        with open(crop_coord_save_path, "rb") as f:
            coord_list = pickle.load(f)
        frame_list = read_imgs(input_img_list)
    else:
        print("Extracting landmarks (this takes a while)…")
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)
        with open(crop_coord_save_path, "wb") as f:
            pickle.dump(coord_list, f)

    bbox_shift_text = get_bbox_range(input_img_list, bbox_shift)

    # ── Encode frames to latents ─────────────────────────────────────────────
    fp = FaceParsing(left_cheek_width=left_cheek_width, right_cheek_width=right_cheek_width)
    input_latent_list = []
    for bbox, frame in zip(coord_list, frame_list):
        if bbox == coord_placeholder:
            continue
        x1, y1, x2, y2 = bbox
        y2 = min(y2 + extra_margin, frame.shape[0])
        crop = cv2.resize(frame[y1:y2, x1:x2], (256, 256), interpolation=cv2.INTER_LANCZOS4)
        input_latent_list.append(vae.get_latents_for_unet(crop))

    # Cycle for smooth loop
    frame_list_cycle        = frame_list        + frame_list[::-1]
    coord_list_cycle        = coord_list        + coord_list[::-1]
    input_latent_list_cycle = input_latent_list + input_latent_list[::-1]

    # ── Batch inference ──────────────────────────────────────────────────────
    print("Running inference…")
    video_num  = len(whisper_chunks)
    batch_size = 8
    gen = datagen(
        whisper_chunks    = whisper_chunks,
        vae_encode_latents= input_latent_list_cycle,
        batch_size        = batch_size,
        delay_frame       = 0,
        device            = device,
    )
    res_frame_list = []
    for whisper_batch, latent_batch in tqdm(gen, total=int(np.ceil(video_num / batch_size))):
        audio_feat  = pe(whisper_batch)
        latent_batch = latent_batch.to(dtype=weight_dtype)
        pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feat).sample
        for res_frame in vae.decode_latents(pred_latents):
            res_frame_list.append(res_frame)

    # ── Paste back ───────────────────────────────────────────────────────────
    print("Compositing frames…")
    for i, res_frame in enumerate(tqdm(res_frame_list)):
        bbox      = coord_list_cycle[i % len(coord_list_cycle)]
        ori_frame = copy.deepcopy(frame_list_cycle[i % len(frame_list_cycle)])
        x1, y1, x2, y2 = bbox
        y2 = min(y2 + extra_margin, ori_frame.shape[0])
        try:
            res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
        except Exception:
            continue
        combine = get_image(ori_frame, res_frame, [x1, y1, x2, y2], mode=parsing_mode, fp=fp)
        cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png", combine)

    # ── Assemble video ───────────────────────────────────────────────────────
    temp_vid = os.path.join(result_dir, "temp_output.mp4")
    valid    = re.compile(r"\d{8}\.png")
    files    = sorted(
        [f for f in os.listdir(result_img_save_path) if valid.match(f)],
        key=lambda x: int(x.split(".")[0])
    )
    images = [imageio.imread(os.path.join(result_img_save_path, f)) for f in files]
    imageio.mimwrite(temp_vid, images, "FFMPEG", fps=25, codec="libx264", pixelformat="yuv420p")

    video_clip = VideoFileClip(temp_vid)
    audio_clip = AudioFileClip(audio_path)
    video_clip.set_audio(audio_clip).write_videofile(
        output_vid_name, codec="libx264", audio_codec="aac", fps=25
    )
    video_clip.close()
    audio_clip.close()
    os.remove(temp_vid)

    print(f"Saved to {output_vid_name}")
    return output_vid_name, bbox_shift_text

# ─── Gradio UI ────────────────────────────────────────────────────────────────
css = "#input_img {max-width:1024px!important} #output_vid {max-width:1024px;max-height:576px}"

with gr.Blocks(css=css) as demo:
    gr.Markdown("## MuseTalk — Real-Time Video Dubbing")
    with gr.Row():
        with gr.Column():
            audio          = gr.Audio(label="Driving Audio", type="filepath")
            video          = gr.Video(label="Reference Video", sources=["upload"])
            bbox_shift     = gr.Number(label="BBox shift (px)", value=0)
            extra_margin   = gr.Slider(label="Extra Margin", minimum=0, maximum=40, value=10, step=1)
            parsing_mode   = gr.Radio(label="Parsing Mode", choices=["jaw", "raw"], value="jaw")
            left_cheek_w   = gr.Slider(label="Left Cheek Width",  minimum=20, maximum=160, value=90, step=5)
            right_cheek_w  = gr.Slider(label="Right Cheek Width", minimum=20, maximum=160, value=90, step=5)
            bbox_info      = gr.Textbox(label="BBox info", interactive=False)
            with gr.Row():
                debug_btn = gr.Button("1. Test Inpainting")
                run_btn   = gr.Button("2. Generate")
        with gr.Column():
            debug_img  = gr.Image(label="Test Result (first frame)")
            debug_info = gr.Textbox(label="Parameter Info", lines=5)
            out_vid    = gr.Video()

    video.change(fn=check_video, inputs=[video], outputs=[video])

    run_btn.click(
        fn=inference,
        inputs=[audio, video, bbox_shift, extra_margin, parsing_mode, left_cheek_w, right_cheek_w],
        outputs=[out_vid, bbox_info],
    )
    debug_btn.click(
        fn=debug_inpainting,
        inputs=[video, bbox_shift, extra_margin, parsing_mode, left_cheek_w, right_cheek_w],
        outputs=[debug_img, debug_info],
    )

# ─── Windows asyncio fix ──────────────────────────────────────────────────────
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ─── Launch ───────────────────────────────────────────────────────────────────
demo.queue().launch(
    share       = args.share,
    debug       = True,
    server_name = args.ip,
    server_port = args.port,
)