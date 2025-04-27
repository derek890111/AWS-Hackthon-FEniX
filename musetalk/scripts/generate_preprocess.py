import os
import cv2
import math
import copy
import torch
import glob
import shutil
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from transformers import WhisperModel

from musetalk.utils.blending import get_image
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.utils import get_file_type, get_video_fps, datagen, load_all_model
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder

@torch.no_grad()
def main(args):
    # Configure ffmpeg path
    if args.ffmpeg_path not in os.getenv('PATH'):
        print("Adding ffmpeg to PATH")
        os.environ["PATH"] = f"{args.ffmpeg_path}:{os.environ['PATH']}"
    
    # Set computing device
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    
    # Load model weights
    vae, unet, pe = load_all_model(
        unet_model_path=args.unet_model_path, 
        vae_type=args.vae_type,
        unet_config=args.unet_config,
        device=device
    )
    timesteps = torch.tensor([0], device=device)

    # Convert models to half precision if float16 is enabled
    if args.use_float16:
        pe = pe.half()
        vae.vae = vae.vae.half()
        unet.model = unet.model.half()
    
    # Move models to specified device
    pe = pe.to(device)
    vae.vae = vae.vae.to(device)
    unet.model = unet.model.to(device)
        
    # Initialize face parser with configurable parameters based on version
    if args.version == "v15":
        fp = FaceParsing(
            left_cheek_width=args.left_cheek_width,
            right_cheek_width=args.right_cheek_width
        )
    else:  # v1
        fp = FaceParsing()
    
    # Load inference configuration
    inference_config = OmegaConf.load(args.inference_config)
    print("Loaded inference config:", inference_config)
    
    # Process each task
    for task_id in inference_config:
        if inference_config[task_id].get("result_dir", False):
            args.result_dir = inference_config[task_id]["result_dir"]
        print(">>>saving points:", args.result_dir)
        try:
            # Get task configuration
            video_path = inference_config[task_id]["video_path"]
            audio_path = inference_config[task_id]["audio_path"]
            
            # Set bbox_shift based on version
            bbox_shift = 0  # v15 uses fixed bbox_shift
            
            # Set output paths
            input_basename = os.path.basename(video_path).split('.')[0]
            audio_basename = os.path.basename(audio_path).split('.')[0]
            output_basename = f"{input_basename}"
            
            # Set result save paths
            result_img_save_path = os.path.join(args.result_dir, output_basename)
            crop_coord_save_path = os.path.join(args.result_dir, input_basename+".pkl")
            os.makedirs(result_img_save_path, exist_ok=True)
            
            # Extract frames from source video
            if get_file_type(video_path) == "video":
                save_dir_full = os.path.join(args.result_dir, input_basename)
                os.makedirs(save_dir_full, exist_ok=True)
                cmd = f"ffmpeg -v fatal -i {video_path} -start_number 0 {save_dir_full}/%08d.png"
                os.system(cmd)
                input_img_list = sorted(glob.glob(os.path.join(save_dir_full, '*.[jpJP][pnPN]*[gG]')))
                fps = get_video_fps(video_path)
            elif get_file_type(video_path) == "image":
                input_img_list = [video_path]
                fps = args.fps
            elif os.path.isdir(video_path):
                input_img_list = glob.glob(os.path.join(video_path, '*.[jpJP][pnPN]*[gG]'))
                input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                fps = args.fps
            else:
                raise ValueError(f"{video_path} should be a video file, an image file or a directory of images")

            
            # Preprocess input images
            if os.path.exists(crop_coord_save_path) and args.use_saved_coord:
                print("Using saved coordinates")
                with open(crop_coord_save_path, 'rb') as f:
                    coord_list = pickle.load(f)
                frame_list = read_imgs(input_img_list)
            else:
                print("Extracting landmarks... time-consuming operation")
                coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)
                # do not save initial
                with open(crop_coord_save_path, 'wb') as f:
                    pickle.dump(coord_list, f)
            
            print(f"Number of frames: {len(frame_list)}")         
            input_latent_list = []
            assert len(coord_list) == len(frame_list)
            _with_face_frame_list = []
            _with_face_coord_list = []
            
            REBUILD_IMG = False
            # rebuild frame and coord if there is no face
            for idx, (bbox, frame) in enumerate(zip(coord_list, frame_list)):
                if bbox == coord_placeholder:
                    # delete frame
                    REBUILD_IMG = True
                    continue
                x1, y1, x2, y2 = bbox
                y2 = y2 + args.extra_margin
                y2 = min(y2, frame.shape[0])
                crop_frame = frame[y1:y2, x1:x2]
                crop_frame = cv2.resize(crop_frame, (256,256), interpolation=cv2.INTER_LANCZOS4)
                # vae
                latents = vae.get_latents_for_unet(crop_frame)
                input_latent_list.append(latents)
                _with_face_coord_list.append(bbox)
                _with_face_frame_list.append(input_img_list[idx])
            
            if REBUILD_IMG:
                # rebuild img                
                if os.name == "nt":
                    root_path = "\\".join(input_img_list[0].split("\\")[:-1])
                else:
                    root_path = "/".join(input_img_list[0].split("/s")[:-1])
                for new_index, original_index in enumerate(_with_face_frame_list):
                    new_name = original_index[:-12] + f"{new_index:08d}.png"
                    shutil.move(original_index, new_name)
                
                for index in range(len(_with_face_frame_list), len(input_img_list)):
                    filename = os.path.join(root_path, f"{index:08d}.png")
                    if os.path.exists(filename):
                        os.remove(filename)
                # rebuild coord
                with open(crop_coord_save_path, 'wb') as f:
                    pickle.dump(_with_face_coord_list, f)

            input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
            
            latent_save_path = os.path.join(args.result_dir, input_basename+"_latent.pkl")
            with open(latent_save_path, "wb") as f:
                pickle.dump(input_latent_list_cycle, f)
        except Exception as e:
            print("Error occurred during processing:", e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ffmpeg_path", type=str, default="./ffmpeg-4.4-amd64-static/", help="Path to ffmpeg executable")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--vae_type", type=str, default="sd-vae-ft-mse", help="Type of VAE model")
    parser.add_argument("--unet_config", type=str, default="./models/musetalk/musetalk.json", help="Path to UNet configuration file")
    parser.add_argument("--unet_model_path", type=str, default="./models/musetalkV15/unet.pth", help="Path to UNet model weights")
    parser.add_argument("--whisper_dir", type=str, default="./models/whisper", help="Directory containing Whisper model")
    parser.add_argument("--inference_config", type=str, default="configs/inference/test_img.yaml", help="Path to inference configuration file")
    parser.add_argument("--bbox_shift", type=int, default=0, help="Bounding box shift value")
    parser.add_argument("--result_dir", default='./results', help="Directory for output results")
    parser.add_argument("--extra_margin", type=int, default=10, help="Extra margin for face cropping")
    parser.add_argument("--fps", type=int, default=25, help="Video frames per second")
    parser.add_argument("--audio_padding_length_left", type=int, default=2, help="Left padding length for audio")
    parser.add_argument("--audio_padding_length_right", type=int, default=2, help="Right padding length for audio")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--output_vid_name", type=str, default=None, help="Name of output video file")
    parser.add_argument("--use_saved_coord", action="store_true", help='Use saved coordinates to save time')
    parser.add_argument("--saved_coord", action="store_true", help='Save coordinates for future use')
    parser.add_argument("--use_float16", action="store_true", help="Use float16 for faster inference")
    parser.add_argument("--parsing_mode", default='jaw', help="Face blending parsing mode")
    parser.add_argument("--left_cheek_width", type=int, default=90, help="Width of left cheek region")
    parser.add_argument("--right_cheek_width", type=int, default=90, help="Width of right cheek region")
    parser.add_argument("--version", type=str, default="v15", choices=["v1", "v15"], help="Model version to use")
    args = parser.parse_args()
    main(args)
