# rewrite model_fn and transform_fn
# model_fn: load model
# transform_fn: get input, transform and output
import os
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from musetalk.utils.utils import get_file_type, get_video_fps, datagen, load_all_model
from transformers import WhisperModel
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.blending import get_image
# rewrite preprocessing, save only read_imgs and coord_placeholder two functions
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder

from omegaconf import OmegaConf
import glob
import pickle
import cv2
import numpy as np
from tqdm import tqdm
import copy
from PIL import Image
import shutil
import io
import json


def model_fn(model_dir):
    """Load the model and return it.
    Providing this function is optional.
    There is a default_model_fn available, which will load the model
    compiled using SageMaker Neo. You can override the default here.
    The model_fn only needs to be defined if your model needs extra
    steps to load, and can otherwise be left undefined.

    Keyword arguments:
    model_dir -- the directory path where the model artifacts are present
    """
    model_dir = "/models"
    config = OmegaConf.load(os.path.join(model_dir, "models.yaml"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae, unet, pe = load_all_model(
        unet_model_path=os.path.join(model_dir, config["unet_model_path"]),
        vae_type=config["vae_type"],
        unet_config=os.path.join(model_dir, config["unet_config"]),
        device=device,
        model_dir=model_dir
    )
    audio_processor = AudioProcessor(feature_extractor_path=os.path.join(model_dir, config["whisper_dir"]))
    
    weight_dtype = unet.model.dtype
    whisper = WhisperModel.from_pretrained(os.path.join(model_dir, config["whisper_dir"]))
    whisper = whisper.to(device=device, dtype=weight_dtype).eval()
    whisper.requires_grad_(False)
    
    pe = pe.to(device)
    vae.vae = vae.vae.to(device)
    unet.model = unet.model.to(device)

    if config["version"] == "v15":
        fp = FaceParsing(left_cheek_width=config["left_cheek_width"],
                         right_cheek_width=config["right_cheek_width"])
    else:
        raise NotImplementedError
    
    models = {
        "vae": vae,
        "unet": unet,
        "pe": pe, 
        "whisper": whisper,
        "AP": audio_processor,
        "FP": fp
    }
    
    return models

@torch.no_grad()
def transform_fn(model, request_body, request_content_type,
                    response_content_type):
    """Run prediction and return the output.
    The function
    1. Pre-processes the input request
    2. Runs prediction
    3. Post-processes the prediction output.
    """
    vae = model["vae"]
    unet = model["unet"]
    pe = model["pe"]
    whisper = model["whisper"]
    audio_processor = model["AP"]
    fp = model["FP"]

    ################################### preprocess ####################################
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    ############## TODO: get config by payload ###################
    wav = None  # here access from request_body or s3 
    id = request_body["id"]      # id access from request_body, table
    video_path = f"materials/{id}/{id}"     # TODO: change to images list or folders
    audio_path = request_body["wav"]
    crop_coord_pkl = f"materials/{id}/{id}.pkl"
    vid_name = request_body["vid_name"]
    output_save_path = f"./results/{id}"
    output_vid_name = os.path.join(output_save_path, f"{vid_name}.mp4")  # modify by payload
    os.makedirs(output_save_path, exist_ok=True)
    input_latent_list_cycle_path = f"materials/{id}/{id}_latent.pkl"

    ##############################################################

    ############### parameter settings ###########################
    bbox_shift = 0
    fps = 25
    audio_padding_length_left = 2
    audio_padding_length_right = 2
    weight_dtype = unet.model.dtype
    extra_margin = 10
    batch_size = 8          # could be change if the VRAM of the machine is big enough
    timesteps = torch.tensor([0], device=device)
    ## tmp dir --> preprocess before pushing
    input_img_list = glob.glob(os.path.join(video_path, '*.[jpJP][pnPN]*[gG]'))
    input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    result_img_save_path = "./tmp_save_img"
    temp_dir = "./temp_video_folder"
    os.makedirs(result_img_save_path, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    ###############################################################
    
    ################ Audio Process ####################
    whisper_input_features, librosa_length = audio_processor.get_audio_feature(audio_path)
    whisper_chunks = audio_processor.get_whisper_chunk(
        whisper_input_features, 
        device, 
        weight_dtype, 
        whisper, 
        librosa_length,
        fps=fps,
        audio_padding_length_left=audio_padding_length_left,
        audio_padding_length_right=audio_padding_length_right,
    )
    ###################################################
    
    ### using saved coordinates
    # TODO: check open pickle files or transfer in by payload
    with open(crop_coord_pkl, "rb") as f:
        coord_list = pickle.load(f)
    frame_list = read_imgs(input_img_list)

    print(f"Number of frames: {len(frame_list)}")         
    ############################# end of preprocessing ##################################

    ################################## predict ##########################################
    frame_list_cycle = frame_list + frame_list[::-1]
    coord_list_cycle = coord_list + coord_list[::-1]
    with open(input_latent_list_cycle_path, "rb") as f:
        input_latent_list_cycle = pickle.load(f)

    video_num = len(whisper_chunks)
    batch_size = batch_size
    print("whisper generating...")
    gen = datagen(
        whisper_chunks=whisper_chunks,
        vae_encode_latents=input_latent_list_cycle,
        batch_size=batch_size,
        delay_frame=0,
        device=device,
    )
    res_frame_list = []
    total = int(np.ceil(float(video_num) / batch_size))
    
    print("Starting inference")
    # Execute inference
    for i, (whisper_batch, latent_batch) in enumerate(tqdm(gen, total=total)):
        audio_feature_batch = pe(whisper_batch)
        latent_batch = latent_batch.to(dtype=weight_dtype)
        
        pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
        recon = vae.decode_latents(pred_latents)
        for res_frame in recon:
            res_frame_list.append(res_frame)
    ############################# end of predict ########################################
    
    ############################## post process ##############################
    frame = frame_list[0]
    print("Padding generated images to original video size")
    for i, res_frame in enumerate(tqdm(res_frame_list)):
        bbox = coord_list_cycle[i%(len(coord_list_cycle))]
        ori_frame = copy.deepcopy(frame_list_cycle[i%(len(frame_list_cycle))])
        x1, y1, x2, y2 = bbox
        
        y2 = y2 + extra_margin
        y2 = min(y2, frame.shape[0])
        try:
            res_frame = cv2.resize(res_frame.astype(np.uint8), (x2-x1, y2-y1))
        except Exception as e:
            print(e)
            continue
        
        # Merge results with version-specific parameters
        combine_frame = get_image(ori_frame, res_frame, [x1, y1, x2, y2], mode="jaw", fp=fp)
        
        cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png", combine_frame)

    ## from image to video
    temp_vid_path = f"{temp_dir}/temp.mp4"
    cmd_img2video = f"ffmpeg -y -v warning -r {fps} -f image2 -i {result_img_save_path}/%08d.png -vcodec libx264 -vf format=yuv420p -crf 18 {temp_vid_path}"
    print("Video generation command:", cmd_img2video)
    os.system(cmd_img2video)   
    
    ## add voice
    cmd_combine_audio = f"ffmpeg -y -v warning -i {audio_path} -i {temp_vid_path} {output_vid_name}"
    print("Audio combination command:", cmd_combine_audio) 
    os.system(cmd_combine_audio)
    
    shutil.rmtree(result_img_save_path)
    os.remove(temp_vid_path)

    ## save to s3
    print(f"Results saved to {output_vid_name}")

    # TODO: use boto3 to upload video to s3
    return output_vid_name, response_content_type

if __name__ == "__main__":
    models = model_fn(".")
    transform_fn(models, 
                 {"wav":"data/audio/yongen.wav", "id": "jiachi_long", "vid_name": "jiachi_test"},
                 None,
                 200
                 )