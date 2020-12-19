from scenedetect import VideoManager
from scenedetect import SceneManager
from scenedetect.detectors import ContentDetector
from model.visil import ViSiL
from datasets import load_video
from tqdm import tqdm
import os
import time
import datetime
import cv2
import numpy as np
import math


def getFilePaths(path):
    # read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(os.path.join(root, filespath))
    ret.sort()
    return ret


def find_scenes(video_path, threshold=30.0):
    # Create our video & scene managers, then add the detector.
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(threshold=threshold))

    # Base timestamp at frame 0 (required to obtain the scene list).
    base_timecode = video_manager.get_base_timecode()

    # Improve processing speed by downscaling before processing.
    video_manager.set_downscale_factor()

    # Start the video manager and perform the scene detection.
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)

    # Each returned scene is a tuple of the (start, end) timecode.
    return scene_manager.get_scene_list(base_timecode)


def seg_frames_by_sec(np_frames, fps, sec=1):
    li_fr_sec = []
    li_tmp = []
    time = sec
    for idx, fr in enumerate(np_frames):
        idx_1 = idx + 1
        if idx_1 % int(time*fps) == 1:
            li_tmp = []
        li_tmp.append(fr)
        if idx_1 % int(time*fps) == 0 or idx_1 == len(np_frames):
            li_fr_sec.append(li_tmp)
    return li_fr_sec


def seg_frames_by_count(frames, frnum=20):
    li_fr_count = []
    li_tmp = []
    for idx, fr in enumerate(frames):
        idx_1 = idx + 1
        if idx_1 % frnum == 1:
            li_tmp = []
        li_tmp.append(fr)
        if idx_1 % frnum == 0 or idx_1 == len(frames):
            li_fr_count.append(li_tmp)
    return li_fr_count


# =========================================================
#               Calculate Based on Segment
# =========================================================
def calc_seg_sims(li_src, li_dst):
    # Initialize ViSiL model and load pre-trained weights
    model = ViSiL('ckpt/resnet/')
    sims = []
    for idx_src, src in enumerate(tqdm(li_src)):
        for idx_dst, dst in enumerate(tqdm(li_dst)):
            dst_frames = np.array(dst)
            src_frames = np.array(src)

            dst_features = model.extract_features(dst_frames, batch_sz=32)
            src_features = model.extract_features(src_frames, batch_sz=32)

            sim = model.calculate_video_similarity(dst_features, src_features)
            
            idx_s = 0
            idx_e = 0
            for i in li_src[:idx_src]:
                idx_s += len(i)
            for i in li_dst[:idx_dst]:
                idx_e += len(i)
            idx_s += 1
            idx_e += 1
            
            sims.append((
                (idx_s, idx_s+len(src)-1),
                (idx_e, idx_e+len(dst)-1),
                sim
            ))

    sims_sorted = sorted(sims, key=lambda x: x[2], reverse=True)
    from pprint import pprint
    pprint(f'====> sims:\n {sims}')
    return sims_sorted


if __name__ == "__main__":
    s1 = 'videos/src/01_castle.mp4'
    s2 = 'videos/src/02_aquarium.mp4'
    s3 = 'videos/src/02_insert.mp4'
    s4 = 'videos/src/03_mountain.mp4'
    s5 = 'videos/src/03_rev.mp4'
    
    s_src_vid = s2
    s_dst_vid = s3
    
    n_src = os.path.splitext(os.path.basename(s_src_vid))[0]
    n_dst = os.path.splitext(os.path.basename(s_dst_vid))[0]
    s_save_txt_dir = './txt'
    s_save_txt = (f'{n_src}_{n_dst}_seg_sec.txt')
    os.makedirs(s_save_txt_dir, exist_ok=True)
    s_save_txt = os.path.join(s_save_txt_dir, s_save_txt)
    
    s_save_fr_src = os.path.join('./frames', n_src)
    s_save_fr_dst = os.path.join('./frames', n_dst)
    os.makedirs(s_save_fr_src, exist_ok=True)
    os.makedirs(s_save_fr_dst, exist_ok=True)
    
    # ======================================
    #           Calculate Sim
    # ======================================
    # --------------------------------------
    #               Start
    # --------------------------------------
    start_time = time.time()
    
    # Get FPS
    query_video = load_video(s_dst_vid, True)
    target_video = load_video(s_src_vid, True)
    cap_dst = cv2.VideoCapture(s_dst_vid)
    fps_dst = cap_dst.get(cv2.CAP_PROP_FPS)
    cap_src = cv2.VideoCapture(s_src_vid)
    fps_src = cap_src.get(cv2.CAP_PROP_FPS)
    
    print(f'====> len(query_video): {len(query_video)}')
    print(f'====> len(target_video): {len(target_video)}')
    
    # Save frames
    zdst = math.ceil(math.log10(len(query_video)))
    zsrc = math.ceil(math.log10(len(target_video)))
    for idx, fr in enumerate(query_video):
        cv2.imwrite(
            os.path.join(s_save_fr_dst, f'{(idx+1):0{zdst}}.jpg'),
            cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))

    for idx, fr in enumerate(target_video):
        cv2.imwrite(
            os.path.join(s_save_fr_src, f'{(idx+1):0{zsrc}}.jpg'),
            cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    
    # --------------------------------------
    #               Get Shots
    # --------------------------------------
    cuts_src_ftc = find_scenes(s_src_vid)
    cuts_dst_ftc = find_scenes(s_dst_vid)
    cuts_src_fr = [(i[0].get_frames(), i[1].get_frames())
                   for i in cuts_src_ftc]
    cuts_dst_fr = [(i[0].get_frames(), i[1].get_frames())
                   for i in cuts_dst_ftc]
    
    li_fr_src_shot = []
    li_fr_dst_shot = []
    for src in cuts_src_fr:
        li_fr_src_shot.append(target_video[src[0]:src[1]])
    for dst in cuts_dst_fr:
        li_fr_dst_shot.append(query_video[dst[0]:dst[1]])

    li_fr_src_shot_sec = []
    li_fr_dst_shot_sec = []
    for src in li_fr_src_shot:
        li_fr_src_shot_sec += seg_frames_by_sec(src, fps_src, 1)
    for dst in li_fr_dst_shot:
        li_fr_dst_shot_sec += seg_frames_by_sec(dst, fps_dst, 1)

    # --------------------------------------
    #               Calculate
    # --------------------------------------
    sims_sorted_sec = calc_seg_sims(li_fr_src_shot_sec, li_fr_dst_shot_sec)

    # --------------------------------------
    #               End
    # --------------------------------------
    cap_dst.release()
    cap_src.release()
    time_past = datetime.timedelta(
        seconds=(time.time()-start_time))
    print(f'====> Elapsed time: {time_past}')

    # ======================================
    #           Obtain Socres
    # ======================================
    score_s2d = 0
    score_d2s = 0
    li_src_fr = []
    li_dst_fr = []
    li_src_write = [f'\n====> {n_src}']
    li_dst_write = [f'\n====> {n_dst}']
    for i in sims_sorted_sec:
        fr_src = i[0]
        fr_dst = i[1]
        sim = i[2]
        if sim > 0 and (fr_src not in li_src_fr) and (fr_dst not in li_dst_fr):
            s_s = f'====> sim: {sim}, src_frame: {fr_src}'
            s_d = f'====> sim: {sim}, dst_frame: {fr_dst}'
            li_src_write.append(s_s)
            li_dst_write.append(s_d)
            print(s_s)
            print(s_d)
            score_s2d += sim * (fr_src[1] - fr_src[0]) / fps_src
            score_d2s += sim * (fr_dst[1] - fr_dst[0]) / fps_dst
            li_src_fr.append(fr_src)
            li_dst_fr.append(fr_dst)
            
    score_s2d /= len(target_video) / fps_src
    score_d2s /= len(query_video) / fps_dst
    print(f'====> socre_s2d: {score_s2d}')
    print(f'====> score_d2s: {score_d2s}')
    
    
    # =======================================
    #           Save Results
    # =======================================
    with open(s_save_txt, 'w') as f:
        for i in sims_sorted_sec:
            f.write(str(i) + '\n')
        f.write(f'====> Elapsed time: {time_past}\n')
        for i in li_src_write:
            f.write(i + '\n')
        f.write('\n')
        for i in li_dst_write:
            f.write(i + '\n')
        f.write('\n')
        f.write(f'score_s2d: {score_s2d}\n')
        f.write(f'score_d2s: {score_d2s}\n')
