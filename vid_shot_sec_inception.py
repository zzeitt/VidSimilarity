import tensorflow as tf
from scenedetect import VideoManager
from scenedetect import SceneManager
from scenedetect.detectors import ContentDetector
from datasets import load_video
from tqdm import tqdm
import os
import time
import datetime
import cv2
import numpy as np
import math
import faiss


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

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

        
g_mine = tf.Graph()
with g_mine.as_default():
    with tf.gfile.FastGFile('./ckpt/inception/my_model.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')


def pred_inception_path(s_image):
    with tf.Session(graph=g_mine) as sess:
        # Preprocess the image
        fname = s_image
        raw_img = tf.io.gfile.GFile(fname, 'rb').read()
        tf_img = tf.image.decode_jpeg(raw_img).eval()
        tf_fl_img = tf.cast(tf_img, dtype=tf.float32).eval()
        if tf_fl_img.shape[2] == 1:
            tf_fl_img = np.repeat(tf_fl_img, 3, axis=2)
        tf_fl_img_rsz2 = tf.image.resize(
            tf_fl_img, [299, 299], 'bilinear').eval()
        tf_fl_img_rsz2_expand = np.expand_dims(tf_fl_img_rsz2, 0)
        tf_input = tf.subtract(tf_fl_img_rsz2_expand,
                               tf.constant([128], dtype=tf.float32)).eval()
        tf_input = tf.multiply(tf_input, tf.constant(
            [0.0078125], dtype=tf.float32)).eval()

        x_pool_3 = sess.graph.get_tensor_by_name('pool_3/AvgPool:0')
        x_softmax = sess.graph.get_tensor_by_name('softmax/truediv:0')
        pred = sess.run(x_pool_3, {'input_1:0': tf_input}).squeeze()
        return pred
    

def pred_inception_cv(cv_image):
    with tf.Session(graph=g_mine) as sess:      
        # Pass in the image
        rgb_img = cv_image[..., ::-1]

        x_pool_3 = sess.graph.get_tensor_by_name('pool_3/AvgPool:0')
        x_softmax = sess.graph.get_tensor_by_name('softmax/truediv:0')
        pred = sess.run(x_pool_3, {'DecodeJpeg:0': rgb_img}).squeeze()
        return pred
    
    
# =========================================================
#               Calculate Based on Segment
# =========================================================
def calc_seg_sims_inception(li_src, li_dst):
    dim = 2048
    faiss_index = faiss.IndexFlatL2(dim)
    print(f'====> Create Faiss Index.')
    dst_inceptions = []
    src_inceptions = []
    dst_fr_segs = []
    src_fr_segs = []
    
    for idx_dst, dst in enumerate(tqdm(li_dst)):
        dst_frames = np.array(dst)
        # Get middle frame's index
        idx_dst_mid = int(len(dst_frames)/2)
        fname = f'{(idx_dst_mid+1):0{zdst}}.jpg'
        s_dst_frame = os.path.join(s_save_fr_dst, fname)
        
        # Extract features, shape=(2048, )
        dst_feature = pred_inception_path(s_dst_frame)
        dst_inceptions.append(dst_feature)

        idx_fr = 0
        for i in li_dst[:idx_dst]:
            idx_fr += len(i)
        idx_fr += 1

        dst_fr_segs.append((idx_fr, idx_fr+len(dst)-1))
    
    for idx_src, src in enumerate(tqdm(li_src)):
        src_frames = np.array(src)
        # Get middle frame's index
        idx_src_mid = int(len(src_frames)/2)
        fname = f'{(idx_src_mid+1):0{zsrc}}.jpg'
        s_src_frame = os.path.join(s_save_fr_src, fname)
        
        # Extract features, shape=(2048, )
        src_feature = pred_inception_path(s_src_frame)
        src_inceptions.append(src_feature)

        idx_fr = 0
        for i in li_src[:idx_src]:
            idx_fr += len(i)
        idx_fr += 1

        src_fr_segs.append((idx_fr, idx_fr+len(src)-1))

    dst_inceptions = np.array(dst_inceptions).astype('float32')
    src_inceptions = np.array(src_inceptions).astype('float32')
    
    faiss_index.add(dst_inceptions)
    dis, ind = faiss_index.search(
        src_inceptions, len(dst_inceptions)) # search src within dst
    
    # TODO: to fix !!!
    sims = []
    for i in range(len(src_inceptions)):
        for j in range(len(dst_inceptions)):
            sims.append((
                src_fr_segs[i],
                dst_fr_segs[ind[i][j]],
                dis[i][j]
            ))
    
    sims_sorted = sorted(sims, key=lambda x: x[2], reverse=False)
    from pprint import pprint
    pprint(f'====> dst_fr_segs: {dst_fr_segs}')
    pprint(f'====> src_fr_segs: {src_fr_segs}')
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
    s_save_txt = (f'{n_src}_{n_dst}_seg_sec_inception.txt')
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
    sims_sorted_sec = calc_seg_sims_inception(
        li_fr_src_shot_sec, li_fr_dst_shot_sec)
    
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
