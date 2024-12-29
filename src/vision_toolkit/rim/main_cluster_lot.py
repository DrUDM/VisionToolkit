# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 07:57:18 2024
"""

import os
import sys
sys.path.append('/home/deyo/.local/bin')
sys.path.append('/home/deyo/')
sys.path.append('/home/deyo/rim_src/pytlsd')

import time
import datetime

import numpy as np
import pandas as pd
import cv2
import torch 
import func_timeout
import glob
import gc
import logging
 
from gluestick import GLUESTICK_ROOT
import gluestick
from gluestick.models.two_view_pipeline import TwoViewPipeline 
import homography_est.homography_est as hest



def preprocessing_rim(gaze_data, 
                      time_stamps, 
                      config
                      ):
    '''
    Parameters
    ----------
    config : TYPE
        DESCRIPTION.

    Returns
    -------
    output_df : TYPE
        DESCRIPTION.

    '''
    gaze_df = pd.read_csv(gaze_data) 
    gaze_ts = gaze_df['timestamp [ns]'].values/1e6
     
    world_df = pd.read_csv(time_stamps)
    world_ts = world_df['timestamp [ns]'].values/1e6
   
    per_n_frames = config['processing']['downsampling_factor'] 
    frame = (np.ones(len(gaze_ts))*(len(world_ts) - 1)).astype(int)
   
    frame_idx = 0
    gaze_index = 0
   
    while True:
        try: 
            g_ts = gaze_ts[gaze_index]
            w_ts = world_ts[frame_idx] 
            if g_ts <= w_ts: 
                frame[gaze_index] = int(frame_idx)
                gaze_index += 1 
            else:
                frame_idx +=per_n_frames 
        except IndexError:
            break  
        
    gaze_x = gaze_df['gaze x [px]']
    gaze_x /= config['processing']['camera']['width'] 
    gaze_y = gaze_df['gaze y [px]']
    gaze_y /= config['processing']['camera']['height'] 
    
    confidence = gaze_df['worn'] 
    output_df = pd.DataFrame(data=dict({'timestamp': gaze_ts, 
                                        'frame_idx': frame, 
                                        'confidence': confidence, 
                                        'norm_pos_x': gaze_x, 
                                        'norm_pos_y': gaze_y}))
    return output_df
 
    
def processRecording(gazeWorld_df, 
                     reference_image,
                     world_camera,
                     out_name,
                     config,
                     warm_start=0):
    ''' 
    Parameters
    ----------
    gazeWorld_df : TYPE
        DESCRIPTION.
    config : TYPE
        DESCRIPTION.
 
    Returns
    -------
    None.
 
    '''
    frameProcessing_startTime = time.time()
      
    ## Create output directory
    outputDir = config['processing']['files']['outputDir']
    os.makedirs(outputDir, exist_ok=True)
        
    ## Copy the reference stim into the output dir 
    framesToCompute = gazeWorld_df['frame_idx'].values.tolist()
    last_ = framesToCompute[-1]
    frameCounter = 0 
    gazeMapped_df = None 
    
    ## Set CUDA device if available
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    ## Initialize the model
    pipeline_model = TwoViewPipeline(config['model']).to(device).eval()
    line_homography = config['processing'], ['model']['use_lines_homoraphy']
    
    
    ## Load and preprocess the reference image
    d_w = config['processing']['camera']['down_width']
    d_h = config['processing']['camera']['down_height']
    down_points = (d_w, d_h)
    ref_frame = cv2.imread(reference_image, 0) 
    ref_frame = cv2.resize(ref_frame, 
                           down_points, 
                           interpolation= cv2.INTER_LINEAR) 
    torch_ref = gluestick.numpy_image_to_torch(ref_frame) 
    ## Perform initial prediction on the reference image
    pred = pipeline_model({'image0': torch_ref.to(device)[None], 
                           'image1': torch_ref.to(device)[None]})  
    ## Open video file
    vid = cv2.VideoCapture(world_camera) 
    
    ## Keep reference frame descriptors 
    pred_ref = dict({})
    for it_ in ['keypoints1', 
                'keypoint_scores1', 
                'descriptors1', 
                'pl_associativity1', 
                'num_junctions1', 
                'lines1', 'orig_lines1', 
                'lines_junc_idx1', 
                'line_scores1',
                'valid_lines1']:
        pred_ref.update({it_[:-1]: pred[it_]}) 
    del pred

    if warm_start>0:
        gazeMapped_df = pd.read_csv('{od}/mappedGaze_{n_}.csv'.format(od = outputDir, 
                                                                      n_ = out_name))
    else:
        gazeMapped_df = pd.DataFrame({
            'gaze_ts': [],
            'worldFrame': [],
            'confidence': [],
            'world_gazeX': [],
            'world_gazeY': [],
            'ref_gazeX': [],
            'ref_gazeY': [], 
            'mapped': [],
            'number_matches': []
            }) 
        
    print("Processing frames") 
    pred=None
    world2ref_transform=None 
    current_pred=False 
    
    m_time = config['processing']['max_time']
    header_saved = False  # Flag to check if header is saved
    
    while vid.isOpened():   
        ret, frame = vid.read()
        if (ret is True) and (frameCounter in framesToCompute) and frameCounter >= warm_start:    
            sys.stdout.flush() 
            sys.stdout.write(f"\r frame {frameCounter} / {last_}")
            
            world_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            world_frame = cv2.resize(world_frame, 
                                     down_points, 
                                     interpolation= cv2.INTER_LINEAR)  
            start_time = time.time()  
            try: 
                torch_world = gluestick.numpy_image_to_torch(world_frame)
                pred = func_timeout.func_timeout(m_time, 
                                                 pipeline_model,
                                                 args=[{'image0': torch_world.to(device)[None], 
                                                        'image1': torch_ref.to(device)[None],
                                                        'ref': pred_ref}])
                current_pred=True 
                del torch_world

            except:
                print('KILLED: too long')
                current_pred=False
                del torch_world
                pass

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(" --- %s sec" % round(elapsed_time, 3))
           
            ## If current pred and enough matches, update world2ref_transform
            if current_pred:   
                try:
                    pred = gluestick.batch_to_np(pred)
                    kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
                    m0 = pred["matches0"] 
                    valid_matches = m0 != -1
                    match_indices = m0[valid_matches]
                    matched_kps0 = kp0[valid_matches]
                    matched_kps1 = kp1[match_indices]
                 
                    ## Find homography
                    if line_homography:
                        line_seg0, line_seg1 = pred["lines0"], pred["lines1"]
                        line_matches = pred["line_matches0"]
                        
                        valid_matches = line_matches != -1
                        match_indices = line_matches[valid_matches]
                        matched_lines0 = line_seg0[valid_matches]
                        matched_lines1 = line_seg1[match_indices]
                        
                        ref2world = hest.ransac_point_line_homography(matched_kps0, matched_kps1, 
                                                                      [hest.LineSegment(l[0], l[1]) for l in matched_lines0.reshape(-1, 2, 2)], 
                                                                      [hest.LineSegment(l[0], l[1]) for l in matched_lines1.reshape(-1, 2, 2)], 
                                                                      1., False, [], [])
                    else: 
                        ref2world, mask = cv2.findHomography(matched_kps1, 
                                                         matched_kps0, 
                                                         cv2.RANSAC, 
                                                         10)  
                        
                    world2ref_transform = cv2.invert(ref2world)[1]  
                    
                except:
                    current_pred=False
                    print('KILLED: could not access prediction')
                    pass
                
            if current_pred:
                try:
                    number_matches = (pred['matches0'] >= 0).sum()
                    del pred
                except:
                    number_matches = 0
                    current_pred=False
            else:
                number_matches = 0
                current_pred=False
                
            ## If world2ref_transform already initialized, compute ref gaze
            if world2ref_transform is not None:
                thisFrame_gazeData_world = gazeWorld_df.loc[gazeWorld_df['frame_idx'] == frameCounter]
                world_pts = []
                ref_pts = []
                for i, gazeRow in thisFrame_gazeData_world.iterrows():  
                    ts = gazeRow['timestamp']
                    conf = gazeRow['confidence'] 
                    ## Translate normalized gaze data to world pixel coords 
                    world_gazeX = gazeRow['norm_pos_x'] * d_w
                    world_gazeY = gazeRow['norm_pos_y'] * d_h
                    world_pts.append([world_gazeX, world_gazeY])
                    ref_gazeX, ref_gazeY = mapCoords2D((world_gazeX, world_gazeY), world2ref_transform)
                    ref_pts.append([ref_gazeX, ref_gazeY]) 
                    thisRow_df = pd.DataFrame({
                        'gaze_ts': ts,
                        'worldFrame': frameCounter,
                        'confidence': conf,
                        'world_gazeX': world_gazeX,
                        'world_gazeY': world_gazeY,
                        'ref_gazeX': ref_gazeX,
                        'ref_gazeY': ref_gazeY, 
                        'mapped': current_pred, 
                        'number_matches': number_matches
                        }, index = [i]) 
                    gazeMapped_df = pd.concat([gazeMapped_df, 
                                               thisRow_df])  
        frameCounter += 1
        
        ## Release video if done processing
        if frameCounter > np.max(np.array(framesToCompute)):
            vid.release()
        
        ## Periodically save intermediate results
        if frameCounter % 1000 == 0:
            if not header_saved:
                gazeMapped_df.to_csv(f'{outputDir}/mappedGaze_{out_name}.csv', mode='a', header=True, index=False)
                header_saved = True
            else:
                gazeMapped_df.to_csv(f'{outputDir}/mappedGaze_{out_name}.csv', mode='a', header=False, index=False)
            gazeMapped_df = pd.DataFrame({'gaze_ts': [], 'worldFrame': [], 'confidence': [], 'world_gazeX': [], 'world_gazeY': [], 'ref_gazeX': [], 'ref_gazeY': [], 'current_transform': []})
            gc.collect()
    
    ## Save final results
    gazeMapped_df.to_csv(f'{outputDir}/mappedGaze_{out_name}.csv', mode='a', header=False, index=False)
    endTime = time.time()
    frameProcessing_time = endTime - frameProcessing_startTime
    print(f"Total processing Time: {frameProcessing_time} seconds")
      
        
def mapCoords2D(coords, transform2D): 
    '''

    Parameters
    ----------
    coords : TYPE
        DESCRIPTION.
    transform2D : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.

    '''
   
    ## Reshape input coordinates
    coords = np.array(coords).reshape(-1, 1, 2)  
    ## Compute reference coordinates according to the homography matrix
    mappedCoords = cv2.perspectiveTransform(coords, transform2D)
    mappedCoords = np.round(mappedCoords.ravel(), 3)

    return mappedCoords[0], mappedCoords[1]
 


def process_recordings(recordings, folder_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = output_dir + f'/log_{current_time}.txt'

    logging.basicConfig(filename=log_file_path, level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    for recording in recordings:
            try:
                logging.info(f"start processing: {recording}")
                print(f"start processing: {recording}")
        
                recording_path = os.path.join(folder_path, recording)
        
                config = {
                    'processing': {
                        'downsampling_factor': 2,
                        'max_time':2,
                        'min_line_matches': 0,
                        'min_point_matches': 100,
                        'camera': {
                            'width': 1600,
                            'height': 1200,
                            'down_width': 600,
                            'down_height': 450
                        },
                        'files': {
                            'outputDir': output_dir,
                        },
                    },
    
                    'model': {
                        'name': 'two_view_pipeline',
                        'use_lines': True,
                        'use_lines_homoraphy': False,
                        'extractor': {
                            'name': 'wireframe',
                            'sp_params': {
                                'force_num_keypoints': False,
                                'max_num_keypoints': 3000,
                            },
                            'wireframe_params': {
                                'merge_points': True,
                                'merge_line_endpoints': True,
                            },
                            'max_n_lines': 0,
                        },
                        'matcher': {
                            'name': 'gluestick',
                            'weights': str(GLUESTICK_ROOT / 'resources' / 'weights' / 'checkpoint_GlueStick_MD.tar'),
                            'trainable': False,
                        },
                        'ground_truth': {
                            'from_pose_depth': False,
                        }
                    }
                }

                gaze_data = os.path.join(recording_path, 'gaze.csv')
                time_stamps = os.path.join(recording_path, 'world_timestamps.csv')
                reference_image = ' '.join(glob.glob(os.path.join(recording_path, 'image_ref.jpg')))
                world_camera = ' '.join(glob.glob(os.path.join(recording_path, '*.mp4')))
                out_name = recording + '_gaze'
    
                preProData = preprocessing_rim(gaze_data, time_stamps, config)
                processRecording(preProData, reference_image, world_camera, out_name, config)
                
                
    
            except Exception as e:
                print(f"Error processing {recording}: {e}")
                logging.error(f"Error processing {recording}: {e}")
                continue
            
            finally:
                logging.info(f"end processing: {recording}")
                print(f"end processing: {recording}")
                
                # Clear variables to free memory
                torch.cuda.empty_cache()
                gc.collect()



if __name__ == '__main__':
    lot = 'lot_test'
   
    # folder_path = './data/deyo/input/' + lot
    # output_dir = './data/deyo/output/' + lot
    
    folder_path = '../../../data/deyo/input/' + lot
    output_dir = '../../../data/deyo/output/' + lot
    
    recordings = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
    process_recordings(recordings, folder_path, output_dir)







