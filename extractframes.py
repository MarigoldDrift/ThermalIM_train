import cv2
import os
import numpy as np
import json 
from datetime import date

root_dir = os.getcwd()
def save_frames_and_keypoints(vid_path,cap,outpath,total_frames):
    vid_name = os.path.split(vid_path)[1]
    openpose_kp = np.load(npz_name)['pose_2d']
    coco_kp = openpose_to_coco(openpose_kp)
    frame_count = 0
    print(coco_kp.shape)
    coco_kp[:,:,0] *= 512/1920
    coco_kp[:,:,1] *= 512/1080
    frame_kpt = []
    while cap.isOpened():
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            #print("Can't receive frame (stream end?). Exiting ...")
            break
        if frame_count%10 ==1:
            #print('frame saved to ',outpath+str(frame_count)+'.jpg')
            total_frames +=1
            frame = cv2.resize(frame,(512,512))
            output_img_name = '{:012d}'.format(total_frames) +'.jpg'
            cv2.imwrite(outpath + output_img_name,frame)
            frame_kpt = coco_kp[frame_count].reshape(51).tolist()
            add_kpts_coco_annot(frame_kpt,output_img_name)
        frame_count+=1
    cap.release()
    return total_frames
def openpose_to_coco(openpose_kp):
    openpose_kp = np.transpose(openpose_kp,(1,0,2))
    
    # rearrange keypoints to fit COCO format
    coco_kp = np.array([openpose_kp[0],openpose_kp[16],openpose_kp[15],openpose_kp[18],openpose_kp[17],openpose_kp[5],openpose_kp[2],
               openpose_kp[6],openpose_kp[3],openpose_kp[7],openpose_kp[4],openpose_kp[12],openpose_kp[9],openpose_kp[13],openpose_kp[10],
               openpose_kp[14],openpose_kp[11]]) 
    
    coco_kp = np.transpose(coco_kp,(1,0,2))
    add_two = np.full((coco_kp.shape[0],coco_kp.shape[1],1),2) # add vis attribute to keypoints, assume all keypoints are visible
    coco_kp = np.append(coco_kp,add_two,axis=2)
    #print(coco_kp.shape)
    return coco_kp

def add_kpts_coco_annot(kpts, img_name):
    
    anno_new_entry = {'num_keypoints': 17, 'area': 18662.33285, 'iscrowd': 0, 'keypoints': [217.67664670658684, 173.056, 2, 0.0, 0.0, 0, 213.07784431137725, 167.936, 2, 0.0, 0.0, 0,
                                                                                       187.0179640718563, 165.888, 2, 199.2814371257485, 188.416, 2, 147.1616766467066, 223.232, 2, 208.4790419161677, 236.544, 2,
                                                                                       131.83233532934133, 276.48, 2, 219.20958083832338, 278.528, 2, 127.23353293413174, 335.872, 2, 196.21556886227546, 280.576, 2,
                                                                                       160.95808383233535, 292.86400000000003, 2, 237.6047904191617, 338.944, 2, 210.0119760479042, 350.208, 2, 298.92215568862275, 400.384, 2, 154.82634730538922, 415.744, 2],
                 'image_name': 6522700, 'bbox': [101.61820359281438, 124.06784, 246.31185628742517, 309.504], 'category_id': 1, 'image_id': 44198600}
    anno_new_entry['keypoints'] = kpts
    anno_new_entry['image_name'] = img_name
    anno_new_entry['image_id'] = int(img_name[:-4])
    
    bbox_x = min(kpts[::3])-15
    bbox_y = min(kpts[1::3])-15
    bbox_width = max(kpts[::3])-min(kpts[::3])+30
    bbox_height = max(kpts[1::3])-min(kpts[1::3])+30
    anno_new_entry['bbox'] = [bbox_x,bbox_y,bbox_width,bbox_height]
    
    
    json_file['annotations'].append(anno_new_entry)
    img_new_entry = {'file_name': '10.jpg', 'coco_url': 'http://images.cocodataset.org/train2017/000000000010.jpg',
                     'height': 512, 'width': 512, 'date_captured': '2013-11-17 05:24:06', 'flickr_url': 'http://farm3.staticflickr.com/2659/3767936916_b2a3b62925_z.jpg', 'id': 0}
    
    img_new_entry['coco_url'] = 'http://images.cocodataset.org/train2017/' + img_name
    img_new_entry['file_name'] = img_name
    img_new_entry['id'] = int(img_name[:-4])*10
    json_file['images'].append(img_new_entry)
    


if __name__ == "__main__":
    
    folderlist = os.listdir(root_dir+'videos')
    print(folderlist)    
    try:
        os.makedirs(root_dir+'frames')
    except:
        pass
    
    with open(r'COCO_sample.json') as ff:
        json_file = json.load(ff)
        json_file['annotations'] = []
        json_file['images'] = []
        total_frames = 0
        for folder in folderlist:
            video_name = root_dir+'videos/'+folder+'/RGBT_T.mp4'
            npz_name =  root_dir+'videos/'+folder+'/info.npz'
            cap = cv2.VideoCapture(video_name)
            output_dir = root_dir+'frames/'
            
            total_frames = save_frames_and_keypoints(video_name,cap,output_dir,total_frames)

    output_json_name = root_dir + str(date.today()) + '.json'
    with open(output_json_name,'w+') as outf:
        key = json.dumps(json_file)
        outf.write(str(key))



