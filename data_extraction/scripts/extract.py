#!/usr/bin/env python
import rospy
import rosbag

import cv2
import numpy as np
import os
import sys
import argparse

class DataExtractor():

    def __init__(self, save_dir, topics=['/auv/front_cam/image_color/compressed']):
        self.save_dir = save_dir
        self.topics = topics

    def extract(self, bag_file, start, end, interval=10, group='train'):
        rospy.loginfo("Extracting from %s: %ds to %ds every %ds", bag_file, start, end, interval)

        with rosbag.Bag(bag_file, 'r') as bag:
            bag_name = bag_file.split('.')[0]
            bag_name = bag_name.split('/')[-1]
            prev = None
            head = True
            for topic, msg, t in bag.read_messages():
                if head:
                    start += t.secs
                    end += t.secs
                    head = False

                if t.secs < start:
                    continue

                if t.secs > end:
                    break

               # print topic,  t.secs, start, end, topic in self.topics
                

                # check if is desired topic
                if topic in self.topics:

                    # save images some intervals apart
                    if prev is not None and t.secs - prev < interval:
                        #print prev, t.secs
                        continue
                    else:
                        prev = t.secs

                    #print 'hi'

                    cvimg = self.compressed_ros_to_cv2(msg)
                    
                    img_name = os.path.join(self.save_dir, group, bag_name + '-' + topic.replace('/', '_') + '-' + str(t.secs) + '.jpg')
                    print img_name 
                    cv2.imwrite(img_name, cvimg)
                    rospy.loginfo('saved to', img_name)
                    
                    

    def compressed_ros_to_cv2(self, img):
        try:
            np_arr = np.fromstring(img.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except CvBridgeError as e:
            rospy.logerr(e)
        return frame 

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('save_dir', help='directory to save extracted images')
#    p.add_argument('--topics', '-t', type=str, default=None)
   
    args = p.parse_args()

    extractor = DataExtractor(args.save_dir)

    bag = '/home/ruth/Documents/Bumblebee/bags/queenstown18/POOL-dice_all-NORMAL_2018-06-12-14-05-52.bag'
    start = 0
    end = 12
    interval = 2
    extractor.extract(bag, start, end, interval, 'train')
