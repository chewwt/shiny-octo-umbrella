# ML workflow for ROS

1. `rosrun data_extraction extract.py ${save_dir}` to convert a bag file into jpg images. Edit the code to change bag file and timings :(

2. split the data into 3 groups, training, validation and test. data taken from the same day and/or time, must be in the same group

3. get bbox label tool and annotate

4. use convert to change labels to tfrecord



