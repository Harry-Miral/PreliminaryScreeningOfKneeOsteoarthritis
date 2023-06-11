import videopose_PSTMO
import npload
import angel

PSTMO_path = 'normal/example.mp4'# Provide your video path here
videopose_PSTMO.inference_video(PSTMO_path, 'alpha_pose')
npload.npload_point('example')# Please cooperate with npload.py to modify the storage path of your joint position time series data
angel.angel_all('example','normal/')# Your angular time series data goes here

