import videopose_PSTMO
import npload
import angel

PSTMO_path = 'normal/zmk gait.mp4'# Provide your video path here
videopose_PSTMO.inference_video(PSTMO_path, 'alpha_pose')
npload.npload_point('zmk gait')# Please cooperate with npload.py to modify the storage path of your joint position time series data
angel.angel_all('zmk gait','normal/')# Your angular time series data goes here

