import videopose_PSTMO
import npload
import angel


delete_Patient_data=[30,31,35,53,60,64,65]#[11,30,31,35,53,60,64,65,73,101]
delete_normal_data=[15,47,49,54,57,58,72,73,74,77,78,90,99,100,101,103,106,111,112,118,122,123,142,147,167,227]
pan = 1



PSTMO_path = 'normal/zmk gait.mp4'# Provide your video path here
videopose_PSTMO.inference_video(PSTMO_path, 'alpha_pose')
npload.npload_point('zmk gait')# Please cooperate with npload.py to modify the storage path of your joint position time series data
angel.angel_all('zmk gait','normal/')# Your angular time series data goes here

