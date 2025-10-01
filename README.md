# Preliminary screening of knee osteoarthritis

## Setup
_Instructions refer to Windows-based systems

Please download this compressed package before using https://drive.google.com/drive/folders/1nATFeytb5y11dNfpT_6nRA3pdo8sv6ux?usp=sharing .

Please unzip `yolov3-spp.weights` to `joints_detectors\Alphapose\models\yolo`.

Please decompress `duc_se.pth` to `joints_detectors\Alphapose\models\sppe`.

Please extract `Dataset.rar` to `Dataset`
Please put `pretrained_h36m_detectron_coco.bin`, `PSTMO_no_refine_6_4215_h36m_cpn.pth`,
`PSTMO_no_refine_11_4288_h36m_cpn.pth`, `PSTMO_refine_6_4215_h36m_cpn.pth`,
`PSTMOS_no_refine_15_2936_h36m_gt.pth`, `PSTMOS_no_refine_28_4306_h36m_cpn.pth`,
`PSTMOS_no_refine_48_5137_in_the_wild.pth`, `PSTMOS_no_refine_50_3203_3dhp.pth` are decompressed under `checkpoint`

## Example 

If you want to extract 3D bone information from your video, you can refer to this example.
Please open `run.py` and change `VIDEO_path` to the local path of your video. Open `npload.py`, `savepath` will be the storage location of your human key point time series data, you can modify it according to your actual situation. Open `angel.py`, `savepath` will be the storage location of your angle time series data, you can modify it according to your actual situation, you can also increase or decrease the code to select any joint angle time series data you want.
Run `run.py`, you can find your result data in the above path in turn, the data including all key points of the human body is stored under `/outputs` by default, you can modify the path in videopose_PSTMO.py according to your actual situation to change the save path.


## Run the model

Please make sure you extract `Dataset.rar` to `/Dataset`.
Open any `.ipynb` file you want to run, confirm that there is no problem with the path of the dataset, and run it to get the result.

## Qualitative analysis and quantitative analysis

First locate the `DWT analysis` directory, download the compressed package from Google Cloud Disk and extract it to the directory, and run the .py file to obtain the analysis report.

`https://drive.google.com/file/d/1xosX7hiPJllVZb8KQPXVCXbUYWxQabyl/view?usp=sharing`

## Remark

we use `python3.9`,`torch1.9.1`,`torchvision0.10.1`.
For the model, we used `tensorflow2.12.0`.

