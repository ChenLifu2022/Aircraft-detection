# Aircraft-detection

## Introduction

#### paper MGCAM: https://ieeexplore.ieee.org/document/9741706

  Multiscale geospatial data analytics are implemented
  as DNNs to achieve effective feature extraction in air-
  craft detection. In SAR images, the scale of aircraft
  varies greatly due to the difference in image resolu-
  tions and sizes of aircraft, which often hinders the
  feature extraction of aircraft detection. In this article,
  the feature PyConv is employed to build the mul-
  tiscale feature learning module—CSPP convolution—
  and the backbone network pyramid cross-stage partial
  darknet (PyCSPDarknet), which integrates CSPP and
  CSPDarknet (the backbone network of YOLOV5 to
  achieve multiscale feature extraction of aircraft in SAR
  image analytics.

#### paper EBPA2N: https://www.mdpi.com/2072-4292/13/15/2940

  An effective and efficient aircraft detection network EBPA2N is proposed for SAR
  image analytics. Combined with the sliding window detection method, an end-to-end
  aircraft detection framework based on EBPA2N was established, which offers accurate
  and real-time aircraft detection from large-scale SAR images.


## Installation

    Ubuntu18.04
    matplotlib>=3.2.2
    numpy>=1.18.5
    opencv-python>=4.1.2
    pillow
    PyYAML>=5.3
    scipy>=1.4.1
    tensorboard>=2.2
    torch>=1.6.0
    torchvision>=0.7.0
    tqdm>=4.41.0
    
## Train

#### 1. Transform data：Use the yolov5 dataset，you can modify data\zao_airplane.yaml
    train: /home/aminj/yolov5-master1/data/zao_airplane/images/train2014          
    val:  /home/aminj/yolov5-master1/data/zao_airplane/images/val2014               

#### 2. Select the model：line 395 of the train.py(MGCAM or EBPA2N)
    parser.add_argument('--cfg', type=str, default='MGCAM.yaml', help='model.yaml path')
    
#### 3.Train：you can set parameters in train.py(data,epochs,batch-size...)
    parser.add_argument('--data', type=str, default='data/zao_airplane.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[512, 512], help='[train, test] image sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--device', default='0，1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', default=False, help='train as single-class dataset')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--logdir', type=str, default='runs/', help='logging directory')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
