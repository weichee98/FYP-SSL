import os.path as osp

__dir__ =  osp.dirname(osp.dirname(osp.abspath(__file__)))
EXPERIMENT_DIR = osp.join(__dir__, "experiments")