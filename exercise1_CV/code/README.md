# Installation guide

1. Download dataset from [here](https://lmb.informatik.uni-freiburg.de/data/DL_Labcourse/coco_subset.zip)  (~870 MB)
2. Install dependencies
3. Adapt path to dataset ("COCO_PATH") in model/data.py and  in model/data_seg.py
4. Verify correct dataset setup by visually inspecting samples of the dataset with

    python3 show_data.py
    python3 show_data_seg.py
    
5. Get the pretrained model from [here](https://lmb.informatik.uni-freiburg.de/data/DL_Labcourse/trained_net.model) (~12MB)
6. Visualize predictions of the model with

    python3 run_forward.py

# Dependencies

    pip install pytorch tqdm numpy Pillow
    
