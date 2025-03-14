import os
import pandas as pd
import torch
import pickle
from torch.utils.data import DataLoader
from gigapath.classification_head import get_model
from datasets.slide_datatset import SlideDataset
from params import get_finetune_params
from task_configs.utils import load_task_config
from utils import seed_torch, get_exp_code, get_splits, get_loader, save_obj

def extract_slide_representations(model, dataloader, device, args):
    model.eval()
    fp16_scaler = None
    if args.fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()
        print('Using fp16 training')

    slide_representations = []

    with torch.no_grad():
        for batch in dataloader:
            images, img_coords, labels, slide_ids = batch['imgs'], batch['coords'], batch['labels'], batch['slide_id']
            images = images.to(device, non_blocking=True)
            img_coords = img_coords.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()

            with torch.cuda.amp.autocast(fp16_scaler is not None, dtype=torch.float16):

                    features = model.extract_features(images, img_coords)

                    
                    slide_representation = {
                        'slide_id': slide_ids[0],
                        'features': features.cpu().numpy(),
                        'label': labels.cpu().numpy()
                    }
                    slide_representations.append(slide_representation)

    return slide_representations

def main():
    args = get_finetune_params()
    print(args)

    # set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    # set the random seed
    seed_torch(device, args.seed)

    # load the task configuration
    print('Loading task configuration from: {}'.format(args.task_cfg_path))
    args.task_config = load_task_config(args.task_cfg_path)
    print(args.task_config)
    args.task = args.task_config.get('name', 'task')

    
    # set the experiment save directory
    args.save_dir = os.path.join(args.save_dir, args.task, args.exp_name)
    args.model_code, args.task_code, args.exp_code = get_exp_code(args) # get the experiment code
    args.save_dir = os.path.join(args.save_dir, args.exp_code)
    os.makedirs(args.save_dir, exist_ok=True)
    print('Experiment code: {}'.format(args.exp_code))
    print('Setting save directory: {}'.format(args.save_dir))

    # set the learning rate
    eff_batch_size = args.batch_size * args.gc
    if args.lr is None or args.lr < 0:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.gc)
    print("effective batch size: %d" % eff_batch_size)

    # set the split key
    if args.pat_strat:
       args.split_key = 'pat_id'
    else:
        args.split_key = 'slide_id'
    print("args.split_key: ", args.split_key)
    # set up the dataset
    args.split_dir = os.path.join(args.split_dir, args.task_code) if not args.pre_split_dir else args.pre_split_dir
    os.makedirs(args.split_dir, exist_ok=True)
    print('Setting split directory: {}'.format(args.split_dir))
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    
    model = get_model(**vars(args))

   # used for slide_representations
    model.load_state_dict(torch.load("../prov-gigapath/outputs/braf/braf/run-globalPool-unfreeze_traintype-UKE_epoch-5_blr-0.003_BS-32_wd-0.05_ld-0.95_drop-0.5_dropPR-0.1_feat-9/eval_pretrained_braf/fold_0/checkpoint.pt"))  # Load the saved model

    model = model.to(device)
    
    dataset = pd.read_csv(args.dataset_csv)
    DatasetClass = SlideDataset
    
    train_splits, val_splits, test_splits = get_splits(dataset, fold=args.folds, **vars(args))
    
    # instantiate the dataset
    train_data, val_data, test_data = DatasetClass(dataset, args.root_path, train_splits, args.task_config, split_key=args.split_key) \
                                    , DatasetClass(dataset, args.root_path, val_splits, args.task_config, split_key=args.split_key) if len(val_splits) > 0 else None \
                                    , DatasetClass(dataset, args.root_path, test_splits, args.task_config, split_key=args.split_key) if len(test_splits) > 0 else None

    # get the dataloader
    train_loader, val_loader, test_loader = get_loader(train_data, val_data, test_data, **vars(args))
    
    # TCGA Slide representations    
    tcga_train_slide_representations = extract_slide_representations(model, train_loader, device, args)
    tcga_val_slide_representations= extract_slide_representations(model, val_loader, device, args)
    
    with open(os.path.join(args.save_dir, 'tcga_train_slide_representations.pkl'), 'wb') as f:
        pickle.dump(tcga_train_slide_representations, f)
    
    with open(os.path.join(args.save_dir, 'tcga_val_slide_representations.pkl'), 'wb') as f:
        pickle.dump(tcga_val_slide_representations, f)
    
    print("TCGA Slide representations saved.")

    # UKE Slide representations
    uke_slide_representations= extract_slide_representations(model, test_loader, device, args)
    
    with open(os.path.join(args.save_dir, 'uke_slide_representations.pkl'), 'wb') as f:
        pickle.dump(uke_slide_representations, f)
    
    print("UKE Slide representations saved.")

if __name__ == '__main__':
    main()

