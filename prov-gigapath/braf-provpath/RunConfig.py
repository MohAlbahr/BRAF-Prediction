import argparse
from datetime import datetime
import os
from zoneinfo import ZoneInfo

class RunConfig():
    # Dataset mapping: mapping dataset identifier (string) to parameters.
    # Format: [train_val_dataset, test_dataset, train_subset_fraction, val_subset_fraction, test_subset_fraction, subset_stratify, num_classes]
    datasets_mapping = {
        'tcga': ['tcga', 'tcga', 1.0, 1.0, 1.0, [3580306, 3500273], 2],
        'uke': ['uke', 'uke', 1.0, 1.0, 1.0, [586547, 419140], 2],
    }

    def __init__(self, root_dir: str, default_config_name: str):
        self.root_dir = root_dir

        # Parse command-line arguments.
        parser = argparse.ArgumentParser()
        parser.add_argument('--config_name', type=str, required=False)
        parser.add_argument('--run_number', type=int, required=False)
        parser.add_argument('--datasets', type=str, required=False, default="tcga",
                            help="Dataset identifier ('tcga' or 'uke')")
        args = parser.parse_args()

        # Set configuration name.
        self.config_name = args.config_name if args.config_name else default_config_name

        # Set run number.
        self.run_number = args.run_number if args.run_number else 1

        # Set dataset identifier (as string, e.g., 'tcga' or 'uke').
        self.datasets = args.datasets

        # Create a unique name for the run.
        self.run_config_name = f"{self.config_name}_ds:{self.datasets}_run:{self.run_number}_{datetime.now(ZoneInfo('Europe/Berlin')).strftime('%d.%m.%Y-%H:%M:%S')}"
        print('Using RunConfig "%s"' % self.get_name())

    def get_name(self):
        return self.run_config_name

    def get_root_dir(self):
        return self.root_dir

    def get_config(self):
        # Retrieve dataset parameters based on the dataset identifier.
        dataset_1, dataset_2, train_subset_fraction, val_subset_fraction, test_subset_fraction, subset_stratify, num_classes = RunConfig.datasets_mapping[self.datasets]

        # Set configuration parameters shared across experiments.
        config = {
            # Run parameters
            'run_number': self.run_number,

            # Dataset parameters
            'num_classes': num_classes,
            'train_val_dataset': dataset_1,
            'test_dataset': dataset_2,
            'train_subset_fraction': train_subset_fraction,
            'val_subset_fraction': val_subset_fraction,
            'test_subset_fraction': test_subset_fraction,
            'subset_stratify': subset_stratify,
            'prov_transforms': True,
            'image_size': 224,
            'hflip': False,
            'vflip': False,
            'color-jitter': False,
            'batch_size': 512,
            'workers': 8,

            # Training parameters
            'nb_epochs': 10,
            'iters-per-epoch': 500,
            'patience': 3,
            'print-freq': 1,

            # File handling
            'run_data_dir': os.path.join(self.root_dir, 'TrainData', self.run_config_name),
        }

        if self.config_name == 'prov_config':
            config = {**config, **{
                # Model parameters
                'architecture': 'prov-gigapath',
                # Training parameters specific to this configuration
                'phase': 'train',
                'finetune': True,
            }}
        else:
            raise NotImplementedError('RunConfig "%s" is not implemented yet.' % self.config_name)

        return config

if __name__ == '__main__':
    # Test the RunConfig class.
    config = RunConfig(root_dir='.', default_config_name='prov_config')
    print(config.get_config())
