import argparse
import yaml
import torch as th
import geopandas as gpd
import pandas as pd
from lightning.pytorch.callbacks import EarlyStopping
import lightning as L
from preprocessing import Preprocessor_Graph  # Assuming this is where your Preprocessor_Graph is defined
from metrics import *
from utils import *
from train import setup_seed



def main(config, args):
    setup_seed(2022) #Setting the seed
    # Load weather data
    weather_weighted = gpd.read_file(config['data_path'])
    weather_weighted['time'] = pd.to_datetime(weather_weighted['time'])

    print("Data loaded")



    model = load_model_from_previous_state(config_path= args.config, checkpoint_path= args.chk_path , hparams_path= args.hparams_path)

    preproc = Preprocessor_Graph(
            batch_size= config['preprocessor']['batch_size'],
            feature_names=config['preprocessor']['feature_names'],
            ratio=config['preprocessor']['ratio'],
            shuffle=config['preprocessor']['shuffle'],
            historic_length=config['model']['historic_length'],
            horizon_pred=config['model']['horizon_pred']

        )
    train_loader, valid_loader, test_loader = preproc.make_pipe(weather_weighted)

    model.data_mean = preproc.data_mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).type(th.float32).to(th.device('mps'))
    model.data_std = preproc.data_std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).type(th.float32).to(th.device('mps'))

    print("Data preprocessed")

    path = args.chk_path.split('/')[:-2]
    path = '/'.join(path)

    # Define early stopping callback
    early_stop_callback = EarlyStopping(
        monitor=config['early_stopping']['monitor'],
        patience=config['early_stopping']['patience'],
        verbose=config['early_stopping']['verbose'],
        mode=config['early_stopping']['mode']
    )

    # Initialize trainer
    trainer = L.Trainer(
        max_epochs=config['trainer']['max_epochs'],
        callbacks=[early_stop_callback],
        log_every_n_steps=config['trainer']['log_every_n_steps'],
        default_root_dir = path
    )
    print('Trainer initialized')

    results = []

    for sample_step in args.sample_steps:
        for beta_schedule in args.beta_schedules:
            for num_samples in args.num_samples:
                # Set the hyperparameters in the config
                model.model.set_ddim_sample_steps(sample_step)
                model.model.beta_schedule = beta_schedule
                model.num_samples = num_samples
                print(f"Testing with sample_step={sample_step}, beta_schedule={beta_schedule}, num_samples={num_samples}")

                # Test the model
                test_results = trainer.test(model = model, dataloaders= test_loader, ckpt_path= args.chk_path)
                test_metrics = test_results[0]

                th.mps.empty_cache()

                # Store the results
                results.append({
                    'sample_step': sample_step,
                    'beta_schedule': beta_schedule,
                    'num_samples': num_samples,
                    **test_metrics
                })

    # Convert results to a DataFrame and save to CSV
    results_df = pd.DataFrame(results)
    save_path = f'{args.chk_path}_hyperparameter_tuning_results.csv'
    results_df.to_csv(save_path, index=False)
    print(f"Results saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument('--config', type=str, required=True, help="Path to the config file")
    parser.add_argument('--sample_steps', type=int, nargs='+', required=True, help="List of sample steps")
    parser.add_argument('--beta_schedules', type=str, nargs='+', required=True, help="List of beta schedules")
    parser.add_argument('--num_samples', type=int, nargs='+', required=True, help="List of number of samples")
    parser.add_argument('--chk_path', type=str, required=True, help="Path to the checkpoint file")
    parser.add_argument('--hparams_path', type=str, required=True, help="Path to the hparams file")

    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    main(config, args)
