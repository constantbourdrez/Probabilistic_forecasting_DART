import argparse
import yaml
import torch as th
import numpy as np
import geopandas as gpd
import pandas as pd
from lightning.pytorch.callbacks import EarlyStopping
import lightning as L
from diffusion_graph.diffusion_model import DART_STG
from lightning_module import LightningDART_STG
from preprocessing import Preprocessor_Graph  # Assuming this is where your Preprocessor_Graph is defined

def setup_seed(seed):
    import random
    th.manual_seed(seed)
    th.mps.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)
    th.use_deterministic_algorithms(True)

def main(config):
    setup_seed(2022) #Setting the seed
    # Load weather data
    weather_weighted = gpd.read_file(config['data_path'])
    weather_weighted['time'] = pd.to_datetime(weather_weighted['time'])

    print("Training data loaded")


    # Initialize preprocessor
    preproc = Preprocessor_Graph(
        batch_size=config['preprocessor']['batch_size'],
        feature_names=config['preprocessor']['feature_names'],
        ratio=config['preprocessor']['ratio'],
        shuffle=config['preprocessor']['shuffle'],
        historic_length=config['model']['historic_length'],
        horizon_pred=config['model']['horizon_pred']
    )
    train_loader, valid_loader, test_loader = preproc.make_pipe(weather_weighted)

    print("Data preprocessed")

    # Initialize model
    dart_stg = DART_STG(
        N=config['model']['N'],
        sample_steps=config['model']['sample_steps'],
        sample_strategy=config['model']['sample_strategy'],
        beta_start=config['model']['beta_start'],
        beta_end=config['model']['beta_end'],
        in_channels=config['model']['in_channels'],
        n_blocks=config['model']['n_blocks'],
        n_resolutions=config['model']['n_resolutions'],
        t_emb_dim=config['model']['t_emb_dim'],
        num_vertices=config['model']['num_vertices'],
        historic_length=config['model']['historic_length'],
        horizon_pred=config['model']['horizon_pred'],
        proj_dim=config['model']['proj_dim'],
        channel_multipliers=config['model']['channel_multipliers'],
        beta_schedule=config['model']['beta_schedule'],
        device=th.device(config['device']),
        dropout=config['model']['dropout']
    )

    # Initialize Lightning model
    lightning_dart = LightningDART_STG(DART_STG=dart_stg, preprocessor=preproc, mask_ratio = config['model']['mask_ratio'])

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
        log_every_n_steps=config['trainer']['log_every_n_steps']
        
    )
    print("Trainer initialized")


    # Train the model
    trainer.fit(model = lightning_dart, train_dataloaders=train_loader, val_dataloaders=valid_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument('--config', type=str, required=True, help="Path to the config file")

    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    main(config)
