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
from tqdm import tqdm

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

    if args.mode == 'train':
        loader = train_loader
    elif args.mode == 'valid':
        loader = valid_loader
    elif args.mode == 'test':
        loader = test_loader
    else:
        raise ValueError("Mode not recognized. Choose between 'train', 'valid' or 'test'")

    model.data_mean = preproc.data_mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).type(th.float32).to(th.device('mps'))
    model.data_std = preproc.data_std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).type(th.float32).to(th.device('mps'))

    print("Data preprocessed")

    path = args.chk_path.split('/')[:-2]
    path = '/'.join(path)

    model.eval()
    final_pred = th.tensor([])
    with th.no_grad():
        for batch in tqdm(loader):
            batch = batch.to(th.device('mps'))
            pred = model(batch, 0, n_samples = 23).cpu().detach()
            final_pred = th.cat([final_pred, pred], dim=0)

    th.save(final_pred, 'final_pred.pt')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument('--config', type=str, required=True, help="Path to the config file")
    parser.add_argument('--chk_path', type=str, required=True, help="Path to the checkpoint file")
    parser.add_argument('--hparams_path', type=str, required=True, help="Path to the hparams file")
    parser.add_argument('--mode', type=str, required=True, help="Mode to make predictions. Choose between 'train', 'valid' or 'test'")

    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    main(config, args)
