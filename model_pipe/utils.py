import torch as th
import yaml
from preprocessing import Preprocessor_Graph
from metrics import *
from lightning_module import LightningDART_STG
from diffusion_graph.diffusion_model import DART_STG


def load_model_from_previous_state(config_path, checkpoint_path, hparams_path):
    #Load config file
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    # Custom constructor for Preprocessor_Graph
    def preprocessor_graph_constructor(loader, node):
        return Preprocessor_Graph()

    # Register the constructor
    yaml.add_constructor('tag:yaml.org,2002:python/object:preprocessing.Preprocessor_Graph', preprocessor_graph_constructor)


    # Manually load and parse the YAML file
    with open(hparams_path, 'r') as file:
        hparams_raw = yaml.load(file, Loader=yaml.FullLoader)

    # Replace the custom tag with an instance of Preprocessor_Graph
    for key, value in hparams_raw.items():
        if isinstance(value, str) and value.startswith('tag:yaml.org,2002:python/object:preprocessing.Preprocessor_Graph'):
            print(key)
            hparams_raw[key] = Preprocessor_Graph()

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
        dropout = config['model']['dropout']
    )


    # Load the model from the checkpoint (initially on CPU)
    model = LightningDART_STG.load_from_checkpoint(
        checkpoint_path,
        map_location='cpu',  # Load to CPU first
        **hparams_raw,
        DART_STG = dart_stg# Pass the hparams dictionary directly
    )

    # Function to recursively move all tensors in the model to 'mps' and convert to float32
    def convert_model_to_device_and_dtype(model, device, dtype):
        for param in model.parameters():
            param.data = param.data.to(device=device, dtype=dtype)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device=device, dtype=dtype)
        for buffer in model.buffers():
            buffer.data = buffer.data.to(device=device, dtype=dtype)
        return model

    # Convert the model to 'mps' device and ensure all tensors are float32
    model = convert_model_to_device_and_dtype(model, device='mps', dtype=th.float32)

    # Move the model to 'mps' device if not done inside the convert_model_to_device_and_dtype function
    return model.to('mps')
