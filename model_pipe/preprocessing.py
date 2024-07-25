import torch as th
import geopandas as gpd
import pandas as pd
import argparse
import random
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch_geometric_temporal.signal import StaticGraphTemporalSignal
from torch_geometric_temporal.signal import temporal_signal_split
from torch_geometric.loader import DataLoader
import pickle


class Preprocessor:

    def __init__(self, batch_size=None, df=None, feature_names=None, shifts=None, ratio=None, shuffle=True, num_elements=None):
        """
        Initializes the Preprocessor with the specified parameters.

        Args:
        batch_size (int, optional): The size of each batch for splitting tensors.
        df (pandas.DataFrame, optional): Input dataframe for conversion to tensor.
        feature_names (list, optional): List of feature names to include in the tensor.
        shifts (dict, optional): Dictionary of feature shifts for the shift_features method.
        ratio (float, optional): Ratio for splitting tensor into training and validation sets.
        shuffle (bool, optional): Whether to shuffle the tensor before splitting.
        num_elements (int, optional): Number of elements for generating a random dictionary.
        """
        self.batch_size = batch_size
        self.df = df
        self.feature_names = feature_names
        self.shifts = shifts
        self.ratio = ratio
        self.shuffle = shuffle
        self.num_elements = num_elements

    def split_batch(self, tensor):
        """
        Splits a tensor into batches of the specified size.

        Args:
        tensor (torch.Tensor): Input tensor of shape (batch, length, features).

        Returns:
        torch.Tensor: Tensor split into batches of the specified size.
        """
        shape = tensor.shape
        number_of_batch = shape[1] // self.batch_size
        rest = shape[1] % self.batch_size
        print(f"{rest} items not taken")
        return tensor.contiguous()[:, :number_of_batch * self.batch_size, :].reshape(shape[0], number_of_batch, self.batch_size, shape[2])

    def to_tensor(self):
        """
        Converts the dataframe to a tensor based on the specified feature names.

        Returns:
        torch.Tensor: Tensor representation of the dataframe with the specified features.
        """
        # Get unique districts
        districts = self.df.District.unique()

        # Sort dataframe by time
        self.df = self.df.sort_values('time')

        # List to collect tensors
        tensor_list = []

        # Iterate through districts and collect tensors
        for district in districts:
            district_df = self.df.loc[self.df.District == district, self.feature_names]
            inputs = district_df.values
            input_tensor = th.tensor(inputs, dtype=th.float32)
            input_tensor = input_tensor.unsqueeze(0)
            tensor_list.append(input_tensor)

        # Concatenate all tensors in the list
        tensor = th.cat(tensor_list, dim=0)

        return tensor

    def shift_features(self, tensor):
        """
        Shifts the elements of each feature in the tensor to the left according to the specified shifts.

        Args:
        tensor (torch.Tensor): Input tensor of shape (batch, length, features).

        Returns:
        torch.Tensor: Tensor with features shifted right according to the specified shifts and homogeneous length.
        """
        # Get the original shape of the tensor
        district, length, num_features = tensor.shape

        # Determine the maximum shift
        max_shift = max(self.shifts.values())

        # Create a new tensor with adjusted length
        new_length = length - max_shift
        shifted_tensor = th.zeros((district, new_length, num_features), dtype=tensor.dtype, device=tensor.device)

        for feature, shift in self.shifts.items():
            # Slice the tensor for the current feature and shift right
            shifted_tensor[:, :, feature] = tensor[:, :new_length, feature]
            if shift > 0:
                shifted_tensor[:, shift:, feature] = tensor[:, :new_length - shift, feature]

        shifted_tensor = shifted_tensor[:, max_shift:, :]


        return shifted_tensor


    def split_train_val(self, tensor):
        """
        Splits a tensor into training and validation sets based on the specified ratio.

        Args:
        tensor (torch.Tensor): Input tensor.

        Returns:
        tuple: Training and validation DataLoader objects.
        """
        # Determine the number of samples for each set
        split_idx = int(tensor.size(1) * self.ratio)

        # Shuffle the tensor if shuffle is True
        if self.shuffle:
            print('Shuffling tensor')
            perm = th.randperm(tensor.size(1))
            tensor = tensor[perm]


        # Split the tensor into train and val sets
        train_tensor = tensor[:, :split_idx, :, :]
        val_tensor = tensor[:, split_idx:, :, :]

        train_inputs, train_labels = train_tensor[:, :, :-1], train_tensor[:, :, :, -1]
        val_inputs, val_labels = val_tensor[:, :, :-1], val_tensor[:, :, :, -1]

        # Create datasets and dataloaders
        train_dataset = TensorDataset(train_inputs, train_labels)
        val_dataset = TensorDataset(val_inputs, val_labels)

        # Define batch size (you can adjust it as needed)
        batch_size = self.batch_size if self.batch_size else 64

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader

    def generate_random_dict(self):
        """
        Generates a dictionary with consecutive integers as keys and random integers
        between 1 and 30 as values.

        Returns:
        dict: Dictionary with consecutive integer keys and random integer values.
        """
        return {i: random.randint(1, 30) for i in range(0, self.num_elements)}

class Preprocessor_Graph:
    def __init__(self,  batch_size = None, feature_names = None, ratio=None, shuffle=True, historic_length = 40, horizon_pred= 20):
        """
        Initializes the Preprocessor with the specified parameters.

        Args:
            batch_size (int, optional): The size of each batch for splitting tensors.
            temporal_data (pandas.DataFrame, optional): Input dataframe containing temporal data for conversion to tensor.
            graph_data (pandas.DataFrame, optional): Input dataframe containing graph data.
            feature_names (list, optional): List of feature names to include in the tensor.
            shifts (dict, optional): Dictionary of feature shifts for the shift_features method.
            ratio (float, optional): Ratio for splitting tensor into training and validation sets.
            shuffle (bool, optional): Whether to shuffle the tensor before splitting.
            num_elements (int, optional): Number of elements for generating a random dictionary.
        """

        self.feature_names = feature_names
        self.ratio = ratio
        self.shuffle = shuffle
        self.n_lags = historic_length
        self.horizon_pred = horizon_pred
        self.batch_size = batch_size
        self.data_mean, self.data_std = th.tensor(0), th.tensor(0)

    def Standardize(self, temporal_data):
        """
        Standardizes the input data.

        Args:
        temporal_data (pandas.DataFrame): Input dataframe containing temporal data.

        Returns:
        pandas.DataFrame: Standardized dataframe.
        """
        # Standardize the input data
        scaler = StandardScaler()
        temporal_data[self.feature_names + ['n_patient']] = scaler.fit_transform(temporal_data[self.feature_names + ['n_patient']])
        self.scaler = scaler
        self.data_mean = th.tensor(scaler.mean_)
        self.data_std = th.tensor(scaler.var_**0.5)
        return temporal_data

    def create_arrays_for_dataset(self, temporal_data):
        """
        Creates arrays for the dataset from the temporal and graph data.

        Args:
        temporal_data (pandas.DataFrame): Input dataframe containing temporal data.
        graph_data (pandas.DataFrame): Input dataframe containing graph data.

        Returns:
        tuple: Arrays for the dataset.
        """
        # Get unique districts
        districts = temporal_data.District.unique()

        # Sort dataframe by time
        temporal_data = temporal_data.sort_values('time').fillna(0)

        # Initialize lists to store features and targets
        all_features_hist = []
        all_targets_hist = []
        all_features_pred = []
        all_targets_pred = []


        # Iterate through districts and collect tensors
        for district in districts:
            # Historical data (both features and targets)
            historical_features = temporal_data.loc[(temporal_data.District == district) & (temporal_data.method == 'hist'), self.feature_names + ['n_patient']].values
            historical_targets = temporal_data.loc[(temporal_data.District == district) & (temporal_data.method == 'hist'), self.feature_names + ['n_patient']].values

            # Prediction data (features and targets)
            prediction_features = temporal_data.loc[(temporal_data.District == district) & (temporal_data.method == 'pred'), self.feature_names + ['n_patient']].values
            prediction_targets = temporal_data.loc[(temporal_data.District == district) & (temporal_data.method == 'pred'), self.feature_names + ['n_patient']].values


            # Append to lists
            all_features_hist.append(historical_features)
            all_targets_hist.append(historical_targets)
            all_features_pred.append(prediction_features)
            all_targets_pred.append(prediction_targets)

        return np.array(all_features_hist), np.array(all_targets_hist), np.array(all_features_pred), np.array(all_targets_pred)


    def split_arrays_in_batches(self, *arrays):
        """
        Splits the arrays into batches of the specified size.

        Args:
        arrays (tuple): Arrays to split into batches.
        batch_size (int): Size of each batch.

        Returns:
        tuple: Arrays split into batches of the specified size.

        """
        all_features_hist, all_targets_hist, all_features_pred, all_targets_pred  = arrays[0], arrays[1], arrays[2], arrays[3]



        features_batched_h  = np.array([all_features_hist[:, i : i + self.n_lags, :].T
                    for i in range(all_features_hist.shape[1] - self.n_lags - self.horizon_pred - 1)
                ])


        features_batched_p  = np.array([all_features_pred[:, i : i + self.n_lags, :].T
                    for i in range(all_features_pred.shape[1] - self.n_lags - self.horizon_pred - 1)
                ])

        features_batched = np.concatenate((features_batched_h, features_batched_p), axis = 0)


        features_batched = list(np.expand_dims(np.transpose(np.array(features_batched), (0,3,1,2)), axis = 1))


        targets_batched_h = np.array([all_targets_hist[:, i +self.n_lags + 1 : i + self.n_lags + self.horizon_pred + 1].T
                    for i in range(all_targets_hist.shape[1] - self.n_lags - self.horizon_pred - 1)
                ])

        targets_batched_p = np.array([all_targets_pred[:, i +self.n_lags + 1 : i + self.n_lags + self.horizon_pred + 1].T
                    for i in range(all_targets_pred.shape[1] - self.n_lags - self.horizon_pred - 1)
                ])

        targets_batched = np.concatenate((targets_batched_h, targets_batched_p), axis = 0)

        targets_batched = list(np.expand_dims(np.transpose(np.array(targets_batched), (0,3,1,2)), axis = 1))


        #targets_batched = list(np.expand_dims(np.array(targets_batched), axis=1))

        return features_batched, targets_batched

    def retrieve_graph_data(self, graph_data):
        """
        Retrieves graph data from the input list.

        Args:
        graph_data (Iterable of graphs): Input dataframe containing graph data.

        Returns:
        tuple: Graph data in the format required for the dataset.
        """
        if graph_data is  None:
            # Specify the file path where you want to save the graph
            file_path = "stationnary_graph.pkl"

            # Open the file in binary write mode
            with open(file_path, "rb") as f:
                # Dump the graph object into the file
                G = pickle.load(f)


        else :

            # Get unique districts
            districts = graph_data.District.unique()

            # Initialize lists to store features and targets
            all_features = []

            # Iterate through districts and collect tensors
            for district in districts:
                graph_inputs = graph_data.loc[graph_data.District == district, feature_names].drop(columns=['time']).values
                all_features.append(graph_inputs)

        edges = list(G.edges)
        weights = [G[u][v]['weight'] for u, v in edges]
        nodes = list(G.nodes)
        node_to_index = {node: index for index, node in enumerate(nodes)}
        edge_index = np.array([[node_to_index[edge[0]], node_to_index[edge[1]]] for edge in edges], dtype=np.int64).T
        edge_weight = np.array(weights, dtype=np.float32)
        edge_index = np.expand_dims(np.array(edge_index), axis= 2)
        edge_weight = np.expand_dims(np.array(edge_weight), axis=0)

        return edge_index, edge_weight

    def make_pipe(self, temporal_data, graph_data = None, verbose = True):
        """
        Constructs a data processing pipeline.

        This method prepares temporal and graph data for model training by performing the following steps:
        1. Standardizes the temporal data.
        2. Converts the standardized temporal data into arrays suitable for dataset creation.
        3. Splits the arrays into batches.
        4. Retrieves graph data if provided.
        5. Creates a dataset by combining the graph data with the batched features and targets.
        6. Splits the temporal dataset into training and testing datasets.
        7. Optionally prints dataset information if verbose mode is enabled.

        Args:
            temporal_data: The temporal data to be processed.
            graph_data (optional): Graph data, if available.
            verbose (bool, optional): A flag indicating whether to print debug information. Defaults to True.

        Returns:
            tuple: A tuple containing two data loaders, one for the training set and the other for the testing set.
        """

        standardized_temporal_data = self.Standardize(temporal_data)

        feature_hist, target_hist, feature_pred, target_pred = self.create_arrays_for_dataset(standardized_temporal_data)

        features_batched, targets_batched = self.split_arrays_in_batches(feature_hist, target_hist, feature_pred, target_pred)

        edge_index, edge_weight = self.retrieve_graph_data(graph_data)

        dataset = StaticGraphTemporalSignal(edge_index=edge_index,
                                            edge_weight=edge_weight,
                                            features=features_batched,
                                            targets=targets_batched)

        train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio= self.ratio)
        train_dataset, valid_dataset = temporal_signal_split(train_dataset, train_ratio= 0.8)

        if verbose:
            print("Dataset type:  ", train_dataset)
            print("Number of samples / sequences: ",  len(list(train_dataset)))
            print(next(iter(train_dataset))) # Show first sample

        train_loader = DataLoader(list(train_dataset), batch_size = self.batch_size, shuffle = self.shuffle, num_workers=7, persistent_workers=True)
        valid_loader = DataLoader(list(valid_dataset), batch_size = self.batch_size, shuffle= False, num_workers=7, persistent_workers=True)
        test_loader = DataLoader(list(test_dataset), batch_size = self.batch_size, shuffle=False, num_workers=7, persistent_workers=True)

        return train_loader, valid_loader, test_loader





def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process climate data.")
    parser.add_argument("--filename", type=str, default='data/weighted_temps.shp', help="Path to the input data file (default: 'data/weighted_temps.shp')")
    parser.add_argument("--features", nargs="+", type=str, default=['t2m', 'rh', 'sh', 'ws', 'tp', 'n_patient'], help="List of feature names to select (default: ['t2m', 'rh', 'sh', 'ws', 'tp', 'n_patient])")
    parser.add_argument("--batch_size", type=int, default=7, help="Size of each batch for splitting tensors (default: 7)")
    parser.add_argument("--shifts", nargs="+", type=dict, default=None, help="List of feature shifts for the shift_features method (default: None)")
    parser.add_argument("--ratio", type=float, default=0.8, help="Ratio for splitting tensor into training and validation sets (default: 0.8)")
    parser.add_argument("--shuffle", action="store_true", help="Whether to shuffle the tensor before splitting (default: False)")
    args = parser.parse_args()

    print("All arguments loaded")

    # Load data from file
    data = gpd.read_file(args.filename)
    data['time'] = pd.to_datetime(data['time'])

    print("Data loaded")

    # Determine the number of elements (features)
    num_elements = len(args.features)

    # Initialize the Preprocessor with parsed arguments
    preprocessor = Preprocessor(df=data, feature_names=args.features, batch_size=args.batch_size, num_elements=num_elements, shifts=args.shifts, ratio=args.ratio, shuffle=args.shuffle)

    print("Preprocessor initialized")

    # If shifts are not provided, generate random shifts
    if args.shifts is None:
        preprocessor.shifts = preprocessor.generate_random_dict()

    # Convert dataframe to tensor
    tensor = preprocessor.to_tensor()


    print("Data converted to tensor")

    # Shift features in the tensor
    shifted_tensor = preprocessor.shift_features(tensor)

    print("Features shifted")

    # Split tensor into batches
    batch_tensor = preprocessor.split_batch(shifted_tensor)

    print("Data split into batches")


    # Split tensor into training and validation sets
    train_loader, val_loader = preprocessor.split_train_val(batch_tensor)

    print("Data split into training and validation sets")

    # Save preprocessed data to file
    th.save(train_loader, 'train_loader.pth')
    th.save(val_loader, 'val_loader.pth')

if __name__ == "__main__":
    main()
