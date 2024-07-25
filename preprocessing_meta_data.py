import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import pickle




def preprocess_file(input_file, output_folder, districts):
    # Load the file
    data = pd.read_csv(input_file)


    # Preprocess the data (example: fill NaN values with 0)
    data_HCMC = data[data.country == 'VNM']
    data_HCMC = data_HCMC[data_HCMC.polygon1_name.isin(districts)]
    data_HCMC = data_HCMC[data_HCMC.polygon2_name.isin(districts)]
    data_HCMC = data_HCMC[data_HCMC.is_home_tile_colocation == False]
    # Get the file name without the folder path
    file_name = os.path.basename(input_file)

    # Define the output file path
    output_file = os.path.join(output_folder, file_name)



    # Save the preprocessed data to the output folder
    data_HCMC.to_csv(output_file, index=False)

def process_files_in_parallel(input_folder, output_folder, districts, use_threads=False, max_workers=None):
    # Get a list of all files in the input folder
    files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]

    #Executor = ThreadPoolExecutor if use_threads else ProcessPoolExecutor

    # Use ProcessPoolExecutor to parallelize the preprocessing
   #with Executor(max_workers=max_workers) as executor:
        # # Submit all tasks to the executor
        # futures = [executor.submit(preprocess_file, file, output_folder, districts) for file in files]

        # # Ensure all tasks are completed
        # for future in futures:
        #     future.result()

    for file in files:
        preprocess_file(file, output_folder, districts)

if __name__ == "__main__":

    # List all mounted volumes
    volumes = os.listdir('/Volumes')
    print("Mounted Volumes:")
    for volume in volumes:
        print(volume)

    # Define the volume name to navigate to
    volume_name = "New Volume/Colocation Maps"  # Change this to your volume name

    # Construct the path to the volume
    volume_path = os.path.join('/Volumes', volume_name)# Define the volume name to navigate to

    volume_name_2 = "Macintosh HD/Users/constouille/Documents/GitHub/DART/data/data_for_good/preprocessed"  # Change this to your volume name

    # Construct the path to the volume
    volume_path_2 = os.path.join('/Volumes', volume_name_2)

    input_folder = volume_path   # Change this to your input folder path
    output_folder = volume_path_2
    with open('data/dict_preproc.pkl', 'rb') as file:
        districts = pickle.load(file)


    # Check if the volume exists
    if os.path.exists(volume_path):
        # Change directory to the volume
        os.chdir(volume_path)

        # List files in the volume
        files = os.listdir('.')
        print(f"\nFiles in {volume_name}:")
        for file in files:
            print(file)
    else:
        print(f"Volume '{volume_name}' not found.")


    # Use threads if I/O-bound, otherwise use processes
    use_threads = False  # Set to True for I/O-bound tasks, False for CPU-bound
    max_workers = 16  # Set to None to use the default number of workers, or specify an integer


    process_files_in_parallel(input_folder, output_folder, districts, use_threads=use_threads, max_workers=max_workers)



## TO MOUNT HARD DRIVE COPY PASTE THAT: ########


# # List all mounted volumes
# volumes = os.listdir('/Volumes')
# print("Mounted Volumes:")
# for volume in volumes:
#     print(volume)

# # Define the volume name to navigate to
# volume_name = "New Volume"  # Change this to your volume name

# # Construct the path to the volume
# volume_path = os.path.join('/Volumes', volume_name)

# # Check if the volume exists
# if os.path.exists(volume_path):
#     # Change directory to the volume
#     os.chdir(volume_path)

#     # List files in the volume
#     files = os.listdir('.')
#     print(f"\nFiles in {volume_name}:")
#     for file in files:
#         print(file)
# else:
#     print(f"Volume '{volume_name}' not found.")
