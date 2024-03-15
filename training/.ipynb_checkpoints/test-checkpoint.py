# Libraries
import torch
import time
from utils.dataset import to_temporal, use_prediction, get_real_rollout

def rollout_test(model, dataset, **temporal_test_dataset_parameters):
    '''
    Function that tests a model and returns the rollout prediction
    ------
    model: nn.Model
        e.g., GNN model
    dataset: torch_geometric.data.data.Data
        Data object as defined in create_dataset.py, contains full simulation
    temporal_dataset_parameters: dict
        parameters used to create the temporal dataset (check to_temporal function)
    '''
    model.eval()
    previous_t=model.previous_t
    time_start = temporal_test_dataset_parameters['time_start']
    time_stop = temporal_test_dataset_parameters['time_stop']

    temporal_data = to_temporal(dataset, previous_t=previous_t, rollout_steps=-1, 
                                **temporal_test_dataset_parameters)
    temp = temporal_data[0]
    assert temp.time == time_start, f"You're starting at time {temp.time} instead of time {time_start}"
    temp.y = 0 # discard real data just to be sure not to copy it at some point
    
    if time_stop == -1:
        final_step = dataset.WD.shape[-1]
    else:
        final_step = time_stop
        
    rollout = []
    with torch.no_grad():
        for time_step in range(time_start+1,final_step):
            pred = model(temp)
            temp.x = use_prediction(temp.x, pred, previous_t)
            rollout.append(pred)

    rollout = torch.stack(rollout, -1)
    
    return rollout

def get_rollouts(model, dataset, **temporal_test_dataset_parameters):
    time_start = temporal_test_dataset_parameters['time_start']
    time_stop = temporal_test_dataset_parameters['time_stop']
    device = dataset.x.device

    start_time = time.time()
    predicted_rollout = rollout_test(model, dataset, **temporal_test_dataset_parameters).to(device)
    elapsed_time = time.time() - start_time

    real_rollout = get_real_rollout(dataset, time_start, time_stop).to(device)
    assert real_rollout.shape == predicted_rollout.shape, "Real and predicted rollout must have the same dimensions"

    return predicted_rollout, real_rollout, elapsed_time