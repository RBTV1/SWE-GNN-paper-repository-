# Libraries
import torch

def get_mean_error(diff_rollout, type_loss, nodes_dim=0):
    if type_loss == 'RMSE':
        average_diff_t = torch.sqrt((diff_rollout**2).mean(nodes_dim))
    elif type_loss == 'MAE':
        average_diff_t = diff_rollout.abs().mean(nodes_dim)
    return average_diff_t

def mask_on_water(preds, real, water_axis=1):
    where_water = ((preds != 0) + (real != 0)).any(water_axis)
    return where_water

def get_loss_variable_scaler(water_variables, velocity_scaler=1):
    '''Scales loss in velocity terms by a factor velocity_scaler'''
    loss_scaler = torch.ones(water_variables)
    if water_variables>1:
        loss_scaler[1:] = velocity_scaler
        
    return loss_scaler

def loss_WD(preds, real, type_loss='RMSE', only_where_water=False, velocity_scaler=1):
    '''
    Calculates RMSE or MAE loss with a smoothness term, as defined in smoothness
    give more weight to places where there is actually a difference between time steps
    ------
    type_loss: str
        options: 'RMSE', 'MAE'
    loss_change_weight: float (default = 1)
        coefficient to weight more only areas where there is flood variation
    weight_time: float (default = 0)
        coefficient to weight more initial time steps
    '''
    diff = preds - real

    if only_where_water:
        where_water = mask_on_water(preds, real)
        diff = diff[where_water]

    loss = get_mean_error(diff, type_loss, nodes_dim=0)

    loss_scaler = get_loss_variable_scaler(diff.shape[-1], velocity_scaler=velocity_scaler).to(diff.device)
    loss = torch.dot(loss, loss_scaler)

    return loss