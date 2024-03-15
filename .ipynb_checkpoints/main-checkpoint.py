# Libraries
import torch
import wandb
from wandb import Config
import PIL, cv2
import torch.optim as optim
from torch_geometric.loader import DataLoader

from utils.dataset import create_model_dataset, to_temporal_dataset
from utils.dataset import get_temporal_test_dataset_parameters
from utils.load import read_config
from utils.visualization import PlotRollout
from utils.miscellaneous import get_numerical_times, calculate_speed_ups, get_model, SpatialAnalysis
from training.train import Trainer

def fix_dict_in_config(wandb):
    config = dict(wandb.config)
    for k, v in config.copy().items():
        if '.' in k:
            new_key = k.split('.')[0]
            inner_key = k.split('.')[1]
            if new_key not in config.keys():
                config[new_key] = {}
            config[new_key].update({inner_key: v})
            del config[k]
    
    wandb.config = Config()
    for k, v in config.items():
        wandb.config[k] = v

def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_parameters = config.dataset_parameters
    scalers = config.scalers
    selected_node_features = config.selected_node_features
    selected_edge_features = config.selected_edge_features

    train_dataset, val_dataset, test_dataset, scalers = create_model_dataset(
        'grid', scalers=scalers, device=device, **dataset_parameters,
        **selected_node_features, **selected_edge_features
    )

    temporal_dataset_parameters = config.temporal_dataset_parameters

    temporal_train_dataset = to_temporal_dataset(
    train_dataset, **temporal_dataset_parameters)

    print('Number of training simulations:\t', len(train_dataset))
    print('Number of training samples:\t', len(temporal_train_dataset))
    print('Number of node features:\t', temporal_train_dataset[0].x.shape[-1])
    print('Number of rollout steps:\t', temporal_train_dataset[0].y.shape[-1])

    node_features, edge_features = temporal_train_dataset[0].x.size(-1), temporal_train_dataset[0].edge_attr.size(-1)
    num_nodes, num_edges = temporal_train_dataset[0].x.size(0), temporal_train_dataset[0].edge_attr.size(0)

    previous_t = temporal_dataset_parameters['previous_t']
    test_size = dataset_parameters['test_size']
    temporal_res = dataset_parameters['temporal_res']
    
    print('Temporal resolution:\t', temporal_res, 'min')

    model_parameters = config.models
    model_type = model_parameters.pop('model_type')

    if model_type == 'GNN':
        model_parameters['edge_features'] = edge_features
    elif model_type == 'MLP':
        model_parameters.num_nodes = num_nodes

    model = get_model(model_type)(
        node_features=node_features,
        previous_t=previous_t,
        device=device,
        **model_parameters).to(device)

    trainer_options = config.trainer_options
    batch_size = trainer_options.pop('batch_size')

    lr_info = config['lr_info']

    # Model optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr_info['learning_rate'], weight_decay=lr_info['weight_decay'])
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=lr_info['step_size'], gamma=lr_info['gamma'])

    # Batch creation
    train_loader = DataLoader(temporal_train_dataset, batch_size=batch_size, shuffle=True)

    # track gradients
    wandb.watch(model, log="all", log_freq=10)

    # info for testing dataset
    temporal_test_dataset_parameters = get_temporal_test_dataset_parameters(
        config, temporal_dataset_parameters)

    total_parameteres = sum(p.numel() for p in model.parameters())

    wandb.log({"total parameters": total_parameteres})

    # Training
    trainer = Trainer(optimizer, lr_scheduler, **trainer_options)
    trainer.fit(model, train_loader, val_dataset, **temporal_test_dataset_parameters)
    trainer._save_model(model, model_name=f'{wandb.run.id}.h5')

    # Numerical simulation times
    maximum_time = test_dataset[0].WD.shape[1]
    numerical_times = get_numerical_times(test_size, temporal_res, maximum_time, 
                    **temporal_test_dataset_parameters,
                    overview_file='database/raw_datasets/overview.csv')

    # Rollout error and time
    spatial_analyser = SpatialAnalysis(model, test_dataset, **temporal_test_dataset_parameters)
    rollout_loss = spatial_analyser._get_rollout_loss(type_loss=trainer.type_loss)
    model_times = spatial_analyser.prediction_times
                                        
    print('test roll loss WD:',rollout_loss.mean(0)[0].item())
    print('test roll loss V:',rollout_loss.mean(0)[1:].mean().item())

    # Speed up
    avg_speedup, std_speedup = calculate_speed_ups(numerical_times, model_times)

    wandb.log({"speed-up": avg_speedup,
               "test roll loss WD":rollout_loss.mean(0)[0].item(),
               "test roll loss V":rollout_loss.mean(0)[1:].mean().item()})

    spatial_analyser = SpatialAnalysis(model, test_dataset, **temporal_test_dataset_parameters)

    fig, _ = spatial_analyser.plot_CSI_rollouts(water_thresholds=[0.05, 0.3])
    fig.savefig("results/temp_CSI.png")
    img = cv2.imread("results/temp_CSI.png")
    image = wandb.Image(PIL.Image.fromarray(img), caption="CSI scores")
    wandb.log({"CSI scores": image})

    best_id = rollout_loss.mean(1).argmin().item()
    worst_id = rollout_loss.mean(1).argmax().item()
    
    for id_dataset, name in zip([best_id, worst_id],['best', 'worst']):
        rollout_plotter = PlotRollout(model, test_dataset[id_dataset], scalers=scalers, 
            type_loss=trainer.type_loss, **temporal_test_dataset_parameters)
        fig = rollout_plotter.explore_rollout()
        fig.savefig("results/temp_summary.png")
        img = cv2.imread("results/temp_summary.png")
        image = wandb.Image(PIL.Image.fromarray(img), caption=f"id: {id_dataset}")

        wandb.log({f"{name} summary image": image})
    
    print('Training and testing finished!')

if __name__ == '__main__':
    # Read configuration file with parameters
    cfg = read_config('config.yaml')

    wandb.init(
        config=cfg,
    )

    fix_dict_in_config(wandb)

    config = wandb.config

    main(config)