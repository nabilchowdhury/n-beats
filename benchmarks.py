import argparse

from dataloaders import *
from nbeats import NBEATSLosses


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='N-Beats Benchmarks: Calculates metrics on datasets using naive prediction (extending last observed value)')
    parser.add_argument('--dataset-name', '-ds', help='Name of Dataset eg. M3Dataset', type=str, required=True)
    args = parser.parse_args()

    DATASET_NAME = args.dataset_name
    DATASET = globals()[DATASET_NAME]

    with torch.no_grad():
        for criterion in [NBEATSLosses.SMAPE, NBEATSLosses.MAPE]:
            losses, weights = [], []
            for horizon in DATASET.HORIZONS:
                FORECAST_LEN = DATASET.HORIZONS[horizon]
                test_set = DATASET(rf'data/{DATASET_NAME}/npy/{horizon}.npy', FORECAST_LEN, 1, MDataset.TEST)
                test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=1024)
                losses_horizon, weights_horizon = [], []
                w = 0
                
                for x, y in test_loader:
                    y_naive = x[:, -1].view(-1, 1).repeat(1, FORECAST_LEN) 
                    losses_horizon.append(criterion(y, y_naive).item())
                    weights_horizon.append(y.shape[0])
                    w += y.shape[0]
                losses.append(np.average(losses_horizon, weights=weights_horizon))
                weights.append(w)

            overall_loss = np.average(losses, weights=weights)
            
            # Output to console
            all_horizons = list(DATASET.HORIZONS)

            print(criterion.__name__)
            print("\n".join(f"{all_horizons[i]}: {losses[i]}" for i in range(len(DATASET.HORIZONS))))
            print(f"Overall: {overall_loss}")
            print()