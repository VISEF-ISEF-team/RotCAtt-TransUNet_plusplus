import argparse
import yaml
import matplotlib.pyplot as plt
import pandas as pd
from skimage.transform import resize as skires

def visualize(epochs, scores, legends, x_label, y_label, title, config):
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan'] 
    for score, legend, color in zip(scores, legends, colors):
        plt.plot(epochs, score, color, label=legend)
        
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(f"outputs/{config['name']}/graph1.jpeg")
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default=None, help='model name')
    return parser.parse_args()

def plotting():
    args = parse_args()
    with open(f'outputs/{args.name}/config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    df = pd.read_csv(f"outputs/{config['name']}/epo_log.csv")
    fields = df.columns.tolist()  
    metrics = []
    for column in df.columns:
        metrics.append(df[column].tolist())
        
    iters = [i for i in range(1, (len(metrics[0])) + 1)]
    visualize(iters, [metrics[4], metrics[5], metrics[6], metrics[7]],  [fields[4], fields[5], fields[6], fields[7]],
            'Epochs', 'Scores', 'Training results', config)