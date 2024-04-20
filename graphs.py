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
    plt.savefig(f"outputs/{config['name']}/graphs/graph3.jpeg")
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default=None, help='model name')
    return parser.parse_args()

def read_data(type):
    # open file
    args = parse_args()
    with open(f'outputs/{args.name}/config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # load data
    df = pd.read_csv(f"outputs/{config['name']}/{type}.csv")
    fields = df.columns.tolist()  
    metrics = []
    for column in df.columns:
        metrics.append(df[column].tolist())
        
    return df, fields, metrics, config

def plotting():
    _, fields, metrics, config = read_data(type='epo_log')
        
    # mapping
    options = {
        'epoch': 0,
        'lr': 1,
        
        'Train loss': 2,
        'Train ce loss': 3,
        'Train dice score': 4,
        'Train dice loss': 5,
        'Train iou score': 6,
        'Train iou loss': 7,
        'Train hausdorff': 8,
        
        'Val loss': 9,
        'Val ce loss': 10,
        'Val dice score': 11,
        'Val dice loss': 12,
        'Val iou score': 13,
        'Val iou loss': 14,
        'Val hausdorff': 15,
    }
        
    iters = [i for i in range(1, (len(metrics[0])) + 1)]
    visualize(
        iters, 
        [metrics[options['Train ce loss']], 
         metrics[options['Val ce loss']]  ],  
        
        [fields[options['Train ce loss']], 
         fields[options['Val ce loss']]  ],
        
        'Epochs', 'Scores', 'Training results', 
        config
    )
    
# Only use for testing 
def boxplot():
    df, _, _, config = read_data(type='ds_class_iter_test')

    # Plotting
    plt.figure(figsize=(10, 6))
    df.boxplot(grid=False) 
    plt.title(f'Boxplot of Dice Score Coefficient per class')
    plt.xlabel('Classes')
    plt.ylabel('Metric Value')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()    

boxplot()