import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
from skimage.transform import resize as skires
from utils import parse_args

class Graphs:
    def __init__(self):
        self.config = parse_args()   
    
    def visualize(self, epochs, scores, legends, x_label, y_label, title):
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'black'] 
        for score, legend, color in zip(scores, legends, colors):
            plt.plot(epochs, score, color, label=legend)
        
        plt.legend(loc='upper right')    
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        plt.savefig(f"outputs/{self.config.name}/graphs/graph2.jpeg")
        plt.show()

    def read_data(self, type):
        df = pd.read_csv(f"outputs/{self.config.name}/{type}.csv")
        fields = df.columns.tolist()  
        metrics = []
        for column in df.columns:
            metrics.append(df[column].tolist())
            
        return df, fields, metrics

    def training_plotting(self):
        _, fields, metrics = self.read_data(type='epo_log')
            
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
        
        train_hausdorff = metrics[options['Train hausdorff']]
        train_hausdorff = [x / 100 for x in train_hausdorff]
        
        val_hausdorff = metrics[options['Val hausdorff']]
        val_hausdorff = [x / 100 for x in val_hausdorff]
        
        
        self.visualize(
            iters, 
            [
            metrics[options['Train dice score']], metrics[options['Val dice score']],
            metrics[options['Train iou score']], metrics[options['Val iou score']],
            metrics[options['Train loss']], metrics[options['Val loss']] 
            ],  
            
            [
            fields[options['Train dice score']] + ' (/100)', fields[options['Val dice score']] + ' (/100)', 
            fields[options['Train iou score']], fields[options['Val iou score']],
            fields[options['Train loss']], fields[options['Val loss']] 
            ],  
            
            'Epochs', 'Scores', 'Training results', 
        )
    
    # Only use for testing 
    def boxplot(self):
        df_dice = pd.read_csv(f"outputs/{self.config.name}/infer_dice_class.csv")
        df_iou  = pd.read_csv(f"outputs/{self.config.name}/infer_iou_class.csv")
        
        df_dice['type'] = 'dice'
        df_iou['type'] = 'iou'

        df_combined = pd.concat([df_dice, df_iou])
        df_combined.reset_index(drop=True, inplace=True)
        df_final = pd.melt(df_combined, id_vars=['type'], var_name='class', value_name='score')
        df_final.sort_values(['type', 'class'], inplace=True)
        df_final.reset_index(drop=True, inplace=True)

        fig = px.box(df_final, x="class", y="score", color="type")
        fig.update_traces(quartilemethod="exclusive")
        fig.update_layout(width=700, height=700)
        fig.show()

if __name__ == '__main__':
    graph = Graphs()
    graph.training_plotting()