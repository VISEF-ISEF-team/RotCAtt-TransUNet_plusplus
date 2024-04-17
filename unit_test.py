import torch
from metrics import WeightedHausdorffDistance

height = 256
width = 256
p = -9
return_2_terms = False
whd_calculator = WeightedHausdorffDistance(height=height,
                                           width=width,
                                           p=p,
                                           return_2_terms=return_2_terms)

prob_map = torch.rand(2, height, width) 
gt = [torch.tensor([[100, 150], 
                    [200, 100]]),           
      torch.tensor([[50, 100]])]             
orig_sizes = torch.tensor([[512, 512], [640, 480]])

whd_result = whd_calculator(prob_map, gt, orig_sizes)
print("Weighted Hausdorff Distance:", whd_result.item())