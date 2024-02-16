
import random, torch

from data_loading import test_data
from model_training_and_inference import grid_scan_loop, inference_resnet_no_tuning
from model_training_and_inference import train_save_best_models, model_inference
from data_visualization import plot_accuracy_plots, uncertainty_hist_and_worst_img

random.seed(42)
torch.manual_seed(42)

if __name__ == "__main__":

    # To execute the grid scan performed in the analysis

    ## grid_scan_loop()

    # To calculate the test accuracy of the ResNet model 
    # with no tuning (91.5% accuracy) 

    ## acc = inference_resnet_no_tuning()
    ## print(acc)

    # To retrain the best model for each category on all 
    # the training dataset  

    ## train_save_best_models()

    # To do inference on some data with best models

    res = model_inference("MP", test_data)
    print(res)

    # visualizations

    plot_accuracy_plots("ResNet")
    uncertainty_hist_and_worst_img()




