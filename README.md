# Project on Neural Networks

This repository contains the code and the [report](https://github.com/Tortar/ML_project/blob/main/ML_report.pdf) 
for the project on Neural Networks of the Machine Learning and Statistical Learning course of the Msc in Data Science 
for Economics.

To reproduce the findings of the analysis described in the report, the `main.py` file 
can be easily edited and called from the command line. Apart from the setup which is
always the same, it contains the following code:

```python
if __name__ == "__main__":

    # To execute the grid scan performed in the analysis

    grid_scan_loop()

    # To calculate the test accuracy of the ResNet model 
    # with no tuning (91.5%) 

    acc = inference_resnet_no_tuning()
    print(acc)

    # To retrain the best model for each category on the 
    # all training dataset  

    train_save_best_models()

    # To do inference on some data with best models

    res = model_inference("CNN", test_data)
    print(res)

    # visualizations of the 

    plot_accuracy_plots("ResNet")
    uncertainty_hist_and_worst_img()
```

Some parts are commented in the actual file because they require a lot of
time to execute, in particular the `grid_scan_loop()` function requires days to finish
(but if restarted it will resume from the point it stopped).

The best models for each tested architecture are saved inside the [`best_models`](https://github.com/Tortar/ML_project/tree/main/best_models) folder.
