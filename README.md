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

    # functions producing the graphs of the report 

    plot_accuracy_plots("ResNet")
    uncertainty_hist_and_worst_img()
```

Some parts are commented in the actual file because they require a lot of
time to execute, in particular the `grid_scan_loop()` function requires 
days to finish (but if restarted it will resume from the point it stopped).

The grid scan loop and the training could give sligthly different results 
because of the non-deterministic nature of the GPU training, the seeds of the
random generator for Python and Pytorch have been fixed to reduce that, 
but no further attempt to reduce the non-determinism have been made during 
these phases because even if `torch.use_deterministic_algorithms(True)` was used, 
it would't have actually guaranteed determinism on different hardwares and softwares,
and at the same time it would have made the training slower. In any case, the best models 
for each tested architecture are already saved inside the [`best_models`](https://github.com/Tortar/ML_project/tree/main/best_models) 
folder, and the results of the grid scan loop in the [`results`](https://github.com/Tortar/ML_project/tree/main/results)
folder.
