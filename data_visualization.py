
import random, re, csv, os, torch
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from data_loading import muffins_filenames, chihuahuas_filenames
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_loading import final_train_dataloader, final_test_dataloader, muffins_filenames, test_data
from models import ML_1, ML_2, ML_3
from models import CNN_1, CNN_2, CNN_3
from models import ResNet50_transfer, ResNet34_transfer, ResNet18_transfer, ResNet50
from steps import train_step, test_step


def plot_transformed_images(image_paths, n=3, transform=None):
    random_image_paths = random.sample(image_paths, k=n)
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            if transform == None:
                fig, ax = plt.subplots(1, 1)
                ax.imshow(f) 
                ax.set_title(f"Original \nSize: {f.size}")
                ax.axis("off")
            else:
                fig, ax = plt.subplots(1, 2)
                ax[0].imshow(f) 
                ax[0].set_title(f"Original \nSize: {f.size}")
                ax[0].axis("off")
                transformed_image = transform(f).permute(1, 2, 0) 
                ax[1].imshow(transformed_image) 
                ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
                ax[1].axis("off")
    plt.show()


def plot_sample_images(nl, transform = None):
    plot_transformed_images(muffins_filenames, n=nl, transform=transform)
    plot_transformed_images(chihuahuas_filenames, n=nl, transform=transform)


def calculate_average_metrics_across_folds(series_folds_loss, series_folds_acc, max_epochs):
    loss = [0.0] * max_epochs
    accuracy = [0.0] * max_epochs
    metrics = (loss, accuracy)
    for j, ms in enumerate((series_folds_loss, series_folds_acc)):
        for s in ms:
            d = max_epochs - len(s)
            s_inputed = [*s, *([s[-1]] * d)]
            for i, o in enumerate(s_inputed):
                metrics[j][i] += o
    for j in range(len(metrics)):
        for i in range(max_epochs):
            metrics[j][i] /= len(series_folds_loss)
    return metrics

def plot_accuracy_plots(highlighted = "ML"):

    df = retrieve_results()

    sns.set_theme()
    t = range(0, 51)

    non_hightleted = [m for m in ("ML", "CNN", "ResNet") if m != highlighted]

    fnns = ["ML_1", "ML_2", "ML_3"]
    cnns = ["CNN_3", "CNN_2"]
    resnets = ["ResNet50_transfer", "ResNet34_transfer", "ResNet18_transfer"]
    models = fnns if highlighted in fnns[1] else (cnns if highlighted in cnns[1] else resnets)

    cmap = matplotlib.cm.get_cmap('inferno')
    colors = [cmap(0.0), cmap(0.35), cmap(0.7)]

    cur_max = 0
    idx_max, params_max = None, None

    for r in df.iterrows():
        if highlighted in r[1]["model_params_choices"]:
            s_max = max(r[1]["list_validation_accs"])
            if s_max > cur_max:
                cur_max = s_max
                idx_max = r[1]["list_validation_accs"].index(s_max)
                params_max = r[1]["model_params_choices"]
            idx = [i for i, j in enumerate(models) if j in r[1]["model_params_choices"]][0]
            plt.plot(t, r[1]["list_validation_accs"], linewidth=0.6, color=colors[idx], zorder=1)
        elif non_hightleted[0] in r[1]["model_params_choices"]:
            plt.plot(t, r[1]["list_validation_accs"], ':', color="lightgray", zorder=0)
        else:
            plt.plot(t, r[1]["list_validation_accs"], '--', color="lightgray", zorder=0)

    print(f"\nBest model for {highlighted}")
    print(f"max cross-validated accuracy {cur_max} at epoch {idx_max}")
    print(f"params: {params_max}")

    save_best_model_results(highlighted, params_max, cur_max, idx_max)

    plt.margins(x=0.01, y=0.02)

    if highlighted == "ML":
        m = "5-fold Cross-Validated Accuracy of Multilayered Perceptrons models"
    if highlighted == "CNN":
        m = "5-fold Cross-Validated Accuracy of Convolutional Neural Networks models"
    if highlighted == "ResNet":
        m = "5-fold Cross-Validated Accuracy of Residual Neural Networks models"
    plt.title(m, fontsize=22)
    plt.xlabel("epoch", fontsize=19)
    plt.ylabel("accuracy", fontsize=19)
    plt.xticks(fontsize=16, rotation=0)
    plt.yticks(fontsize=16, rotation=0)
    plt.gca().yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=100.0))
    plt.show()

def retrieve_results():
    df = pd.read_csv("./results/results.csv")
    df = df[["list_validation_accs", "list_validation_losses", "model_params_choices"]]
    df["list_validation_accs"] = df["list_validation_accs"].apply(eval)
    df["list_validation_losses"] = df["list_validation_losses"].apply(eval)

    for i in range(len(df)):
        metrics = calculate_average_metrics_across_folds(df["list_validation_losses"][i], 
                                                         df["list_validation_accs"][i], 51)
        df.at[i, "list_validation_losses"] = metrics[0]
        df.at[i, "list_validation_accs"] = metrics[1]
    return df

def save_best_model_results(category, params, max_acc, epochs):
    mode = "a" if os.path.isfile("./results/best_models.csv") else "w"
    with open(f'./results/best_models.csv', mode) as f: 
        w = csv.writer(f, ["Category", "Parameters", "Max_Accuracy", "Optimal_Epochs"])
        if mode == "w": w.writerow(("Category", "Parameters", "Max_Accuracy", "Optimal_Epochs")) 
        w.writerow((category, params, max_acc, epochs))
    
def uncertainty_hist_and_worst_img():
    df = pd.read_csv("./results/best_models_p.csv")
    df["Parameters"] = df["Parameters"].apply(lambda r: eval(r, globals()))
    p = tuple(df[df["Category"] == "ResNet"].iloc[0])[1]
    model_cl = p["model"]
    if "hidden_units" in p:
        model = model_cl(p["input_shape"], p["hidden_units"], p["output_shape"])
    else:
        model = model_cl()
    model.load_state_dict(torch.load(f"./best_models/best_model_ResNet.pt"))
    model.eval()
    all_probs = []
    i = 0
    k = [1,3,2,0]
    f, axarr = plt.subplots(2,4) 
    for _, (X, y) in enumerate(final_test_dataloader):
        if hasattr(model, 'weights') and model.weights != None: 
            X2 = model.weights.transforms(antialias=True)(X)
        y_pred = model(X2)
        y_prob = torch.softmax(y_pred, dim=1)
        y_prob = float(y_prob[0][y])

        all_probs.append(y_prob)
        if y_prob < 0.9:
            X_m = X.reshape(X.shape[1:])
            axarr[0][k[i]].imshow(X_m.permute(1, 2, 0))
            X_m = X2.reshape(X2.shape[1:])
            axarr[1][k[i]].imshow(X_m.permute(1, 2, 0))
            i += 1

    for ax in axarr.flat:
        ax.grid(False)
        plt.setp(ax.spines.values(), alpha = 0)
        ax.tick_params(which = 'both', size = 0, labelsize = 0)
    f.set_figheight(15)
    f.set_figwidth(15)
    f.suptitle("Images classified with more uncertainty by the model", fontsize=22)
    plt.show()

    sns.set_theme()
    plt.hist(all_probs, bins=100, log=True, edgecolor='black', linewidth=1.2)
    plt.title("Histogram for the (un)certainty of the best model in its predictions", fontsize=22)
    plt.xlabel("probability", fontsize=19)
    plt.ylabel("frequency", fontsize=19)
    plt.xticks(fontsize=16, rotation=0)
    plt.yticks(fontsize=16, rotation=0)
    plt.show()


