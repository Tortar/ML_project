
import gc, torch, random, platform
from torch import nn
from time import time
from tqdm.auto import tqdm
from sklearn.model_selection import ParameterGrid
from data_loading import final_train_dataloader, final_test_dataloader, muffins_filenames, KFoldDataLoader, test_data
from models import ML_1, ML_2, ML_3
from models import CNN_1, CNN_2, CNN_3
from models import ResNet50_transfer, ResNet34_transfer, ResNet18_transfer, ResNet50
from steps import train_step, test_step, EarlyStopper
from model_saving import save_run_data
import pandas as pd
import numpy as np

random.seed(42)
torch.manual_seed(42)

def grid_scan_loop():

	models_ML = [ML_1, ML_2, ML_3]
	models_CNN = [CNN_2, CNN_3]
	models_transfer = [ResNet18_transfer, ResNet34_transfer, ResNet50_transfer]

	max_epochs = 50

	parameters_ML = {
		"model": models_ML,
		"loss": [nn.CrossEntropyLoss()],
		"optimizer": [torch.optim.SGD, torch.optim.Adam],
		"learning_rate": [0.1, 0.01, 0.001, 0.0001],
		"input_shape": [49152],
		"hidden_units": [128, 256, 512],
		"output_shape": [2],
		"max_epochs": [max_epochs]
	}

	parameters_CNN = {
		"model": models_CNN,
		"loss": [nn.CrossEntropyLoss()],
		"optimizer": [torch.optim.SGD, torch.optim.Adam],
		"learning_rate": [0.1, 0.01, 0.001, 0.0001],
		"input_shape": [3],
		"hidden_units": [128, 256],
		"output_shape": [2],
		"max_epochs": [max_epochs]
	}

	parameters_trasfer = {
		"model": models_transfer,
		"loss": [nn.CrossEntropyLoss()],
		"optimizer": [torch.optim.SGD, torch.optim.Adam],
		"learning_rate": [0.0001, 0.00001, 0.000001],
		"max_epochs": [max_epochs]
	}

	scan_params = []
	for v in (parameters_ML, parameters_CNN, parameters_trasfer):
		scan_params += list(ParameterGrid(v))

	for j, p in enumerate(scan_params[119:]):
		print(f"Starting experiment {j+1}")
		print(f"Configuration\n{p}")
		device = 'cuda' if torch.cuda.is_available() else 'cpu'

		t0 = time()

		k = 5

		list_train_losses = [[] for _ in range(k)]
		list_validation_losses = [[] for _ in range(k)]
		list_validation_accs = [[] for _ in range(k)]
		kdataloaders = KFoldDataLoader(k)

		for i, dataloader in enumerate(kdataloaders):
			gc.collect()
			print(f"Fold {i+1}")
			model_cl = p["model"]
			if "hidden_units" in p:
				model = model_cl(p["input_shape"], p["hidden_units"], p["output_shape"])
			else:
				model = model_cl()
			model.to(device)
			optimizer = p["optimizer"](params = model.parameters(), lr=p["learning_rate"])
			loss = p["loss"]
			epochs = p["max_epochs"]
			stopper = EarlyStopper()

			clipping_value = 1
			torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value)

			train_dataloader, validation_dataloader = dataloader
			last_validation_loss, last_validation_acc = test_step(model, validation_dataloader, loss, device)
			list_validation_losses[i].append(last_validation_loss.item())
			list_validation_accs[i].append(last_validation_acc)
			for epoch in tqdm(range(epochs)):
				last_train_loss = train_step(model, train_dataloader, loss, optimizer, device)
				last_validation_loss, last_validation_acc = test_step(model, validation_dataloader, loss, device)
				list_train_losses[i].append(last_train_loss.item())
				list_validation_losses[i].append(last_validation_loss.item())
				list_validation_accs[i].append(last_validation_acc)
				print(f"Epoch: {epoch}\n-------")
				print(f"\nTrain loss: {last_train_loss:.5f} | Validation loss: {last_validation_loss:.5f}, Validation acc: {last_validation_acc:.2f}%\n")
				last_epoch = epoch + 1
				if stopper.early_stop(last_validation_loss.item()): 
					print("Early stopped")
					break

		t1 = time()

		print(f"\nTotal training time on {device}: {t1 - t0}")

		training_time = t1 - t0
		device_name = platform.processor() if device == "cpu" else torch.cuda.get_device_name("cuda")

		components = {}
		components["model_params_choices"] = p
		components["device_name"] = device_name
		components["training_time"] = training_time
		components["list_train_losses"] = list_train_losses
		components["list_validation_losses"] = list_validation_losses
		components["list_validation_accs"] = list_validation_accs
		components["min_train_losses"] = [min(l) for l in list_train_losses]
		components["min_validation_losses"] = [min(l) for l in list_validation_losses]
		components["max_validation_accs"] = [max(l) for l in list_validation_accs]

		save_run_data(components)


def inference_resnet_no_tuning(model = ResNet50()):
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	model = model.to(device)
	model.eval()
	n, n_correct = 0, 0
	pred_categories = []
	with torch.inference_mode():
		for X, y in final_test_dataloader:
			X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
			if hasattr(model, 'weights') and model.weights != None: 
				X = model.weights.transforms(antialias=True)(X)
			test_pred = model(X)
			# dog species are from 151 to 268
			pred = int(not 0 <= int(np.argmax(test_pred.to("cpu"))) <= 397)
			if pred == int(y):
				n_correct += 1	
			n += 1

	return n_correct/n

def train_save_best_models():
	df = pd.read_csv("./results/best_models_p.csv")
	df["Parameters"] = df["Parameters"].apply(lambda r: eval(r, globals()))
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	for r in df.itertuples():
		gc.collect()
		p = r[2]
		model_cl = p["model"]
		if "hidden_units" in p:
			model = model_cl(p["input_shape"], p["hidden_units"], p["output_shape"])
		else:
			model = model_cl()
		model.to(device)
		loss, optimizer = p["loss"], p["optimizer"](params = model.parameters(), lr=p["learning_rate"])
		for epoch in tqdm(range(r[4])):
			last_train_loss = train_step(model, final_train_dataloader, loss, optimizer, device)
			last_test_loss, last_test_acc = test_step(model, final_test_dataloader, loss, device)
			print(f"Epoch: {epoch}\n-------")
			print(f"\nTrain loss: {last_train_loss:.5f} | Test loss: {last_test_loss:.5f}, Test acc: {last_test_acc:.2f}%\n")
		torch.save(model.state_dict(), f"./best_models/best_model_{r[1]}.pt")


def model_inference(category, data):
    df = pd.read_csv("./results/best_models_p.csv")
    df["Parameters"] = df["Parameters"].apply(lambda r: eval(r, globals()))
    p = tuple(df[df["Category"] == category].iloc[0])[1]
    model_cl = p["model"]
    if "hidden_units" in p:
        model = model_cl(p["input_shape"], p["hidden_units"], p["output_shape"])
    else:
        model = model_cl()
    model.load_state_dict(torch.load(f"./best_models/best_model_{category}.pt"))
    model.eval()

    predictions = []

    for _, (X, y) in enumerate(final_test_dataloader):
        if hasattr(model, 'weights') and model.weights != None: 
            X = model.weights.transforms(antialias=True)(X)
        y_pred = model(X)
        y_pred = y_pred.argmax(dim=1)
        predictions.append((y_pred, bool(y_pred == y)))

    return [("Cihuahua", x[1]) if x[0] == 0 else ("Muffin", x[1]) for x in predictions]

        