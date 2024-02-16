
import os
import datetime
import csv
import re

try: os.mkdir("./model")
except FileExistsError: pass

try: os.mkdir("./results")
except FileExistsError: pass

def print_last_run():
	res = [f for f in os.listdir("./results") if f.endswith(".csv")]
	print(max([int(re.search(r'\d+', x).group()) for x in res]))

def save_run_data(components):
	mode = "a" if os.path.isfile("./results/results.csv") else "w"
	with open(f'./results/results.csv', mode) as f:  
		w = csv.DictWriter(f, sorted(components.keys()))
		if mode == "w": w.writeheader()        
		w.writerow(components)

def save_model(model, optimizer, loss, epoch):
	timestamp = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
	filepath = f"./model/{model_name}_full_{timestamp}.pth"
	state = {'epoch': epoch, 'model_state': model.state_dict(),
     		 'optimizer_state': optimizer.state_dict(), 'loss': loss}
	torch.save(obj=state, f=filepath)
			