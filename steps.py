
import torch 
from evaluation_metrics import accuracy_fn
import torchvision.transforms as T
from PIL import Image

def train_step(model, dataloader, loss_fn, optimizer, device):
	model.train()

	train_loss = 0

	for batch, (X, y) in enumerate(dataloader):
		X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
		if hasattr(model, 'weights') and model.weights != None: 
			X = model.weights.transforms(antialias=True)(X)
		# 1. Forward pass
		y_pred = model(X)
		# 2. Calculate loss (per batch)
		loss = loss_fn(y_pred, y)
		# 3. Add up the loss per epoch 
		train_loss += loss
		# 4. Optimizer zero grad
		optimizer.zero_grad()
		# 5. Loss backward
		loss.backward()
		# 6. Optimizer step
		optimizer.step()

		del loss
		if batch % 400 == 0:
		    print(f"Looked at {batch * len(X)}/{len(dataloader.dataset)} samples")

	train_loss /= len(dataloader)

	return train_loss
   
def test_step(model, dataloader, loss_fn, device):

	test_loss, test_acc = 0, 0 
	model.eval()
	with torch.inference_mode():
		i = 0
		for X, y in dataloader:
			X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
			if hasattr(model, 'weights') and model.weights != None: 
				X = model.weights.transforms(antialias=True)(X)
			# 1. Forward pass
			test_pred = model(X)
			# 2. Calculate loss (accumatively)
			test_loss += loss_fn(test_pred, y)
			# 3. Calculate accuracy (preds need to be same as y_true)
			acc = accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))
			test_acc += acc
        
		test_loss /= len(dataloader)

		test_acc /= len(dataloader)

	return test_loss, test_acc

class EarlyStopper:
	def __init__(self, n=10):
		self.n = n
		self.last_losses = []
		self.min_validation_loss = float('inf')

	def early_stop(self, validation_loss):
		self.last_losses.append(validation_loss)
		if len(self.last_losses) < self.n: 
			return False
		else:
			if min(self.last_losses) <= self.min_validation_loss:
				self.min_validation_loss = min(self.last_losses)
				self.last_losses.pop(0)
				return False
			else:
				return True
