from torchvision import transforms
import torch

DATA_DIRECTORY = "data\\"
MODEL_DIRECTORY = "model\\"
MODEL_FILENAME = 'handwriting_cnn_model.ckpt'
MINST_MEAN, MINST_STANDARD_DIV = 0.1307, 0.3081
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((MINST_MEAN,), (MINST_STANDARD_DIV,))])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")