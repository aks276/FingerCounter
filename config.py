import torch.cuda as cuda

TRAIN_PATH='./train'
TEST_PATH='./test'

BATCH_SIZE=16

EPOCHS = 10
DEVICE = 'cuda' if cuda.is_available() else 'cpu'
LEARNING_RATE = 0.001
MODEL_PATH = './models'