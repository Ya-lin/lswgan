# %%
# import yaml
import time
import utils
import torch
import numpy as np
from box import Box
from tqdm import tqdm
from config import Config
import torch.optim as optim
from models import PointCloudAE
import matplotlib.pyplot as plt
from dataloaders import GetDataLoaders
from pytorch3d.loss import chamfer_distance


# 
def train_epoch():
    epoch_loss = 0
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        
        data = data.to(args.device)
        output = ae(data.permute(0,2,1)) # transpose data for NumberxChannelxSize format
        loss, _ = chamfer_distance(data, output) 
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss/(i+1)

# test process
@torch.no_grad()
def test_batch(data): # test with a batch of inputs

    data = data.to(args.device)
    output = ae(data.permute(0,2,1))
    loss, _ = chamfer_distance(data, output)
    
    return loss.item(), output.cpu()

def test_epoch(): # test with all test set

    epoch_loss = 0
    for i, data in enumerate(test_loader):
        loss, _ = test_batch(data)
        epoch_loss += loss

    return epoch_loss/(i+1)

# if args.save_results:
#     utils.clear_folder(args.ae_output)


# %% train ae and save it
if  __name__ == "__main__":

    # load config and data
    args = Config()
    print(args)

    data_file = args.data_dir/args.dataset
    pc_array = np.load(data_file)
    print(f"data shape: {pc_array.shape}.")

    # Assuming all models have the same size,
    point_size = pc_array.shape[1]
    print(f"number of points in one 3D cloud: {point_size}.")

    # load dataset from numpy array and divide 90%-10% randomly for train and test sets
    train_loader, test_loader = GetDataLoaders(npArray=pc_array, batch_size=args.batch_size)


    # initialize model and set up optimizier
    ae = PointCloudAE(point_size, args.latent_size).to(args.device)
    optimizer = optim.Adam(ae.parameters(), lr=0.0005)

    # train ae model and save it
    history = Box(train=[], test=[])
    for i in tqdm(range(args.ae_epoch)):

        startTime = time.time()
        
        train_loss = train_epoch() #train one epoch, get the average loss
        history.train.append(train_loss)
        
        test_loss = test_epoch() # test with test set
        history.test.append(test_loss)
        
        epoch_time = time.time() - startTime
        
        writeString = f"epoch {i} train loss : {train_loss} test loss : {test_loss} epoch time : {epoch_time}\n"

        # plot train/test loss graph
        plt.plot(history.train, label="Train")
        plt.plot(history.test, label="Test")
        plt.legend()

        if args.save_results: # save all outputs to the save folder

            # write the text output to file
            output_file = args.ae_output / "prints.txt"
            with output_file.open("a") as file:
                file.write(writeString)

            # update the loss graph
            plt.savefig(args.ae_output/"loss.png")
            plt.close()

            # save input/output as image file
            if(i%50==0):
                test_samples = next(iter(test_loader))
                loss , test_output = test_batch(test_samples)
                utils.plotPCbatch(test_samples, test_output, show=False, save=True, 
                                  name = args.ae_output / f"epoch_{i}.png")

        else : # display all outputs
            test_samples = next(iter(test_loader))
            loss , test_output = test_batch(test_samples)
            utils.plotPCbatch(test_samples, test_output)

            print(writeString)

            plt.show()

    print("Save model ae!")
    torch.save(ae.cpu().state_dict(), args.model_dir/"ae.pt")


#%%
