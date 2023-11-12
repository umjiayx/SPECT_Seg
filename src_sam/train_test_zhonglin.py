from utils import *
from transformers import SamProcessor
from torch.utils.data import DataLoader
from transformers import SamModel 
from torch.optim import Adam
import monai
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from prepare_dataset_zhonglin import zhonglindataloader, create_dataset
import transcript
from test_only import test
from torch.optim.lr_scheduler import ReduceLROnPlateau



def train_test(traindataset, testdataset, savedir, bbox_threshold, prompt_mode):
    assert prompt_mode in {'point', 'box', 'point+box'}
    writer = SummaryWriter(savedir)
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    train_dataset = SAMDataset(dataset=traindataset, processor=processor,
                               bbox_threshold=bbox_threshold)
    test_dataset = SAMDataset(dataset=testdataset, processor=processor,
                              bbox_threshold=bbox_threshold)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    model = SamModel.from_pretrained("facebook/sam-vit-base")
    for name, param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)
    optimizer = Adam(model.mask_decoder.parameters(), lr=1e-4, weight_decay=0)
    scheduler = ReduceLROnPlateau(optimizer, 'min')
    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    num_epochs = 50

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Using device: {device}")
    valid_loss_history = []
    for epoch in range(num_epochs):
        train_loss = 0
        niter = 0
        ############## start training ################
        print('############## START TRAINING ###############')
        model.train()
        for batch in tqdm(train_dataloader):
            # forward pass
            outputs = forward_pass(model, batch, prompt_mode, device)

            # compute loss
            predicted_masks = outputs.pred_masks.squeeze(1)
            ground_truth_masks = batch["ground_truth_mask"].float().to(device)
            loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))
            train_loss += loss.item()
            # backward pass (compute gradients of parameters w.r.t. loss)
            optimizer.zero_grad()
            loss.backward()
            # optimize
            optimizer.step()
            niter += 1
        train_loss /= niter
        writer.add_scalar('Loss/train', train_loss, epoch)
        
        ############## start validation ################
        print('############## START VALIDATION ###############')
        # start validation
        model.eval()
        val_loss = 0
        niter = 0
        with torch.no_grad():
            for batch in tqdm(test_dataloader):
                outputs = forward_pass(model, batch, prompt_mode, device)
                # compute loss
                predicted_masks = outputs.pred_masks.squeeze(1)
                ground_truth_masks = batch["ground_truth_mask"].float().to(device)
                loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))
                val_loss += loss.item()
                niter += 1
            val_loss /= niter
            scheduler.step(val_loss)
            valid_loss_history.append(val_loss)
            writer.add_scalar('Loss/val', val_loss, epoch)
        
        print(f'Epoch: {epoch+1:03d}/{num_epochs:03d} ||' \
              f'Training loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}')
        
        #################### save epochs ####################
        if (epoch + 1) % 5 == 0:
            print('Save the current model to checkpoint!')
            save_checkpoint(model.state_dict(), is_best=False,
                            checkpoint_dir=f'{savedir}/checkpoint')
        if epoch == np.argmin(valid_loss_history):
            print('The current model is the best model! Save it!')
            save_checkpoint(model.state_dict(), is_best=True,
                            checkpoint_dir=f'{savedir}/checkpoint')
    writer.close()


if __name__ == "__main__":
    bbox_threshold = 10
    prompt_mode = 'box'
    savedir = '/home/zhonglil/ondemand/data/sys/myjobs/default/SAM_seg/test-sam-zhonglin'+f'-bbox={bbox_threshold}-prompt={prompt_mode}-updata'
    init_env(seed_value=42)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
        print(f"The new directory {savedir} is created!")
    copytree_code(os.getcwd(), savedir)
    transcript.start(f'{savedir}/logfile.log', mode='a')
    # prepare training data
    train_loader = zhonglindataloader(datasetdir='/home/zhonglil/ondemand/data/sys/myjobs/default/SAM_seg/train_data.mat', delete_zeros=True)
    train_loader.convert_data()
    traindataset = create_dataset(images=train_loader.meta['spect'], labels=train_loader.meta['seg'])
    # prepare validation data
    val_loader = zhonglindataloader(datasetdir='/home/zhonglil/ondemand/data/sys/myjobs/default/SAM_seg/val_data.mat', delete_zeros = True)
    val_loader.convert_data()
    valdataset = create_dataset(images=val_loader.meta['spect'], labels=val_loader.meta['seg'])
    train_test(traindataset=traindataset, 
               testdataset=valdataset, 
               savedir=savedir,
               bbox_threshold=bbox_threshold, 
               prompt_mode=prompt_mode)
    transcript.stop()
    
    if not os.path.exists(savedir):
        AssertionError(f"Directory {savedir} NOT found!")
    with torch.no_grad():
        transcript.start(f'{savedir}/logfile.log', mode='a')
        print("################# START TESTING ###################")
        test_loader = zhonglindataloader(datasetdir='/home/zhonglil/ondemand/data/sys/myjobs/default/SAM_seg/test_data.mat', delete_zeros = False)
        test_loader.convert_data()
        testdataset = create_dataset(images=test_loader.meta['spect'], labels=test_loader.meta['seg'])
        test(testdataset=testdataset, 
             savedir=savedir, 
             bbox_threshold = bbox_threshold,
             prompt_mode= prompt_mode)
        transcript.stop()