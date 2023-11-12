from utils import *
from transformers import SamProcessor
from torch.utils.data import DataLoader
from transformers import SamModel 
from tqdm import tqdm
from prepare_dataset_liver_phantom import liverphantomdataloader
from prepare_dataset import create_dataset
from julia import Julia
jl = Julia(compiled_modules=False)
from julia.AVSfldIO import fld_write
import transcript

    
def test(testdataset, ckptdir, savedir, bbox_threshold, prompt_mode):
    assert prompt_mode in {'point', 'box', 'point+box'}
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    test_dataset = SAMDataset(dataset=testdataset, processor=processor,
                              bbox_threshold=bbox_threshold)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    model = SamModel.from_pretrained("facebook/sam-vit-base")
    checkpoint_path = os.path.join(ckptdir, 'checkpoint', 'best_checkpoint.pytorch')
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    niter = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    all_seg = []
    all_true = []
    for batch in tqdm(test_dataloader):
        outputs = forward_pass(model, batch, prompt_mode, device)
        # compute loss
        predicted_masks = torch.sigmoid(outputs.pred_masks.squeeze(1))
        # convert soft mask to hard mask
        predicted_masks = predicted_masks.cpu()
        predicted_masks = (predicted_masks > 0.5).to(torch.uint8)
        ground_truth_masks = batch["ground_truth_mask"].float().to(device)
        all_seg.append(predicted_masks)
        all_true.append(ground_truth_masks)
        niter += 1
    
    all_seg_stack = torch.stack(all_seg).squeeze()
    all_seg_stack = all_seg_stack.cpu().numpy()
    all_true_stack = torch.stack(all_true).squeeze()
    all_true_stack = all_true_stack.cpu().numpy()
    
    fld_write(os.path.join(savedir, 'outputseg.fld'), all_seg_stack)
    fld_write(os.path.join(savedir, 'trueseg.fld'), all_true_stack)
    
    
if __name__ == "__main__":
    ckptdir = '../result/test-sam-105-bbox=1-prompt=point'
    savedir = '../result/liver-phantom-point/'
    if not os.path.exists(savedir):
        os.makedirs(savedir)
        print(f"The new directory {savedir} is created!")
    init_env(seed_value=42)
    if not os.path.exists(ckptdir):
        AssertionError(f"Directory {ckptdir} NOT found!")
    with torch.no_grad():
        transcript.start(f'{savedir}/logfile.log', mode='a')
        print("################# START TESTING ###################")
        test_loader = liverphantomdataloader(datasetdir='../y90-data-wden/liver-phantom/')
        testdataset = create_dataset(images=test_loader.meta['spect'], labels=test_loader.meta['seg'])
        test(testdataset=testdataset, ckptdir=ckptdir,
             savedir=savedir, bbox_threshold=1,
             prompt_mode='point')
        transcript.stop()