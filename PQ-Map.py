import clip
import torch
import cv2
import numpy as np
from PIL import Image
from  matplotlib import pyplot as plt
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC
from segment_anything import sam_model_registry, SamPredictor
import argparse

device='cuda'
model, preprocessor = clip.load("CS-ViT-B/16", device=device, jit=False)
model.eval()

def load_embed(state_dict):
    ctx=state_dict['params']['prompt_learner.ctx']
    token_prefix=state_dict['params']['prompt_learner.token_prefix']
    token_suffix=state_dict['params']['prompt_learner.token_suffix']
    pos=torch.cat((state_dict['params']['prompt_learner.token_prefix'], state_dict['params']['prompt_learner.ctx'], state_dict['params']['prompt_learner.token_suffix']), 1).to(device)
    name_len=3
    half_n_ctx=8
    prefix_i = token_prefix[:, :, :]
    class_i = token_suffix[:, :name_len, :]
    suffix_i = token_suffix[:, name_len:, :]
    ctx_i_half1 = ctx[:, :half_n_ctx, :]
    ctx_i_half2 = ctx[:, half_n_ctx:, :]
    text_embed = torch.cat(
        [
            prefix_i,     # (1, 1, dim)
            ctx_i_half1,  # (1, n_ctx//2, dim)
            class_i,      # (1, name_len, dim)
            ctx_i_half2,  # (1, n_ctx//2, dim)
            suffix_i,     # (1, *, dim)
        ],
        dim=1,
    ).to(device).type(model.dtype)
    return text_embed

def encode_text(text_embed, pos):
    x = text_embed

    #x = x + model.positional_embedding.type(model.dtype)
    x = x + pos
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = model.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = model.ln_final(x).type(model.dtype)

    # x.shape = [batch_size, n_ctx, transformer.width]
    # take features from the eot embedding (eot_token is the highest number in each sequence)
    x = x[torch.arange(x.shape[0]), torch.tensor([20, 20])] @ model.text_projection

    return x

def single_quality_map(image,text_embed,img_size):
    with torch.no_grad():
    # Extract image features
        image_features = model.encode_image(image)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)

        # Prompt ensemble for text features with normalization
        text_features = encode_text(text_embed, pos_embed)

        # Similarity map from image tokens with min-max norm and resize, B,H,W,N
        features = image_features @ text_features.t()
        similarity_map = clip.get_similarity_map(features[:, 1:, :], img_size)

        feature_per=features[0][0].cpu().numpy()
        perceptual_score=np.exp(feature_per)[0]/np.sum(np.exp(feature_per))
        perceptual_map=similarity_map[0,:,:,0]
        return perceptual_map, perceptual_score

def perceptual_quality_map(pil_img, preprocess, multi=False, draw=True, bound1=1.0, bound2=0.6, bound3=0.8):
    #bound characterize importance of technical, rational, and natural quality
    
    image = preprocess(pil_img).unsqueeze(0).to(device)
    pm0, ps0 = single_quality_map(image,text_embed,pil_img.size)
    pm0 = pm0.cpu().numpy()
    
    if multi:
        ps1 = single_quality_map(image,text_embed_1,pil_img.size)[1]
        ps2 = single_quality_map(image,text_embed_2,pil_img.size)[1]
        ps3 = single_quality_map(image,text_embed_3,pil_img.size)[1]
        
        ### average perference value of AGIN database
        bound1=0.935*bound1
        bound2=0.892*bound2
        bound3=0.826*bound3
        
        # Normalize
        t=min(ps1/bound1,1) * min(ps2/bound1,1) * min(ps3/bound1,1)
        return pm0, ps0*t
    
    else:
        return pm0, ps0




#all_texts=["low quality", "high quality"]
#state_dict = torch.load("D:/SJTUMM/IQA-PyTorch-main/experiments/CLIPIQA_RN50_AGIN_embed_0/models/net_best.pth")
state_dict = torch.load("./embeding/PQ-overall.pth")
state_dict_1 = torch.load("./embeding/PQ-technical.pth")
state_dict_2 = torch.load("./embeding/PQ-rational.pth")
state_dict_3 = torch.load("./embeding/PQ-natural.pth")

text_embed = load_embed(state_dict)
text_embed_1 = load_embed(state_dict_1)
text_embed_2 = load_embed(state_dict_2)
text_embed_3 = load_embed(state_dict_3)   

pos_embed = model.positional_embedding.type(model.dtype)

if __name__ == "__main__":    
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "-p", "--path", type=str, 
        default="perceptual-example.jpg", 
        help="input image path",
    )
    
    parser.add_argument(
        "-d", "--draw", type=bool, 
        default=True, 
        help="draw quality map or not"
    )

    parser.add_argument(
        "-m", "--multi", type=bool, 
        default=False, 
        help="quality multiple dimension"
    )

    
    args = parser.parse_args()
    
    pil_img = Image.open(args.path)

    pm, ps = perceptual_quality_map(pil_img=pil_img, preprocess=preprocessor, draw=args.draw, multi=args.multi)
    print('The quality score is: ' +str(ps))

    if args.draw:
        cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        vis = (pm * 255).astype('uint8')
        vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
        vis = cv2_img * 0.4 + vis * 0.6
        vis = cv2.cvtColor(vis.astype('uint8'), cv2.COLOR_BGR2RGB)
        plt.axis('off')
        plt.imshow(vis)
        plt.savefig('PQ-Map.png', bbox_inches='tight')