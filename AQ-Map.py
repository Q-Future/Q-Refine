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


import spacy
from nltk import Tree
import string
from nltk.corpus import stopwords

def partition(sentence, noun_phrases):
    if len(noun_phrases) == 1 or len(noun_phrases) == 0:
        return [sentence] 
    parts = []
    cur_part_index = 0
    cur_find_begin_index = 0
    for phrase_idx, cur_noun_phrase in enumerate(noun_phrases):
        find_index = sentence.find(cur_noun_phrase, cur_find_begin_index)
        assert find_index != -1
        if phrase_idx == 0:
            cur_part_index = 0
            cur_find_begin_index = find_index + len(cur_noun_phrase)
        elif phrase_idx == len(noun_phrases) - 1:
            parts.append(sentence[cur_part_index:find_index].strip())
            cur_part_index = find_index
            parts.append(sentence[cur_part_index:].strip())
        else:
            parts.append(sentence[cur_part_index:find_index].strip())
            cur_part_index = find_index
            cur_find_begin_index = find_index + len(cur_noun_phrase)
    return parts

def remove_stopwords_and_punctuation(phrases):
    stop = set(stopwords.words('english')) 
    new_phrases = []
    for phrase in phrases:
        new_phrase= ' '.join([w for w in phrase.split(' ') if w not in stopwords.words('english') and w not in string.punctuation])
        new_phrases.append(new_phrase)
    return new_phrases

def to_nltk_tree(node):
    # print(node)
    # print(list(node.children))
    # print(node.pos_)
    if node.n_lefts + node.n_rights > 0:
        return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
    else:
        return node.orth_

def to_noun_tree(node):
    
    if node.pos_ == "NOUN" or node.pos_ == "PROPN":
        if node.n_lefts + node.n_rights > 0:
            return Tree(node.orth_, [to_noun_tree(child) for child in node.children])
        else:
            return node.orth_
    else:
        # return [to_noun_tree(child) for child in node.children]
        if node.n_lefts + node.n_rights > 0:
            if len(list(node.ancestors))>0:
                # return Tree(list(node.ancestors)[0].orth_, [to_noun_tree(child) for child in node.children])
                return Tree("None", [to_noun_tree(child) for child in node.children])
            else:
                return Tree(node.orth_, [to_noun_tree(child) for child in node.children])
        else:
            # return []   
            return None
        # return Tree([to_noun_tree(child) for child in node.children])
    
def get_phrase_level(sentence, phrases, nltk_tree):
    words = sentence.strip().split()
    # words_to_level = {}
    node_queue = [nltk_tree.root]
    words_to_level = {nltk_tree.root: 0}
    while node_queue:
        cur_node = node_queue[0]
        for child in list(cur_node.children):
            node_queue.append(child)
            words_to_level[child] =  words_to_level[cur_node] + 1
        node_queue = node_queue[1:]
    # print(words_to_level)
    return words_to_level

def get_token_to_pos_dictionary(doc):
    token_to_pos_dictionary = {}
    for token in doc:
        token_to_pos_dictionary[token.text] = token.pos_
    return token_to_pos_dictionary


def get_phrase_parent(sentence, phrases, filtered_phrases, nltk_tree, token_to_pos_dictionary, spacy_nlp):
#     print(token_to_pos_dictionary)
    # words = sentence.strip().split()
    phrase_to_parent = {}
    word_to_phrase = {}
    for phrase, filtered_phrase in zip(phrases, filtered_phrases):
        phrase_to_parent[filtered_phrase] = filtered_phrase
        # for word in phrase.strip().split():
        for token in spacy_nlp(phrase.strip()):
            word_to_phrase[token.text] = filtered_phrase
    # print(word_to_phrase)
    node_queue = [nltk_tree.root]
    
    while node_queue:
        cur_node = node_queue[0]
        # print(f"{cur_node.text} {list(cur_node.ancestors)}") 
        cur_phrase = word_to_phrase[str(cur_node)]
        for child in list(cur_node.children):
            node_queue.append(child)
            if word_to_phrase[str(list(child.ancestors)[0])] != word_to_phrase[str(child)]:
                # print(f"{child.text} {list(child.ancestors)}") 
                for cur_ancestor in list(child.ancestors):
                    # print(f"{str(cur_ancestor)}: {token_to_pos_dictionary[str(cur_ancestor)]}")
                    if token_to_pos_dictionary[str(cur_ancestor)] == "NOUN" or token_to_pos_dictionary[str(cur_ancestor)] == "PROPN":
                        phrase_to_parent[word_to_phrase[str(child)]] = word_to_phrase[str(cur_ancestor)]
                        break
        node_queue = node_queue[1:]
    return phrase_to_parent






'''
# download model:
import spacy.cli 
spacy.cli.download("en_core_web_md")

# or:
download: https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz
pip install /path/to/en_core_web_sm-2.2.0.tar.gz
'''
# nlp = spacy.load('en')

def alignment_map(pil_img,preprocess,texts,red=[""],draw=False):
    with torch.no_grad():
        # CLIP architecture surgery acts on the image encoder
        cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        image = preprocess(pil_img).unsqueeze(0).to(device)
        image_features = model.encode_image(image)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)

        # Prompt ensemble for text features with normalization
        text_features = clip.encode_text_with_prompt_ensemble(model, texts, device)

        # Extract redundant features from an empty string
        redundant_features = clip.encode_text_with_prompt_ensemble(model, red, device)
        
        features = image_features @ (text_features-redundant_features).t()
        similarity_map = clip.get_similarity_map(features[:, 1:, :], cv2_img.shape[:2])

        feature_ali=features[0][0].cpu().numpy()
        alignment_score=np.exp(10*feature_ali)/(np.exp(10*feature_ali)+1)
        
        
        # similarity = clip.clip_feature_surgery(image_features, text_features)
        similarity_map = clip.get_similarity_map(features[:, 1:, :], cv2_img.shape[:2])
        
        am=[]
        as0=alignment_score
        
        for num in range(len(texts)):
            if draw:
                cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                vis = (similarity_map[0,:,:,num] * 255).cpu().numpy().astype('uint8')
                vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
                vis = cv2_img * 0.4 + vis * 0.6
                vis = cv2.cvtColor(vis.astype('uint8'), cv2.COLOR_BGR2RGB)
                print(texts[num])
                plt.axis('off')
                plt.imshow(vis)
                plt.show()
            am.append((similarity_map[0, :, :, num].cpu().numpy() * 255).astype('uint8'))

    return am,as0



if __name__ == "__main__":    
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "-p", "--path", type=str, 
        default="alignment-example.jpg", 
        help="input image path",
    )
    
    parser.add_argument(
        "-d", "--draw", type=bool, 
        default=True, 
        help="draw quality map or not"
    )

    parser.add_argument(
        "-q", "--query", type=str, 
        default="Mr. Beans wearing sun glasses with blue doctor suit and stripe tie", 
        help="prompt"
    )
    args = parser.parse_args()

    nlp = spacy.load('en_core_web_sm')
    query = args.query
    doc = nlp(query)


    noun_phrases = [chunk.text for chunk in doc.noun_chunks]

    parts = partition(query, noun_phrases)

    filtered_parts = remove_stopwords_and_punctuation(parts)


    token_to_pos_dictionary = get_token_to_pos_dictionary(doc)

    phrase_to_parent_dictionary = get_phrase_parent(query, parts, filtered_parts, list(doc.sents)[0], token_to_pos_dictionary, nlp)


    noun_parts=[]
    for filtered_part in filtered_parts:
        tmp=''
        for item in filtered_part.split(' '):
            if token_to_pos_dictionary[item]=='NOUN' or token_to_pos_dictionary[item]=='PROPN':
                tmp=item
        noun_parts.append(tmp)

    
    
    
    pil_img = Image.open(args.path)

    am,as0=alignment_map(pil_img=pil_img, preprocess=preprocessor,texts=noun_parts,red=[""],draw=args.draw)
    _,as1=alignment_map(pil_img=pil_img, preprocess=preprocessor,texts=filtered_parts,red=[""],draw=False)
    print('The alignment score is: ' +str(as0))
