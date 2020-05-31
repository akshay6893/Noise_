import streamlit as st
from io import BytesIO
from PIL import Image
import requests
import base64
from PIL import Image,ImageFile
import torch                                  
from fastai import *                                    
from fastai.vision import * 
from matplotlib import pyplot as plt

st.title("something")

uploaded_file = st.file_uploader("only give in pictures") # pass uploaded_file to open_image
# uploaded_file = uploaded_file.read()


class FeatureLoss(nn.Module):
    def __init__(self, m_feat, layer_ids, layer_wgts):
        super().__init__()
        self.m_feat = m_feat
        self.loss_features = [self.m_feat[i] for i in layer_ids]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts
        self.metric_names = ['pixel',] + [f'feat_{i}' for i in range(len(layer_ids))
              ] + [f'gram_{i}' for i in range(len(layer_ids))]

    def make_features(self, x, clone=False):
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]
    
    def forward(self, input, target):
        out_feat = self.make_features(target, clone=True)
        in_feat = self.make_features(input)
        self.feat_losses = [base_loss(input,target)]
        self.feat_losses += [base_loss(f_in, f_out)*w
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.feat_losses += [base_loss(gram_matrix(f_in), gram_matrix(f_out))*w**2 * 5e3
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.metrics = dict(zip(self.metric_names, self.feat_losses))
        return sum(self.feat_losses)
    
    def __del__(self): self.hooks.remove()




if  uploaded_file :

    # string = (base64.b64encode(uploaded_file))   #.decode('utf-8')

    # r = requests.get(f'http://127.0.0.1:5000/?image={string}')
    # yu = Image.open(BytesIO(r.content))
    model = load_learner(path ,'saved_model576.hdf5')
    img = open_image(uploaded_file)
    # st.image(yu)
    q,w,e = model.predict(img1)    

    st.image(q)
else :
    st.text('Please upload a file of a valid file')