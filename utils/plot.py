import torch

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def draw_losses():
    state = torch.load("./saved_models/state.pth", map_location=torch.device('cpu'))
    
    state['rec_loss'] = np.array(state['rec_loss'])


    state['id_loss'] = np.array(state['id_loss'])


    state['att_loss'] = np.array(state['att_loss'])
 

    state['g_loss'] = np.array(state['g_loss'])


    state['d_loss'] = np.array(state['d_loss'])

    if state["iter"] <= 10000:
        df1 = pd.DataFrame(state['rec_loss'], columns=['rec_loss'])
        df2 = pd.DataFrame(state['id_loss'], columns=['id_loss'])
        df3 = pd.DataFrame(state['att_loss'], columns=['att_loss'])
        # df4 = pd.DataFrame(state['rec_features_loss'], columns=['rec_features_loss'])
        df5 = pd.DataFrame(state['g_loss'], columns=['g_loss'])
        df6 = pd.DataFrame(state['d_loss'], columns=['d_loss'])
    else:
        df1 = pd.DataFrame(state['rec_loss'][-10000:], columns=['rec_loss'])
        df2 = pd.DataFrame(state['id_loss'][-10000:], columns=['id_loss'])
        df3 = pd.DataFrame(state['att_loss'][-10000:], columns=['att_loss'])
        # df4 = pd.DataFrame(state['rec_features_loss'], columns=['rec_features_loss'])
        df5 = pd.DataFrame(state['g_loss'][-10000:], columns=['g_loss'])
        df6 = pd.DataFrame(state['d_loss'][-10000:], columns=['d_loss'])

    f1 = plt.figure()
    sns.lineplot(data=df1, hue="rec_loss")
    f1.savefig("./plots/rec_loss.jpg")

    f2 = plt.figure()
    sns.lineplot(data=df2, hue="id_loss")
    f2.savefig("./plots/id_loss.jpg")

    f3 = plt.figure()
    sns.lineplot(data=df3, hue="att_loss")
    f3.savefig("./plots/att_loss.jpg")

    f5 = plt.figure()
    sns.lineplot(data=df5, hue="g_loss")
    f5.savefig("./plots/g_loss.jpg")

    f6 = plt.figure()
    sns.lineplot(data=df6, hue="d_loss")
    f6.savefig("./plots/d_loss.jpg")

    f1.close()
    f2.close()
    f3.close()
    f5.close()
    f6.close()


def draw_losses_DDP():
    state = torch.load("./saved_models/state_DDP.pth", map_location=torch.device('cpu'))
    
    state['rec_loss'] = np.array(state['rec_loss'])


    state['id_loss'] = np.array(state['id_loss'])


    state['att_loss'] = np.array(state['att_loss'])
 

    state['g_loss'] = np.array(state['g_loss'])


    state['d_loss'] = np.array(state['d_loss'])

    if state["iter"] <= 10000:
        df1 = pd.DataFrame(state['rec_loss'], columns=['rec_loss'])
        df2 = pd.DataFrame(state['id_loss'], columns=['id_loss'])
        df3 = pd.DataFrame(state['att_loss'], columns=['att_loss'])
        # df4 = pd.DataFrame(state['rec_features_loss'], columns=['rec_features_loss'])
        df5 = pd.DataFrame(state['g_loss'], columns=['g_loss'])
        df6 = pd.DataFrame(state['d_loss'], columns=['d_loss'])
    else:
        df1 = pd.DataFrame(state['rec_loss'][-10000:], columns=['rec_loss'])
        df2 = pd.DataFrame(state['id_loss'][-10000:], columns=['id_loss'])
        df3 = pd.DataFrame(state['att_loss'][-10000:], columns=['att_loss'])
        # df4 = pd.DataFrame(state['rec_features_loss'], columns=['rec_features_loss'])
        df5 = pd.DataFrame(state['g_loss'][-10000:], columns=['g_loss'])
        df6 = pd.DataFrame(state['d_loss'][-10000:], columns=['d_loss'])

    f1 = plt.figure()
    sns.lineplot(data=df1, hue="rec_loss")
    f1.savefig("./plots/rec_loss_DDP.jpg")

    f2 = plt.figure()
    sns.lineplot(data=df2, hue="id_loss")
    f2.savefig("./plots/id_loss_DDP.jpg")

    f3 = plt.figure()
    sns.lineplot(data=df3, hue="att_loss")
    f3.savefig("./plots/att_loss_DDP.jpg")

    f5 = plt.figure()
    sns.lineplot(data=df5, hue="g_loss")
    f5.savefig("./plots/g_loss_DDP.jpg")

    f6 = plt.figure()
    sns.lineplot(data=df6, hue="d_loss")
    f6.savefig("./plots/d_loss_DDP.jpg")

    f1.close()
    f2.close()
    f3.close()
    f5.close()
    f6.close()

def draw_losses_DDP_unlimit():
    state = torch.load("./saved_models/state_DDP.pth", map_location=torch.device('cpu'))
    
    state['rec_loss'] = np.array(state['rec_loss'])


    state['id_loss'] = np.array(state['id_loss'])


    state['att_loss'] = np.array(state['att_loss'])
 

    state['g_loss'] = np.array(state['g_loss'])


    state['d_loss'] = np.array(state['d_loss'])

    df1 = pd.DataFrame(state['rec_loss'], columns=['rec_loss'])
    df2 = pd.DataFrame(state['id_loss'], columns=['id_loss'])
    df3 = pd.DataFrame(state['att_loss'], columns=['att_loss'])
    # df4 = pd.DataFrame(state['rec_features_loss'], columns=['rec_features_loss'])
    df5 = pd.DataFrame(state['g_loss'], columns=['g_loss'])
    df6 = pd.DataFrame(state['d_loss'], columns=['d_loss'])
   

    f1 = plt.figure()
    sns.lineplot(data=df1, hue="rec_loss")
    f1.savefig("./plots/rec_loss_DDP.jpg")

    f2 = plt.figure()
    sns.lineplot(data=df2, hue="id_loss")
    f2.savefig("./plots/id_loss_DDP.jpg")

    f3 = plt.figure()
    sns.lineplot(data=df3, hue="att_loss")
    f3.savefig("./plots/att_loss_DDP.jpg")

    f5 = plt.figure()
    sns.lineplot(data=df5, hue="g_loss")
    f5.savefig("./plots/g_loss_DDP.jpg")

    f6 = plt.figure()
    sns.lineplot(data=df5, hue="d_loss")
    f6.savefig("./plots/d_loss_DDP.jpg")

    plt.close('all')



if __name__ == "__main__":
    draw_losses_DDP_unlimit()