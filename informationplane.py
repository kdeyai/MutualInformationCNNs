import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
# matplotlib.rcParams.update({'font.size': 14})
# figsize = (8, 5)


# def plot(train_logs, test_logs, size = figsize):
    
#     plt.figure(1, figsize=size)

#     lists = sorted(train_logs.items())
#     x, y = zip(*lists)
#     plt.plot(x, y, label = 'Training')

#     lists = sorted(test_logs.items()) 
#     x, y = zip(*lists) 
#     plt.plot(x, y, label = 'Testing')

#     plt.ylabel('Accuracy ')
#     plt.xlabel('Number of Epoches')
#     plt.legend()
#     plt.title('Accuracy VS. Number of Epoches')

# def mi_plot(MI_client):
#     en_mis = np.array(MI_client.en_mi_collector)
#     de_mis = np.array(MI_client.de_mi_collector)

#     fig, ax = plt.subplots(figsize=(8, 8))
#     ax.set_ylabel('MI_T,Y')
#     ax.set_xlabel('MI_X,T')
#     title = ax.set_title('Information plane')
#     plt.close(fig)

#     cmap = plt.cm.get_cmap('cool')

#     def plot_point(i):
#         ax.plot(en_mis[i,:], de_mis[i,:], 'k-', alpha=0.2)
#         if i > 0:
#             for j in range(len(en_mis[0])):
#                 ax.plot(en_mis[(i-1):(i+1),j],de_mis[(i-1):(i+1),j],'.-', c = cmap(i*.008), ms = 8)
            
#     for i in range(len(en_mis)):
#         plot_point(i)

#     return fig


# according to other process work in progress

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
sns.set_style('darkgrid')


def plotinformationplane(measures,PLOT_LAYERS):

    plt.figure(figsize=(4,8))
    gs = gridspec.GridSpec(2,len(measures))
    for activation, vals in measures.items():
        epochs = sorted(vals.keys())
            

        #plot bin I(X,M)
        plt.subplot(gs[0,0])
        for lndx, layerid in enumerate(PLOT_LAYERS):
            hbinnedvals = np.array([vals[epoch]['MI_XM_bin'][layerid] for epoch in epochs])
            plt.semilogx(epochs, hbinnedvals, label='Layer %d'%layerid)
    #     plt.ylim([0,15])
        plt.xlabel('Epoch')
        plt.ylabel('I(X;T)')

        
        #plot bin I(X,M)
        plt.subplot(gs[1,0])
        for lndx, layerid in enumerate(PLOT_LAYERS):
            hbinnedvals = np.array([vals[epoch]['MI_YM_bin'][layerid] for epoch in epochs])
            plt.semilogx(epochs, hbinnedvals, label='Layer %d'%layerid)
    #     plt.ylim([0,5])
        plt.xlabel('Epoch')
        plt.ylabel('I(Y;T)')

        
        plt.legend(loc='lower right')
            
    plt.tight_layout()

    plt.savefig('./',bbox_inches='tight')



    max_epoch = max( (max(vals.keys()) if len(vals) else 0) for vals in measures.values())
    sm = plt.cm.ScalarMappable(cmap='gnuplot', norm=plt.Normalize(vmin=0, vmax=500))
    sm._A = []
    infoplane_measures = ['bin', 'upper', 'lower']
    for infoplane_measure in infoplane_measures:
        fig=plt.figure(figsize=(10,5))
        count = 0
        for actndx, (activation, vals) in enumerate(measures.items()):
            epochs = sorted(vals.keys())
            if not len(epochs):
                continue
            plt.subplot(1,2,actndx+1)    
            for epoch in epochs:
                c = sm.to_rgba(epoch)
                xmvals = np.array(vals[epoch]['MI_XM_'+infoplane_measure])[PLOT_LAYERS]
                ymvals = np.array(vals[epoch]['MI_YM_'+infoplane_measure])[PLOT_LAYERS]

                # s = np.argsort(xmvals)
                # xmvals = xmvals[s]
                # ymvals = ymvals[s]
                plt.plot(xmvals, ymvals, c=c, alpha=0.1, zorder=1)
                plt.scatter(xmvals, ymvals, s=20, facecolors=[c for _ in PLOT_LAYERS], edgecolor='none', zorder=2)
            

            plt.xlabel('I(X;M)')
            plt.ylabel('I(Y;M)')
            plt.title(activation)
            
        cbaxes = fig.add_axes([1.0, 0.125, 0.03, 0.8]) 
        plt.colorbar(sm, label='Epoch', cax=cbaxes)
        plt.tight_layout()

        if not os.path.isdir('plots'):
            os.mkdir('plots')
        plt.savefig('plots/' + 'relu'+ '_infoplane_'+infoplane_measure,bbox_inches='tight')
