#%%
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/content/ecs189g/venv/Scripts')
sys.path.insert(1, '/content/drive')
sys.path.append('/content/ecs189g/venv/Scripts/local_code/stage_5_code')
sys.path.append('/content/ecs189g/venv/Scripts/local_code/stage_5_code/import_this.py')
from import_this import Method_GCN, Evaluate_Metrics, Dataset_Loader
#import stage_3_data.script_data_loader
import numpy as np
import torch
import matplotlib.pyplot as plt
'''
Concrete IO class for a specific dataset

'''


#---- Multi-Layer Perceptron script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    test_split = 0.9
    itr = 2
    #------------------------------------------------------


    # ---- objection initialization setction ---------------
    data_loader = Dataset_Loader()
    data_loader.dataset_source_folder_path = 'stage_5_data/'
    data_loader.dataset_source_file_name = 'pubmed'
    data = data_loader.load()
    perm = torch.randperm(len(data['ids']))
    ids_train = perm[:int(test_split*len(perm))].to(int)
    #ids_train = torch.choice(data['ids'], int(0.5*len(data['ids'])), replacement=False)
    ids_test = torch.ones(data['ids'].shape) == 1
    ids_test[ids_train]  = False
    ids_test = data['ids'][ids_test].to(int)

    # data["ids"][data["ids"] not in ids_train]
    method_obj = Method_GCN('node classifier', '', data=data, max_epoch=250, learning_rate=0.01, update_lr=0.7, use_bias=False, hidden_dim_1=64, hidden_dim_2=16, dropout=0.1, weight_decay=0.0005, num_layers=3)
    method_obj.ids_train = ids_train
    method_obj.ids_test = ids_test

    test_results = method_obj.run()
    curves = test_results['curves']

    plt.figure()
    metrics = ['Accuracy', 'F1 micro', 'F1 macro', 'F1 weighted', 'Precision micro', 'Precision macro', 'Precision weighted', 'Recall micro', 'Recall macro', 'Recall weighted']
    for metric in metrics:
        plt.plot(curves['epochs'],curves[metric], label=metric)
    plt.title('Evaluation metrics on training data')
    plt.xlabel('epochs')
    plt.ylabel('value')
    plt.legend()
    plt.savefig(f'/content/drive/result/stage_5_result/metrics_{itr}.png')
    #plt.show()

    plt.figure()
    plt.title('Training curves')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.plot(curves['epochs'],curves['loss'],label='training loss')
    plt.plot(curves['epochs'],curves['test loss'],label='testing loss')
    plt.savefig(f'/content/drive/result/stage_5_result/loss_curve_{itr}.png')
    plt.legend()
    #plt.show()
    
    plt.figure()
    plt.title('Training curves')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.plot(curves['epochs'],curves['Accuracy'],label='training accuracy')
    plt.plot(curves['epochs'],curves['test accuracy'],label='testing accuracy')
    plt.savefig(f'/content/drive/result/stage_5_result/acc_curve_{itr}.png')
    plt.legend()
    #plt.show()

    eval = Evaluate_Metrics()
    eval.data = {'true_y': test_results['true_y'], 'pred_y': test_results['pred_y']}

    evals = eval.evaluate()
    print('Test results:')
    for metric in evals.keys():
        print(f"{metric}: {evals[metric]:.4f}", end=", ")

    for metric in evals.keys():
        print(f"{metric}: {curves[metric][-1]:.4f}", end=", ")

    
# %%
