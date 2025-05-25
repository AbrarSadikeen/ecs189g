#%%
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '..')
sys.path.insert(1, '../../../../data')
sys.path.append("/content/ecs189g/venv/Scripts/local_code/")
sys.path.append("/content/ecs189g/venv/Scripts/")
sys.path.append("/content/ecs189g/data")
sys.path.append("/content/ecs189g/data/stage_4_data")

from local_code.stage_4_code.stage_4_local_classifier import Method_CNN, Evaluate_Metrics, Dataset_Loader
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
    #------------------------------------------------------

    # ---- objection initialization setction ---------------
    train_loader = Dataset_Loader('train', '')
    train_loader.dataset_source_folder_path = 'stage_4_data/text_classification/'
    train_loader.dataset_source_file_name = 'train'
    train_data = train_loader.load()

    test_loader = Dataset_Loader('test', '')
    test_loader.dataset_source_folder_path = 'stage_4_data/text_classification/'
    test_loader.dataset_source_file_name = 'test'
    test_data = test_loader.load()

    data = {'train': train_data, 'test': test_data}

    method_obj = Method_CNN('MNIST classifier', '')
    method_obj.data = data

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
    plt.savefig('result/stage_3_result/metrics.png')
    plt.show()

    plt.figure()
    plt.title('Training curves')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.plot(curves['epochs'],curves['loss'],label='training loss')
    plt.plot(curves['epochs'],curves['test loss'],label='testing loss')
    plt.savefig('result/stage_3_result/loss_curve.png')
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.title('Training curves')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.plot(curves['epochs'],curves['Accuracy'],label='training accuracy')
    plt.plot(curves['epochs'],curves['test accuracy'],label='testing accuracy')
    plt.savefig('result/stage_3_result/acc_curve.png')
    plt.legend()
    plt.show()

    eval = Evaluate_Metrics()
    eval.data = {'true_y': test_results['true_y'], 'pred_y': test_results['pred_y']}

    evals = eval.evaluate()
    print('Test results:')
    for metric in evals.keys():
        print(f"{metric}: {evals[metric]:.4f}", end=", ")

    # plt.figure()
    # plt.title('[Training curves')
    # plt.xlabel('epochs')
    # plt.ylabel('loss')
    # plt.plot(curves['epochs'],curves['loss'],label='training loss')
    # plt.plot(curves['epochs'],curves['test loss'],label='testing loss')
    # #plt.plot(curves['epochs'],curves['Accuracy'],label='training accuracy')
    # #plt.plot(curves['epochs'],curves['test accuracy'],label='testing accuracy')
    # plt.savefig('result/stage_3_result/loss_curve.png')
    # plt.legend()
    # plt.show()

    # plt.figure()
    # plt.title('Training curves')
    # plt.xlabel('epochs')
    # plt.ylabel('accuracy')
    # # plt.plot(curves['epochs'],curves['loss'],label='training loss')
    # # plt.plot(curves['epochs'],curves['test loss'],label='testing loss')
    # plt.plot(curves['epochs'],curves['Accuracy'],label='training accuracy')
    # plt.plot(curves['epochs'],curves['test accuracy'],label='testing accuracy')
    # plt.savefig('result/stage_2_result/loss_curve.png')
    # plt.legend()
    # plt.show()

    for metric in evals.keys():
        print(f"{metric}: {curves[metric][-1]:.4f}", end=", ")

    
# %%
