import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, 'C:/Users/ataki/Documents/ECS189G_Winter_2025_Source_Code_Template/venv/Scripts')

from local_code.stage_1_code.Result_Loader import Result_Loader

if 1:
    result_obj = Result_Loader('saver', '')
    result_obj.result_destination_folder_path = 'C:/Users/ataki/Documents/ECS189G_Winter_2025_Source_Code_Template/result/stage_1_result/SVM_'
    result_obj.result_destination_file_name = 'prediction_result'

    for fold_count in [1, 2, 3, None]:
        result_obj.fold_count = fold_count
        result_obj.load()
        print('Fold:', fold_count, ', Result:', result_obj.data)