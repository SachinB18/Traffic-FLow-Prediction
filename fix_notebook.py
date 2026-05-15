import json

file_path = r"c:\Users\LOQ 15IRX9\Downloads\DL_Project_2\AIML_Traffic_Flow_Prediction\notebooks\09_Model_Visualization_Interpretability.ipynb"
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

for cell in data['cells']:
    if cell['cell_type'] == 'code':
        for i, line in enumerate(cell['source']):
            if "metrics_lstm = compute_metrics(y_true, y_pred_lstm)" in line:
                cell['source'][i] = line.replace("y_true, y_pred_lstm", "y_true.squeeze(), y_pred_lstm.squeeze()")
            if "metrics_gru = compute_metrics(y_true, y_pred_gru)" in line:
                cell['source'][i] = line.replace("y_true, y_pred_gru", "y_true.squeeze(), y_pred_gru.squeeze()")

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=1)
