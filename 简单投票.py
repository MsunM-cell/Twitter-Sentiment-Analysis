import pandas as pd
from collections import Counter

########################## 对20个预测表格进行简单的投票，筛选出频率最高的值，作为最终预测结果 ##############################
aa = pd.read_csv('./predict/predict_by_model_RF_1.csv')
bb = pd.read_csv('./predict/predict_by_model_SVM_1.csv')
print(aa)
print(bb)
print(aa.dtypes)
print(bb.dtypes)
# 左右拼接
for i in range(2, 11):
    aa = pd.merge(aa, pd.read_csv(f'./predict/predict_by_model_RF_{i}.csv'))
for i in range(2, 11):
    bb = pd.merge(bb, pd.read_csv(f'./predict/predict_by_model_SVM_{i}.csv'))

cc = pd.merge(aa,bb)
print(cc)
print(cc.dtypes)
label = cc.columns.drop(['id','text'])
print(label)
all_label = pd.DataFrame(cc, columns=label)
print(all_label)

label_merge = []
for _ in all_label.values:
    c = Counter(_)
    label_merge.append(c.most_common(1)[0][0])
print(label_merge)

result_merge = pd.DataFrame({'id': aa['id'], 'text':aa['text'],'sentiment': label_merge})
result_merge.to_csv('./predict/final_submit.csv', index=None)
