import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn import svm
from imblearn.over_sampling import SMOTE
from nltk import WordNetLemmatizer
from stop_words import get_stop_words
from string import punctuation
import re
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold #交叉验证
from sklearn.model_selection import GridSearchCV #网格搜索
import _pickle as cPickle




# 1.读取数据
train_df = pd.read_csv('./Twitter_sentiment_selfdrive_DFE.csv')
train_df = train_df.drop(['_unit_id', '_golden', '_unit_state', '_trusted_judgments', '_last_judgment_at', 'sentiment:confidence', 'our_id', 'sentiment_gold', 'sentiment_gold_reason'], axis=1)
train_df = train_df[train_df['sentiment'] != 'not_relevant']
train_df['sentiment'] = train_df['sentiment'].astype('int64')
test_df = pd.read_csv('./test.csv')
print(train_df,'\n')
print(train_df['sentiment'].value_counts())

#2.合并在一起准备做数据清洗
combi = train_df.append(test_df, ignore_index=True,)
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
    return input_txt
sw = set(get_stop_words("en"))
sw.add('user')
puncts = set(punctuation)
lemmatizer = WordNetLemmatizer()
def preprocess_text(txt):
    txt = str(txt)
    txt = "".join(c for c in txt if c not in puncts)
    txt = txt.lower()
    txt = re.sub("not\s", "no", txt)
    txt = [lemmatizer.lemmatize(word) for word in txt.split() if word not in sw]
    return " ".join(txt)
# 去除@用户名
combi['tidy_text'] = np.vectorize(remove_pattern)(combi['text'], "@[\w]*")
combi['tidy_text'] = combi['tidy_text'].replace('@[^\s]+', '', regex=True)
combi['tidy_text'] = combi['tidy_text'].replace('@ [^\s]+', '', regex=True)
#转小写
combi['tidy_text'] = combi['tidy_text'].str.lower()
#去除url
combi['tidy_text'] = combi['tidy_text'].replace(r'((www\.[^\s]+)|(https?://[^\s]+))', '', regex=True)
# 去除标点符号，数字和特殊字符
combi['tidy_text'] = combi['tidy_text'].str.replace("[^a-zA-Z#]", " ")
combi['tidy_text'] = combi['tidy_text'].replace('([^\s\w-]|_)+', "", regex=True)
combi['tidy_text'] = combi['tidy_text'].replace('ì|¡||ü|è|ï|å|ã|_|ì|±|¢|‰|â|å||Ì|¢|‰|Â|Ò|ò|û|ª', " ", regex=True)
combi['tidy_text'] = combi['tidy_text'].replace('-', " ", regex=True)
# 缩写扩展
combi['tidy_text'] = combi['tidy_text'].replace(r"won't", "will not", regex=True)
combi['tidy_text'] = combi['tidy_text'].replace(r"can\'t", "can not", regex=True)
combi['tidy_text'] = combi['tidy_text'].replace(r"n\'t", " not", regex=True)
combi['tidy_text'] = combi['tidy_text'].replace(r"\'re", " are", regex=True)
combi['tidy_text'] = combi['tidy_text'].replace(r"\'s", " is", regex=True)
combi['tidy_text'] = combi['tidy_text'].replace(r"\'d", " would", regex=True)
combi['tidy_text'] = combi['tidy_text'].replace(r"\'ll", " will", regex=True)
combi['tidy_text'] = combi['tidy_text'].replace(r"\'t", " not", regex=True)
combi['tidy_text'] = combi['tidy_text'].replace(r"\'ve", " have", regex=True)
combi['tidy_text'] = combi['tidy_text'].replace(r"\'m", " am", regex=True)
# 移除短单词
combi['tidy_text'] = combi['tidy_text'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
# 去停用词
# 从停用词中保留nor not no
stop_words = set(stopwords.words('english')) - {"nor", "not", "no"}
stopwords_re = re.compile(r"(\s+)?\b({})\b(\s+)?".format("|".join(stop_words), re.IGNORECASE))
whitespace_re = re.compile(r"\s+")
combi['tidy_text'] = combi['tidy_text'].replace(stopwords_re, " ").str.strip().str.replace(whitespace_re, " ")
combi['tidy_text'] = combi['tidy_text'].apply(preprocess_text)
print(combi.head())
# 保存
# combi.to_csv('combi1.csv',index=False)
# print(combi['tidy_text'][0])
# print(combi['tidy_text'][6676])
# print(combi['tidy_text'][6677])
# print(combi['tidy_text'][6678])
# print(combi['tidy_text'][6679])

# 3.tfidf提取特征
tfidf_vectorizer = TfidfVectorizer(max_df=0.9, min_df=1,  stop_words='english')
xdata = tfidf_vectorizer.fit_transform(combi['tidy_text'][0:6677])
ydata = train_df['sentiment']
# smote使得类别分布均衡
sm = SMOTE(random_state = 2022, n_jobs = -1)
xdata, ydata = sm.fit_resample(xdata, ydata)
print(ydata.value_counts())

# 4.划分训练集、验证集、测试集
xtrain, xvalid, ytrain, yvalid = train_test_split(xdata, ydata, random_state=2022, test_size=0.1)
xtest = tfidf_vectorizer.transform(combi['tidy_text'][6677:])
print('xtrain.shape:',xtrain.shape)
print('xvalid.shape:',xvalid.shape)
print('ytrain.shape:',ytrain.shape)
print('yvalid.shape:',yvalid.shape)
print('xtest.shape:',xtest.shape)

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=2022)  # 10折交叉验证


# 5.构建svm模型
# svm10折交叉,生成10个预测文件保存在predict下
i = 0
for train_index, valid_index in kfold.split(xdata, ydata):
        print("\nFold {}".format(i + 1))
        train_x, val_x = xdata[train_index],xdata[valid_index]
        train_y, val_y = ydata[train_index],ydata[valid_index]
        model = svm.SVC(C=1.1)
        model.fit(train_x,train_y)
        # with open(f'./model/SVM_{i+1}.pkl', 'wb') as f:
        #     cPickle.dump(model, f)
        ypred_valid = model.predict(val_x)
        print("预测结果",ypred_valid)
        # 验证集效果检验
        print(metrics.classification_report(val_y, ypred_valid))
        print("准确率:", metrics.accuracy_score(val_y, ypred_valid))
        # 预测测试集
        ypred_test = model.predict(xtest)
        # 保存结果
        new_test_df = pd.DataFrame({'id':test_df['id'],'text':test_df['text'],f'sentiment_SVM_{i+1}':ypred_test})
        new_test_df.to_csv(f'./predict/predict_by_model_SVM_{i+1}.csv',index=False)
        i += 1



# 6. 随机森林10折交叉,生成10个预测文件保存在predict下
i = 0
for train_index, valid_index in kfold.split(xdata, ydata):
        print("\nFold {}".format(i + 1))
        train_x, val_x = xdata[train_index],xdata[valid_index]
        train_y, val_y = ydata[train_index],ydata[valid_index]
        model = RandomForestClassifier(random_state=2022,n_estimators=500)
        model.fit(train_x,train_y)
        # with open(f'./model/RF_{i+1}.pkl', 'wb') as f:
        #     cPickle.dump(model, f)
        ypred_valid = model.predict(val_x)
        print("预测结果",ypred_valid)
        # 验证集效果检验
        print(metrics.classification_report(val_y, ypred_valid))
        print("准确率:", metrics.accuracy_score(val_y, ypred_valid))
        # 预测测试集
        ypred_test = model.predict(xtest)
        # 保存结果
        new_test_df = pd.DataFrame({'id': test_df['id'], 'text': test_df['text'], f'sentiment_RF_{i+1}': ypred_test})
        new_test_df.to_csv(f'./predict/predict_by_model_RF_{i + 1}.csv', index=False)
        i += 1






