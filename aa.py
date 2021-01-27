import pandas as pd
import nltk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sns
from collections import Counter
from datetime import datetime
from wordcloud import WordCloud, STOPWORDS
from nltk.metrics.distance import jaccard_distance
from nltk.tokenize import sent_tokenize , word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

def ScrFileProcessor (InpuSrcStr):
	OutSrcList=[]
	for SrcFile in InpuSrcStr.split(';'):
		if "lhsj_main/" in SrcFile:
			OutSrcList.append(SrcFile)
	if len(OutSrcList) != 0:
		OutSrcListStr=';'.join(sorted(OutSrcList))
	else:
		OutSrcListStr = ''
	return OutSrcListStr

def basicHistograms(df):

    #plotModuleCount = False
    #plotSourceCount = False

    #if (plotModuleCount):
    mainmodulecounts = Counter(df.Module)
    data = pd.DataFrame.from_dict(mainmodulecounts, orient='index')
    ax = data.plot(kind='bar',fontsize=6)
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005),rotation='horizontal',fontsize=6,transform=ax.transAxes)
    plt.show()

    #if (plotSourceCount):
#     uniquemodules = df.Module.unique()
#     df.groupby(["Module","Source Files"])

#     mainmodulecounts = Counter(df["Source Files"])
#     data = pd.DataFrame.from_dict(mainmodulecounts, orient='index')
#     ax = data.plot(kind='bar',fontsize=6)
#     for p in ax.patches:
#         ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005),rotation='horizontal',fontsize=6,transform=ax.transAxes)
#     plt.show()

def lemmatization(X):
    #lemmatization function, received a dataFrame, finds lemmatized text for each line of the dataFrame
    # and returns a new dataFrame with lemmatized values
    #print("Observation column (before lemmatization) shape", X.shape)
    #rowCount=X.shape[0]
    customizedStopWords=['and\'', 'is\'', 'the\'', 'The\'','for\'','with\'', '_x000']
    
    lemmatizedX = np.empty([len(X)], dtype=object) 
    #just to test:
    #print("lemmatizedX shape",lemmatizedX.shape)
    wordnet_lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    separator=' '
    processedText=''
    for i in range(0,len(X)):
        t = X[i]
        processedText=''
        #building the sets
        list_d1 = []  
        s1 = sent_tokenize(t)
        for s in s1:
            w1 = word_tokenize(s)
            for w in w1:
                if(len(w) > 1 and w.isalpha() and w.lower not in stop_words ):
                    lem1 = wordnet_lemmatizer.lemmatize(w)
                    list_d1.append(lem1)
        processedText=separator.join(list_d1)
        lemmatizedX[i]=processedText
    #just to test:
    # print("unlemmatized entry",X[2])
    # print("lemmatized entry",lemmatizedX[2])
    #print("lemmatized Observation column shape", lemmatizedX.shape)
    return lemmatizedX

def calculateJaccardDistance(X):
    #Calculate Jaccard distance of Obsevation Column
    #row, column=X.shape
    row=column=X.shape[0]
    print("In calculateJaccardDistance Function, row & column:", row , column)
    #for jaccardDistance, we need set of words in each entry of X, not just lemmatized text
    text_sets=[]
    for i in range(0, len(X)):
        list_d=[]
        w1=word_tokenize(X[i])
        for w in w1:
            list_d.append(w)
        text_sets.append(set(list_d))
    
    #text_sets=lemmatization(X["Observation.Observation"])
    #Find min and max distance as well.
    maxd = 0
    mind = 1
    mini = -1
    minj = -1
    maxi = -1
    maxj = -1

    #Calculate Jaccard distance for all tokenized and lemmatized words. Jaccard distance between each entry of "Obsevation" Column. 
    dist = np.empty(shape=(len(text_sets),len(text_sets)))# Matrix is initialized with float zeros.
    #Print just to test data
    print("5 rows and columns from the initial matrix:\n")
    print(dist[0:5,0:5])

    for i in range(len(text_sets)-1):
        for j in range(i+1,len(text_sets)): 
            dist[i,j] = jaccard_distance(text_sets[i],text_sets[j])
            dist[j,i] = jaccard_distance(text_sets[i],text_sets[j])
            if(dist[i,j] > maxd):
                maxd = dist[i,j]
                maxi = i
                maxj = j
            if(dist[i,j] < mind):
                mind = dist[i,j]
                mini = i
                minj = j
    #Print just to test data 
    print("print 5 rows and columns from the matrix:\n")     
    print(dist[0:5,0:5])
    print("closest entries are dist[i,j] for i , j " , mind ,mini, minj)
    print("most different entries are d[i,j] for i, j" , maxd , maxi , maxj)
    return dist

def modelSelection():
    models = [RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
              LinearSVC(),
              MultinomialNB(),
              LogisticRegression(random_state=0),]
    CV = 5
    cv_df = pd.DataFrame(index=range(CV * len(models)))
    entries = []
    for model in models:
      model_name = model.__class__.__name__
      accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
      for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))
    cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

    import seaborn as sns

    sns.boxplot(x='model_name', y='accuracy', data=cv_df)
    sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
                  size=8, jitter=True, edgecolor="gray", linewidth=2)
    plt.show()
    cv_df.groupby('model_name').accuracy.mean()

df=pd.read_excel("./data/search-2019-09-13(9578)-DATA SET.xlsx").fillna("")
##### CleanUp Started ########
df.loc[df['Main Module']==';', 'Main Module']=""
df.loc[df['Source Files']==';', 'Source Files']=""
df['Source Files']=df['Source Files'].apply(ScrFileProcessor)
df=df.loc[df['Observation.Observation']!='Error: Data too large for Excel cell']
df=df.groupby(['General.Eriref', 'Observation.Observation', 'Main Module','Source Files']).size().reset_index(name='count').drop('count', axis=1)
df=df.loc[df['Main Module']=='kw'].head(30).append(df.loc[df['Main Module']!='kw'])
df=df.loc[df['Main Module']!='']
df['ModSrc']=df['Main Module']+'#'+df['Source Files']
##### Clean Up Ended ########

# Renaming 'Observation.Observation' to 'Observation' and 'Main Module' to 'Module' for simplification
df.rename(columns = {'Observation.Observation':'Observation','Main Module':'Module'}, inplace = True)

# Applying proper indices
df.index = range(2917)

# list of all the modules !!!be auto
keyword_list = ['cms', 'cx', 'dab', 'bch', 'servlib', 'fms', 'pms', 'ax', 'gmd', 'mih', 'func_frmwk_srv', 'suf', 'account', 'Px', 'wms', 'DocGenLib', 'wab', 'func_frmwk_clt', 'wmx', 'admx', 'rlh', 'dch', 'common', 'kw', 'bgh', 'mx', 'mla', 'lacalib', 'billsrv', 'bat', 'Ei', 'data', 'fuomlib', 'rih', 'dcx', 'wsi', 'Rd', 'rp', 'as', 'Ra', 'tax', 'fih', 'dcs', 'utility', 'udrlib', 'wma', 'teh', 'dxlib', 'urh', 'pih', 'pth', 'ceb', 'foh', 'cerm', 'secServer', 'func_frmwk_cmn', 'java.x', 'cah', 'func_util', 'bee', 'jcil', 'judrlib', 'Ta', 'esh', 'uch_k', 'omx', 'opSupport', 'BAggLib', 'XMLRPCSimulator', 'Gl', 'bjx', 'ioh', 'taplib', 'oms', 'cs_rer', 'pbm', 'pwdmgr', 'ppth', 'auth', 'rhel72_x86.x', 'prepare_bscs_server.sh', 'birdlib', 'licadp', 'scheduler', 'cdh', 'cab', 'csadapters', 'remote', 'udrag', 'rch', 'license', 'jdxlib', 'gdh', 'vmdrep', 'pbs', 'rhel65_x86.x', 'aih', 'adp600', 'rulelib', 'include', 'docgen', 'bumt', 'udmaplib']

# Adding a column to encode the Main Module as an integer because categorical variables are often better represented by integers than strings.
# We also create a couple of dictionaries for future use.
df['module_id'] = df['Module'].factorize()[0]
module_id_df = df[['Module', 'module_id']].drop_duplicates().sort_values('module_id')
module_to_id = dict(module_id_df.values)
id_module_df = df[['module_id', 'Module']]
id_to_module = dict(id_module_df.values)

basicHistograms(df)
#display(df)

# Generate wordcloud image before lemmatization
allwords = WordCloud(stopwords=STOPWORDS).generate(str(df["Observation"].values))
plt.imshow(allwords, interpolation="bilinear")
plt.axis("off")
plt.show()

# Applying lemmetization on the Observation column
start = datetime.now()
text_sets=lemmatization(df["Observation"])
df["Observation"] = text_sets
end = datetime.now()
print("Lemmatization took: ", (end-start))

#Generate wordcloud image after lemmatization:
allwords = WordCloud(stopwords=STOPWORDS).generate(str(text_sets))
plt.imshow(allwords, interpolation="bilinear")
plt.axis("off")
plt.show()
#display(df)

# for each term in our dataset, we will calculate a measure called Term Frequency, Inverse Document Frequency, abbreviated to tf-idf
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')

features = tfidf.fit_transform(df.Observation).toarray()
labels = df.Module
print(features.shape)
print(labels.shape)
#display(df)

modelSelection()


###### Train Test Split Started ###########
##### Model Optimization needs to be done by 'Main Module' on Train Set ----- (X_Train, y_trainTune)
##### Tuned Model needs to be fit by 'ModSrc' on TrainSet  --------- (X_Train, y_trainFit)
##### Fitted Model above needs to be used on Test Set over 'ModSrc'  -------  (X_Test, y_Test)
##### All performance measurements needs to be done over Test set using 'ModSrc'
label=['Module', 'ModSrc']
X_trn, X_tes, y_trn, y_tes = train_test_split(df['Observation'], df[label], test_size = 0.3)

print("train", X_trn.shape, y_trn.shape)
print("test", X_tes.shape, y_tes.shape)
X_Train=X_trn
y_trainTune=y_trn['Module']
y_trainFit=y_trn['ModSrc']
X_Test=X_tes
y_Test=y_tes['ModSrc']

print("train", X_Train.shape, y_trainTune.shape, y_trainFit.shape)
print("test", X_Test.shape, y_Test.shape)
###### Train Test Split Done ###########

# X_train, X_test, y_train, y_test = train_test_split(df['Observation'], df['Module'], random_state = 0)
X_train, X_test, y_train, y_test = train_test_split(X_Train, y_trainTune,test_size = 0.33, random_state = 0)
nb = Pipeline([('vect', CountVectorizer()),
              ('tfidf', TfidfTransformer()),
              ('clf', MultinomialNB()),
              ])

nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)

print("train", X_Train.shape, y_train.shape)
print("test", X_Test.shape, y_Test.shape)

print('accuracy %s' % accuracy_score(y_pred, y_test))
#print(classification_report(y_test, y_pred, labels=keyword_list, target_names=keyword_list))

# X_train, X_test, y_train, y_test = train_test_split(features, labels,test_size = 0.33, random_state = 0)
# clf = Pipeline([('vect', CountVectorizer()),
#               ('tfidf', TfidfTransformer()),
#               ('clf', LinearSVC(random_state=0, tol=1e-5)),
#               ])

# #clf.fit(X_train,y_train)
# clf.fit(X_train,y_train)
# y_pred = clf.predict(X_test)

# print('accuracy %s' % accuracy_score(y_pred, y_test))
# #print(classification_report(y_test, y_pred, labels=keyword_list, target_names=keyword_list))

model = LinearSVC()

train_X, test_X, train_y, test_y, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.33, random_state=0)
model.fit(train_X, train_y)
y_pred = model.predict(test_X)
from sklearn import metrics
print('accuracy %s' % accuracy_score(y_pred, test_y))
#print(metrics.classification_report(test_y, y_pred, labels=keyword_list, target_names=df['Module'].unique()))

conf_mat = confusion_matrix(test_y, y_pred)
print(test_y.shape)
print(y_pred.shape)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=module_id_df.Module.values, yticklabels=module_id_df.Module.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

from IPython.display import display
col = ['Module', 'Observation']
col
for predicted in module_id_df.module_id[:86]:
    for actual in module_id_df.module_id[:86]:
        if predicted != actual and conf_mat[actual, predicted] >= 2:
            print("'{}' predicted as '{}' : {} examples.".format(id_to_module[actual], id_to_module[predicted],
                                                                conf_mat[actual,predicted]))
            #display(df.loc[indices_test[(y_test == actual) & (y_pred == predicted)]][['Module', 'Observation']])
            print('')

sgd = Pipeline([('vect', CountVectorizer()),
              ('tfidf', TfidfTransformer()),
              ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5,tol=None)),
              ])

sgd.fit(X_train,y_train)
y_pred = sgd.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
#print(classification_report(y_test, y_pred, labels=keyword_list, target_names=keyword_list))

logreg = Pipeline([('vect', CountVectorizer()),
              ('tfidf', TfidfTransformer()),
              ('clf', LogisticRegression(n_jobs=1, C=1e5, max_iter = 5)),
              ])
logreg.fit(X_train,y_train)

y_pred = logreg.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
#print(classification_report(y_test, y_pred, labels=keyword_list, target_names=keyword_list))

features = calculateJaccardDistance(df["Observation"])
labels = df["Module"]

train_X, test_X, train_y, test_y, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.33, random_state=0)

knn = Pipeline([("classification",KNeighborsClassifier(n_neighbors=1))])
knn.fit(train_X, train_y)

y_pred = knn.predict(test_X)
 
print('accuracy %s' % accuracy_score(y_pred, test_y))
print(classification_report(test_y, y_pred, labels=keyword_list, target_names=df['Module'].unique()))

conf_mat = confusion_matrix(y_pred, test_y)
 
pyplot.figure(figsize=(8,8))
pyplot.clf()
sns.set()
sns.heatmap(conf_mat.T,square=True,annot=False,xticklabels=module_id_df.Module.values,yticklabels=module_id_df.Module.values,cbar=True,vmin=0, vmax=np.max(conf_mat),cmap='Blues',fmt="d")
pyplot.xlabel("True Label")
pyplot.ylabel("Predicted Label")
pyplot.title(type, pad='25')
pyplot.tight_layout()
pyplot.show()
