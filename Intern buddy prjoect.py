#!/usr/bin/env python
# coding: utf-8

# In[208]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics


# In[209]:


os.chdir("C:/Users/rakshith/Desktop/DataSets/Intern_buddy")


# In[210]:


train = pd.read_excel('internbuddy_data_v1.xlsx')


# In[211]:


train.shape


# In[212]:


train.head()


# In[213]:


train.info()


# In[214]:


train.describe()


# # Missing Value Analysis

# In[215]:


missing_df=pd.DataFrame(train.isnull().sum().sort_values(ascending=False)).reset_index()
missing_df.columns = ['Feature','Missing_count']
plt.figure(figsize=(30,10))
sns.barplot(x='Feature',y='Missing_count',data=missing_df)
plt.tight_layout()


# In[216]:


# Will drop unnamed:10 as it contains only null value
# Will drop performance_PG,majority of entries are null values, and most might not have had PG Level of education.
# Will drop performance_10, as it does'nt provide much info about candidate's performance and it also has majority null values.
drop_cols = ['Unnamed: 10','Performance_PG','Performance_10']
train.drop(drop_cols,axis=1,inplace=True)


# In[217]:


# Will convert performance in  UG to a scale of 10.
def ug_convert(perf):
        if perf.split('/')[1] == '100':
            return float(perf.split('/')[0])/100
        else:
            return float(perf.split('/')[0])
            

train['Performance_UG']=train['Performance_UG'].dropna().apply(lambda x:ug_convert(x))            


# In[218]:


# Will convert performance in 12 to a scale of 10
def perf_12_convert(perf):
    return float(perf.split('/')[0])/10


train['Performance_12']=train['Performance_12'].dropna().apply(lambda x:perf_12_convert(x))


# In[219]:


# Will impute the missing data of Performance in 12 and UG with median values
train['Performance_12'].fillna(train['Performance_12'].median(),inplace=True)
train['Performance_UG'].fillna(train['Performance_UG'].median(),inplace=True)


# In[220]:


train.isnull().sum()


# In[221]:


# Handling Missing data in 'Other skills ','Degree','Stream '
train[['Other skills','Degree','Stream']].head()


# In[222]:


# Replacing computer related streams with 'Computer Science & Engineering'
train['Stream'].replace(['Computer Science And Application','Software Engineering','Computer Science And Mathematics',
                       'Computer Science AndEngineering','Computer  Science And Engineering','computer science',
                       'CSE With Cloud Computing','Computer  Science','cs'],'Computer Science & Engineering',inplace=True)


# In[223]:


# Imputing missing value with 'Computer Science & Engineering' as it has the most number.
train['Stream'].fillna('Computer Science & Engineering',inplace=True)


# In[224]:



train[train['Degree'].isnull()]['Stream']


# In[225]:


#Imputing the missing values of degree based on their respective streams
def degree_impute(null_df,df):
    for i in null_df.index:
        if 'Engineering' in null_df['Stream'][i].split():
            df['Degree'].loc[i] = 'Bachelor of Engineering (B.E)'
        elif 'Management' in null_df['Stream'][i].split():
            df['Degree'].loc[i] = 'MBA'
        elif 'Diploma' in null_df['Stream'][i].split():
            df['Degree'].loc[i] = 'Post Graduate Diploma'
        else:
            df['Degree'].loc[i] = 'Bachelor of Technology (B.Tech)'
            

degree_impute(train[train['Degree'].isnull()],train)


# In[226]:


train.isnull().sum()


# In[227]:


#Treating the missing values in 'Other skills'
#Extracting the skills from each individual
skill_list = []
for i in train['Other skills'].dropna().index:
    skills = train['Other skills'][i].split()
    for j in range(len(skills)):
        skill_list.append(skills[j])


# In[228]:


#Checking the Main skills of candidates whose other skills are missing
skill_df=train[train['Other skills'].isnull()][['Python (out of 3)','R Programming (out of 3)','Deep Learning (out of 3)','PHP (out of 3)',
                                       'MySQL (out of 3)','HTML (out of 3)','CSS (out of 3)','JavaScript (out of 3)',
                                     'AJAX (out of 3)','Bootstrap (out of 3)','MongoDB (out of 3)','Node.js (out of 3)','ReactJS (out of 3)']]


# In[229]:


skill_df


# In[230]:


# Replacing missing value with one of the skill provided
for i in skill_df.columns:
    for j in skill_df.index:
        if skill_df[i][j] != 0:
            train['Other skills'].loc[j] = i.split()[0]
            


# In[231]:


train['Other skills'].replace('Deep','Deep Learning',inplace=True)


# In[232]:


train['Other skills'].isnull().sum()  # 6 more null entries for 'Other skills'


# In[233]:


# Since these candidates have 0 scoring on all of the programming features and no entries in the 'Other skill'. will drop these.
poor_skill_indexes = train[train['Other skills'].isnull()].index # will use this for our model evaluation
train.fillna('no relevant skills',inplace = True)


# # Feature Engineering

# In[234]:


# Features to be derived from the existing features are
# Data science score
# Web development score
# other_skills_DS_score
# other_skills_WD_score
# Years after graduating
# Whether holding quantitative degree


# In[235]:


train['DS_score'] = train['Python (out of 3)']+train['R Programming (out of 3)']+train['Deep Learning (out of 3)']+train['MySQL (out of 3)']+train['MongoDB (out of 3)']
train['WD_score'] = train['PHP (out of 3)']+train['HTML (out of 3)']+train['CSS (out of 3)']+train['JavaScript (out of 3)']+train['AJAX (out of 3)']+train['Bootstrap (out of 3)']+train['Node.js (out of 3)']+train['ReactJS (out of 3)']


# In[236]:


ds_skill = ['Data Analytics','Natural Language Processing','Machine Learning','MATLAB','Image Processing',
            'Artificial Intelligence','MS-Excel','Neural Networks','Stastical modelling','Computer vision']

train['otherSkill_ds_score'] = 0
for skill in ds_skill:
    for i in train.index:
        if skill.lower() in train['Other skills'][i].lower():
            train['otherSkill_ds_score'][i] = train['otherSkill_ds_score'][i]+1
        else:
            train['otherSkill_ds_score'][i] = train['otherSkill_ds_score'][i]
    


# In[237]:


wd_skill = ['Data structures','jQuery','C++','Django','Java','AngularJS']

train['otherSkill_wd_score'] = 0
for skill in wd_skill:
    for i in train.index:
        if skill.lower() in train['Other skills'][i].lower():
            train['otherSkill_wd_score'][i] = train['otherSkill_wd_score'][i]+1
        else:
            train['otherSkill_wd_score'][i] = train['otherSkill_wd_score'][i]


# In[238]:


train['Year_after_grad'] = 2020 - train['Current Year Of Graduation']


# In[239]:


train.head()


# In[242]:


train.head()


# In[243]:


# Creating feature to indicate whether they belong to quantitative field or not
quant_degrees = ['Engineering','Technology','Tech','Science']
train['Quantitative_degree'] = 0
for deg in quant_degrees:
    for i in train.index:
        if deg.lower() in train['Degree'][i].lower():
            train['Quantitative_degree'].loc[i]=1
        else:
            pass
        


# In[244]:


train.head()


# # Visualizations

# In[245]:


plt.figure(figsize=(20,10))
faces = sns.FacetGrid(col = 'Quantitative_degree',data = train )
faces.map(plt.hist,'DS_score')


# In[246]:


plt.figure(figsize=(20,10))
faces = sns.FacetGrid(col = 'Quantitative_degree',data = train )
faces.map(plt.hist,'WD_score')


# From the above plots we can conclude that people with quantitative field tend to have higher score in web development and Data Science.

# In[247]:


plt.figure(figsize=(20,10))
faces = sns.FacetGrid(col = 'Quantitative_degree',data = train )
faces.map(plt.scatter,'Performance_12','Performance_UG')


# In[248]:


plt.figure(figsize=(12,6))
sns.distplot(train['Performance_UG'],bins=30,kde=True)


# In[249]:


fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(16,8))
sns.distplot(train['DS_score'],bins=20,ax=ax[0],kde=False)
ax[0].set_title('Data science score distribution')
sns.distplot(train['WD_score'],bins=20,ax=ax[1],kde=False)
ax[1].set_title('Web development score distribution')
plt.tight_layout()


# In[ ]:





# # Correlation amongst features

# In[250]:


plt.figure(figsize=(12,6))
sns.heatmap(train.corr(),cmap='viridis',annot = True)


# In[251]:


train.columns


# In[252]:


# dropping few of the features used for feature engineering along with few other features which does'nt contribute much
drop_var= ['Degree','Current Year Of Graduation','Other skills','Stream','Application_ID','Current City']
train.drop(drop_var,axis=1,inplace=True)


# # Scaling The data

# In[253]:


scaler = MinMaxScaler()
train_scale = scaler.fit_transform(train)


# In[254]:


train_scale.shape


# # Principal component analysis

# In[255]:


var_ratio={}
for n in range(3,21):
    pc=PCA(n_components=n)
    train_pca=pc.fit(train_scale)
    var_ratio[n]=sum(train_pca.explained_variance_ratio_)


# In[256]:


var_ratio


# In[257]:


var_series=pd.Series(var_ratio)
plt.figure(figsize=(10,5))
plt.plot(var_series)


# From above graph we can conclude number of principal components to be 14, as 95% variance in the data is being explained

# In[258]:


pc= PCA(n_components=14).fit(train_scale)
pc_data = pc.transform(train_scale)


# In[259]:


PCA_df = pd.DataFrame(pc_data)
pc_data.shape


# In[260]:


# Variance explained by each of PC's
pd.Series(pc.explained_variance_ratio_,index=['PC_'+ str(i) for i in range(14)]) 


# In[261]:


pc_df=pd.DataFrame(pc.components_.T,columns=['PC_'+str(i) for i in range(14)],index=train.columns)
pc_df


# # Number of clusters

# In[262]:


no_of_clusters=range(1,20)
cluster_errors=[]
for num_clusters in no_of_clusters:
    clusters=KMeans(num_clusters).fit(pc_data)
    cluster_errors.append(clusters.inertia_)


# In[263]:


cluster_df=pd.DataFrame({'cluster_count':no_of_clusters,'error_rate':cluster_errors})


# In[264]:



plt.figure(figsize=(12,6))
plt.plot(cluster_df.cluster_count,cluster_df.error_rate,marker="o")
plt.xlabel('NUMBER OF CLUSTERS')
plt.ylabel('ERROR RATE')


# In[265]:


# Chekcing for silhoutee score and concluding about the number of clusters to be used
no_of_clusters= range(2,10)
silhouette_score=[]
for num_clusters in no_of_clusters:
    cluster_model=KMeans(n_clusters=num_clusters)
    cluster_labels=cluster_model.fit_predict(pc_data)
    silhouette_value=metrics.silhouette_score(pc_data,cluster_labels)
    silhouette_score.append( silhouette_value)
    print('For n_cluster equal to ',num_clusters,' The silhouette score is ',silhouette_value)


# In[266]:


sil_df=pd.DataFrame({'number_of_clusters':range(2,10),'silhouette_score':silhouette_score})


# In[267]:


plt.figure(figsize=(12,6))
plt.plot(sil_df.number_of_clusters,sil_df.silhouette_score,marker="o")
plt.xlabel('NUMBER OF CLUSTERS')
plt.ylabel('ERROR RATE')


# # From Elbow plot and silhouette graph 3 clusters or segments would be formed

# # KMeans with 3 clusters

# In[268]:


#Generating clusters
clusters=KMeans(n_clusters=3,random_state=101).fit(pc_data)
#adding cluster to the pc_df
PCA_df['cluster_group'] = clusters.labels_


# In[269]:


def scatter_plot(feature1,feature2,feature3,feature4):
    fig,ax=plt.subplots(1,2,figsize=(14,6))
    ax[0].scatter(feature1,feature2,c=clusters.labels_,cmap='viridis_r')
    ax[0].set_xlabel('feature1')
    ax[0].set_ylabel('feature2')
    
    ax[1].scatter(feature3,feature4,c=clusters.labels_,cmap='viridis_r')
    ax[1].set_xlabel('feature3')
    ax[1].set_ylabel('feature4')


# In[270]:


scatter_plot(PCA_df.iloc[:,0],PCA_df.iloc[:,1],PCA_df.iloc[:,2],PCA_df.iloc[:,3])


# In[271]:


scatter_plot(PCA_df.iloc[:,0],PCA_df.iloc[:,2],PCA_df.iloc[:,1],PCA_df.iloc[:,3])


# # Assigning the cluster group to the real data

# In[273]:


train['Cluster_group'] = clusters.labels_


# In[274]:


train.head()


# In[275]:


train.groupby('Cluster_group').apply(lambda x:x.mean()).T


# In[291]:


# Lets visualize only with few important features
cols = ['DS_score','WD_score','otherSkill_ds_score','otherSkill_wd_score','JavaScript (out of 3)','Deep Learning (out of 3)']
imp_feat = train.groupby('Cluster_group').apply(lambda x:x[cols].mean()).T
imp_feat


# In[283]:


2.98795/(2.98795+6.638554)


# In[284]:


3.264706/(3.264706+7.220588)


# In[277]:


# Insights from above table
#candidates in cluster 0 have high web development score
#candidates in cluster 1 have high data science score
#candidates in cluster 2 have poor score in both fields


# In[278]:


train[train['Cluster_group'] == 0].shape  # probable Number of candidates for web development = 83


# In[279]:


train[train['Cluster_group'] == 2].shape # probable candidates who might not get short listed = 173


# In[280]:


train[train['Cluster_group'] == 1].shape # Probable number of candidates for data science = 136


# In[289]:


plt.figure(figsize=(16,8))
x_values=np.arange(len(imp_feat.columns))
plt.bar(x_values,imp_feat.loc['DS_score',:].values,color='r',label='DS_score',width=0.1)
plt.bar(x_values+0.1,imp_feat.loc['WD_score',:].values,color='g',label='WD_score',width=0.1)
plt.bar(x_values+0.2,imp_feat.loc['otherSkill_ds_score',:].values,color='b',label='otherSkill_ds_score',width=0.1)
plt.bar(x_values+0.3,imp_feat.loc['otherSkill_wd_score',:].values,color='y',label='otherSkill_wd_score',width=0.1)
plt.bar(x_values+0.4,imp_feat.loc['JavaScript (out of 3)',:].values,color='k',label='JavaScript (out of 3)',width=0.1)
plt.bar(x_values+0.5,imp_feat.loc['Deep Learning (out of 3)',:].values,color='m',label='Deep Learning (out of 3)',width=0.1)


plt.xlabel("Cluster groups")
plt.title("Cluster Insights")
plt.xticks(x_values + 0.2, ('Cl_0', 'Cl_1', 'Cl_2'))
plt.legend(loc=0)


# # 1. Cluster 0
# 1. The score in deep learning is lower than that of cluster1, and score of java script is higher than cluster 1, suggesting that these candidates can be called for interview for the field of web development.

# # 2.Cluster 1
# 1. These candidates have high score in Deep learning and lower score in Javascript, and also has higher data science score compared to cluster 0, Hence These candidates can be called for interview in the field of Data science.

# # 3.Cluster 2
# 1. The candidates belonging to this cluster, can be considered as rejected, since they have poor skill score in both Web development field and Data Science field.

# In[282]:


# checking the perforrmance of model/ saved these indexes since they had poor skills in both departments
train.loc[poor_skill_indexes,'Cluster_group']


# Our model has predicted these candidates as cluster 2, which has candidates with poor skill score, Hence, the model is segmenting candidates clearly.

# In[292]:


not_selected_df = train[train['Cluster_group']==2]
DS_candidates = train[train['Cluster_group']==1]
WD_candidates = train[train['Cluster_group']==0]


# In[293]:


not_selected_df.to_excel('not_selected.xlsx',index=False)
DS_candidates.to_excel('DS_candidates.xlsx',index=False)
WD_candidates.to_excel('WD_candidates.xlsx',index=False)


# # End Notes
# 1. The otherskill score for both fields can be varied, by adding relevant skills, or removing irrelevant one's.(In the feature engineering part)
# 2. Other feature's which includes other programming tools can be considered while visuaizing cluster group.
# 3. Location of the candidate has not been considered for segmenting,since candidates relocating information was not provided.
# 4. By providing Threshold value number of candidates being shortlisted for interview can be controlled.
# 

# In[ ]:




