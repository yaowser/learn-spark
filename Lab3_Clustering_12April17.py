
# coding: utf-8

# # Business Understanding
# <hr>
# _Data Scientist:_ _Kyle Killion, Tracie Scott, Vinh Le_<br>
# <br>
# The project data is from the Open University Learning Analytics Dataset (OULAD) made available on the
#  University of California, Irvine, (UCI), Machine Learning site
#  https://archive.ics.uci.edu/ml/datasets.html. OULAD contains data for seven selected courses,
# the students registered for the courses, and the students’ interactions with the Virtual Learning
# Environment (VLE) for those courses.
# 
# 
# The Open University currently collects similar data on a on-going basis as input to algorithms they developed to identify students at risk for failing a course. Identification of at-risk students then triggers automated intervention measures to encourage behavior that would create success. For example, the algorithm might identify a student with low grades on intermediate assessments (quizes). That student may be sent an automated email reminder about available tutoring options. The goal of the data collection effort is to maximize student success, which has numberous benefits for the University.
# 
# This subsest of anonymized data was made available to the public for educational purposes on Machine Learning approaches.    
# 
# For the purpose of this project, the data will be used to determine if socio-economic and/or behavior-based data can be used to predict a student's performance in a course. Performance is determined by the final result of the student’s effort and is characterized by completing the course with a passing score, either with or without Distinction. The specific Questions of Interest are: 
# 
# Using the data available,
# * Can we predict a student's final status in a course based on socio-economic factorrs and/or patterns of interaction with the VLE? A good prediction algorithm would allow prediction of student success 90% of the time or 75% of the time after the first 30 days of the course.
# * Can we identify clusters of participants that are more likely to succeed within the first year of education using the Open University based on socio-econominc factors?  
# * Further, can we identify study habits or patterns of interaction with the VLE that enhance the probability of success? 
#         

# # Data Understanding
# <hr>
# This project leverages the anonymized Open University Learning Analytics Dataset (OULAD). The data set
# contains data about courses and students as well as the students’ interactions with the Virtual Learning
# Environment (VLE) for seven selected courses (called modules). The data represents a small slice of time
# for a very  limited number of courses (7).
# <br>
# While there are seven data sets comprising OULAD, only two data sets were referenced in the analysis. 
# 
# References:
# More information, examples and news can be found at: [Web Link]. This dataset is released
#  under CC-BY 4.0 license. Citation request is Kuzilek, J., Hlosta, M., Herrmannova, D.,
# Zdrahal, Z. and Wolff, A. OU Analyse: Analysing At-Risk Students at The Open University.
# Learning Analytics Review, no. LAK15-1, March 2015, ISSN: 2057-7494.
# 
# Data was used in a prior exercise by Tracie Scott in Experimental Statistics II. Data Description are largely taken from the OULAD site and that earlier research paper. 
# 
# ### studentInfo data attributes
# <hr>
# The “studentInfo” dataset contains demographic and personal information for each student
# 
# Feature           |  Type    | Description
# ------------------|----------|------------------------------------------------------------------
# code_module       | string   | Identification code for a module for which the student is registered 
#                   |          |  (at SMU, this would be a “section”, like Experimental Statistics 402). 
#                   |          |  Used for data identification (key value) only. This code has been anonymized and there is no 
#                   |          |  information concerning the subject matter or difficulty of the module.
# code_presentation | string   | Identification code of the presentation during which the student is registered for the module 
#                   |          |  (at SMU, this would be a school term, like “Spring 2017”). Used for identification (key value) 
#                   |          |  only. It consists of the year and “B” for the presentation starting in February and “J” for the 
#                   |          |  presentation starting in October.
# id_student        | string   | Unique identification number for the student. Used for identification (key value) only. 
#                   |          |  Note: The above three fields were concatenated into a single field to create a unique identifier
#                   |          |  joining the student to the student’s study activities.
# gender            | string   | The student’s gender
#                   |          |  INITIAL VALUE                           CODED VALUES
#                   |          |  Male                                    0
#                   |          |  Female                                  1
# region            | string   | Identifies the geographic region, where the student lived while taking the module-presentation                     |          | May not be included in any analysis that is unable to use categorical data
# highest_education | string   | Highest student education level on entry to the module presentation
#                   |          |   INITIAL VALUE                            CODED VALUES
#                   |          |   No Formal quals (qualification)         0
#                   |          |   Lower Than A Level                      1
#                   |          |   A Level or Equivalent                   2
#                   |          |   HE Qualification                        3
#                   |          |   Post Graduate Qualification             4
# imd_band          | string   | Secifies the Index of Multiple Depravation band of the place where the student lived during 
#                   |          |    the module-presentation. The higher the number, the more challenged the location for resources
#                   |          |  INITIAL VALUE                            CODED VALUES
#                   |          |  0-10%                                    10
#                   |          |  10-20 (minor error in raw data)          20
#                   |          |  20-30%                                   30 
#                   |          |  30-40%                                   40
#                   |          |  40-50%                                   50
#                   |          |  50-60%                                   60
#                   |          |  60-70%                                   70
#                   |          |  70-80%                                   80 
#                   |          |  80-90%                                   90
#                   |          |  90-100%                                 100
# age_band          | String   | Band of the student’s age. 
#                   |          | The oldest band was not included in the analysis as there were so few in that band as to be 
#                   |          |    considered outliers
#                   |          | INITIAL VALUE                            CODED VALUES
#                   |          |  0-35                                    0
#                   |          |  35-55                                   1
#                   |          |  55<=                                    2 - removed for analysis
# num_of_pre_attempts| Integer |  Number of times the student has attempted this module at time of registration. 
#                   |          | Rows with values > 0 were excluded from the analysis.
# studied_credits   | Integer  | Total number of credits for the modules the student is currently studying. 
#                   |          | This field contained unexpected large values and was excluded from the final analysis. 
#                   |          | More information about this field is needed if further research is performed on this data.
# disability        | String   | Indicates whether the student has declared a disability
#                   |          | INITIAL VALUE                           CODED VALUES
#                   |          | No disability declared                  0           
#                   |          | Disability declared                     1
# final_result      | String   | Student’s final result in the module-presentation
#                   |          | INITIAL VALUE                           CODED VALUES
#                   |          | Withdraw                                0 
#                   |          | Fail                                    1 
#                   |          | Pass                                    2
#                   |          | Distinction                             3
#     
# 

# # Data Understanding (continued)
# <hr>
# ### studentVle data attributes
# <hr>
# The studentVle.csv file contains information about each student’s interactions with the materials in the 
# Virtual Learning Environment (VLE). This file contains the following columns:  
# <br>
# 
#     Feature       |  Type    | Description
# ------------------|----------|-----------------------------------------------------------------
# code_module       | String   | An identification code for a module.
# code_presentation | String   | Identification code of the module presentation.
# id_student        | String   | Unique identification number for the student. 
# id_site           | String   | Identification number for the VLE material. 
#                   |          | There are many different types of materials, 
#                   |          |    such as oucontent, ouwiki, url, forumng, etc…, and 
#                   |          |    zero to many instances of each type for each module.
# date              | Integer  | The day of the student’s interaction with the material; measured as the number of days since 
#                   |          |   the start of the module-presentation. Represented as an integer value and 
#                   |          |   can be negative if the material is accessed before the start of the presentation.
# sum_click         | Integer  | Number of times a student interacts with the material in that day.
# 
# ### Calculated Fields
# <hr>
# The studentVLE set was collapsed to provide summary statistics for student activity that could be assocaited 
# with the ‘studentInfo’ data. The resulting data set was used to look for study habits or patterns of usage that 
# lead to a greater propensity for program success.
# 
# <br>
# 
# Calculated Field  | Type    |   Description
# ------------------------------------------------------------------------------------
# CntActivities     | Integer |  Count of unique study events (resource / day)
# SumAllClicks      | Integer |  Total number of clicks for all activities in the presentation
# SumClicks30       | Integer |  Clicks within the first 30 days of the presentations
# EarlyAccess       | Integer |  Earliest day of study activity for the presentation (integer value, 
#                   |         |   may be negative if precedes first day of presentation)
# 
# 

# # Verify Data Quality
# <hr>
# A small percentage of rows are missing one or more values in the socio-economic fields. The information was probably omitted by the student during registration. In the raw data, these are notated with a ? and were converted to NAN values to be recognized as such by Python.  
# Rows missing any values used in the analysis will be dropped from that analysis.
# One of the values for imb_bad is not formatted consistently with the ohters, the '%' was missing from the 10-20% segment. This was addressed with data transformation..
# 
# Otherwise, the data were remarkably clean. There were no invalid entries in fields containing discreet values. As indicated above, character values were replaced with ordinal values to assist with analysis. 
# 
# ### Outliers
# In anticipation of cluster analysis, which is senstive to outliers, special care was taken to eliminate unusally high values.  Specifically,
# 
# When reviewing the data, the highest value for age_band (over 55) was used so infrequently as to appear to be an outlier in the diagnostics. These rows were removed from the data set.
# 
# The calculated values occasionally recorded unusual values as well. The following thesholds were established to prevent outliers in thsoe fields. Any data that exceeded these thresholds were eliminated from the analysis.
# FIELD               DESCRIPTION                                     THRESHOLD VALUE
# Cntactivity         Count of unique study evends (resource/day)     over 300
# EarlyAccess         First day course materials were referenced      over 200
# SumAllClicks        Sum of all clicks                               over 20,000
# SumClicks30         Clicks within the first 30 days                 over 4,000
# 
# ### Avoiding Time Series Issues
# <hr>
# As students can register for multiple classes, there can be several rows for one student related to different modules or different presentations of the same modules.
# 
# * To minimize the time-series nature of the data, all repeat attempts by a student for a module were removed. 
# * Only data associated with students whose highest level of education as below “HE Qualification” (aka Freshmen) at the time of module registration were retained in the analysis. 
# * No attempt was made to eliminate student repetition across in multiple modules as a visual review of the data indicated few instances of participation by one student in multiple modules in the remaining data.
# 
# 

# In[7]:

#Hosekeeping chores; Import modules, set working directory, print session info
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import os
import sys

#Set the Current Working Directory of your choice of files
os.chdir(r'C:\Users\Yao\Dropbox\immersion future contacts\OULAD')

print(os.getcwd())


#Print out our session info versions of dependencies
print('Pandas :',pd.__version__)
print('numpy :',np.__version__)
print('seaborn :',sns.__version__)
print('Matplotlib :', matplotlib.__version__)


# In[8]:

#Read .csv files into pandas dataframes

studentInfo = pd.read_csv('studentInfo.csv')
vle = pd.read_csv('studentVle.csv')


# In[9]:

# Sample data rows for studentInfo
studentInfo.head()


# In[10]:

#studentInfo.describe()


# In[11]:

#Data formates for studentInfo
#studentInfo.info()


# In[12]:

#Sample data for vle (Virtual Learning Environment)
vle.head()


# In[13]:

#vle.describe()


# In[14]:

#################################################################################
# Begin Data Alignments, Alterations, and Custom Calc additions for consumption #
#################################################################################

# Start with the Student Info 
dfStudents = studentInfo

# All question marks to Nan to denote missing values 
dfStudents = dfStudents.replace('?', np.nan)

# Replace M & F for binomial data: M=0 F=1
dfStudents.gender.replace({'M': 0, 'F': 1}, inplace=True)

# Ordinize imd_band vector to numeric value
corrList = ["0-10%",
            "10-20",
            "20-30%",
            "30-40%",
            "40-50%",
            "50-60%",
            "60-70%",
            "70-80%",
            "80-90%",
            "90-100%"]
count = 10
for i in corrList:
    dfStudents.imd_band = dfStudents.imd_band.replace(i, count)
    count += 10

# Ordinize highest_education vector
corrList2 = ['No Formal quals (qualification)',
             'Lower Than A Level',
             'A Level or Equivalent',
             'HE Qualification',
             'Post Graduate Qualification']

  
for i in range(0,len(corrList2),1):    
    dfStudents['highest_education'] = dfStudents['highest_education'].replace(corrList2[i], i)
    
    
dfStudents['highest_education'] = dfStudents['highest_education'].convert_objects(convert_numeric=True)

# Ordinize the age_band vector
corrList3 = ['0-35', '35-55', '55<=']
for i in range(0,len(corrList3), 1):
    dfStudents.age_band = dfStudents.age_band.replace(corrList3[i], i)

# Ordinize the Disability into binomial data
dfStudents.disability.replace({'N': 0, 'Y': 1}, inplace=True)
    

# Ordinize the Final_Result vector
dfStudents.final_result = dfStudents.final_result.replace({'Withdrawn':0,'Fail':1,'Pass':2,'Distinction':3})

# Student Info set ready
dfStudents.head(200)


# In[15]:

#dfvle_collapsed 
# First, we need to collapse the dataset so that we have a sum_click sum for every unique occurances of 
# module, presentation id_student, ID_site (resource), and date 
# once we have that, we can do the other calculations based on that collapsed dataset

dffull_vle = vle
dfvle_collapsed = dffull_vle.groupby(['code_module','code_presentation','id_student','id_site','date'], as_index=False).sum()
#dfvle_collapsed.head()


# In[16]:

#CNTactviity - count of study sessions per student (resource + date)
#use the groupby option to calculate various values based on vle activity
#and then append those calculated values back to the student file we can use for analytics

#lets count all the study activities as defined by each student having a uniue site (aka resource) and date value

dftemp_vle = dfvle_collapsed.groupby(['code_module','code_presentation','id_student'], as_index=False).count()
dftemp_vle.rename(columns={"id_site" : "CNTactivity"}, inplace=True)
del dftemp_vle['date']
del dftemp_vle['sum_click']
#dftemp_vle.head(20)




# In[17]:

# Merge CNTActivities onto a new dataframe dfStudentVle using code_module code_presentation id_student as the keys

# two datasets you are trying to merge
dfStudents = dfStudents.reset_index()
dftemp_vle = dftemp_vle.reset_index()
dfStudentsVle = pd.merge(dftemp_vle, dfStudents, how='outer', on=["code_module", "code_presentation", "id_student"])
#dfStudentsVle.head(20)


# In[18]:

#EarlyAccess
#use the groupby option to calculate the minimum value for date for each module/presentation/student
# This will indicate the earliest date of access to materials
# and then append that calculated value back to one "flat" file we can use for analytics

dftemp_vle = dfvle_collapsed.groupby(['code_module','code_presentation','id_student'], as_index=False).min()
dftemp_vle.rename(columns={"date" : "EarlyAccess"}, inplace=True)
del dftemp_vle['id_site']
del dftemp_vle['sum_click']
#dftemp_vle.head(20)


# In[19]:

# two datasets you are trying to merge
dfStudentsVle = dfStudentsVle.reset_index()
dftemp_vle = dftemp_vle.reset_index()
dfStudentsVle = pd.merge(dftemp_vle, dfStudentsVle, how='outer', on=["code_module", "code_presentation", "id_student"])
#dfStudentsVle.head(20)


# In[20]:

#SumAllClicks
# use the groupby option to calculate the minimum value for date for each module/presentation/student
# This will indicate the earliest date of access to materials
# and then append that calculated value back to one "flat" file we can use for analytics

dftemp_vle = dfvle_collapsed.groupby(['code_module','code_presentation','id_student'], as_index=False).sum()
dftemp_vle.rename(columns={"sum_click" : "SumAllClicks"}, inplace=True)
del dftemp_vle['id_site']
del dftemp_vle['date']
#dftemp_vle.head(20)


# In[21]:

# two datasets you are trying to merge
dfStudentsVle = dfStudentsVle.reset_index()
dftemp_vle = dftemp_vle.reset_index()
dfStudentsVle = pd.merge(dftemp_vle, dfStudentsVle, how='outer', on=["code_module", "code_presentation", "id_student"])
#dfStudentsVle.head(20)


# In[22]:

#AllClicks30
# use the groupby optoin to calculate the clicks within the first 30 days for each module/presentation/student
# This will indicate early effort in the program
# and then append that calculated value back to one "flat" file we can use for analytics

dftemp_vle = dfvle_collapsed[dfvle_collapsed.date <= 30].groupby(['code_module','code_presentation','id_student'], as_index=False).sum()
dftemp_vle.rename(columns={"sum_click" : "SumClicks30"}, inplace=True)
del dftemp_vle['id_site']
del dftemp_vle['date']
#dftemp_vle.head(20)


# In[23]:

# two datasets you are trying to merge
dfStudentsVle = dfStudentsVle.reset_index()
dftemp_vle = dftemp_vle.reset_index()
dfStudentsVle = pd.merge(dftemp_vle, dfStudentsVle, how='outer', on=["code_module", "code_presentation", "id_student"])
#dfStudentsVle.head(20)


# In[24]:

#dfStudentsVle exploration
# if you rerun this block, comment out the next two lines of code or you will error out.
del dfStudentsVle['index_x']
del dfStudentsVle['index_y']

dfStudentsVle.info()


# In[25]:

### Implement default values for calculated values  

#if SumClicks30 NAN; 0
#if SumAllClicks NAN; 0
#if CNTactivity NAN; 0
#if EarlyAccess NAN;200  #since zero is a valid value and is actually a positive outcome, set this high but don't drop

dfStudentsVle['SumClicks30'].fillna(value=0, inplace=True)
dfStudentsVle['SumAllClicks'].fillna(value=0, inplace=True)
dfStudentsVle['CNTactivity'].fillna(value=0, inplace=True)


# # Statistical Visualization
# <hr>
# Note: 
# * Values for discreet values are all within range.
# * Calculated values seem to have outliers. Max values are far higher than 75%. Implement maximim thresholds.
# * Values for 'studied credits' do not seem to align to data description. Further investigation is required.
# 

# In[26]:

dfStudentsVle.describe()


# In[27]:

#Elimniate rows where the calculated values were unusually high to eliminate outliers

dfStudentsVle2 = dfStudentsVle[dfStudentsVle['SumClicks30'] < 4000]
dfStudentsVle2 = dfStudentsVle2[dfStudentsVle2['SumAllClicks'] <20000]
dfStudentsVle2 = dfStudentsVle2[dfStudentsVle2['CNTactivity'] <2000]
#dfStudentsVle2 = dfStudentsVle2[dfStudentsVle2['EarlyAccess'] < 201]
#dfStudentsVle2 = dfStudentsVle[dfStudentsVle['age_band'] < 2]              #eliminate >55 as it creates an outlier situation

#Minimize time series issues

dfStudentsVle2 = dfStudentsVle2[dfStudentsVle2['num_of_prev_attempts'] < 1]  #eliminate retakes  NOT WORKING!!! see charts below
#dfStudentsVle2 = dfStudentsVle[dfStudentsVle['highest_education'] <2]     #eliminate students who have more than one year of higher education

#these folks 'never' accessed resources, but I don't want to drop the row; set value higher than highest valid value
dfStudentsVle2['EarlyAccess'].fillna(value=240, inplace=True)  

#Eliminate rows with remaining missing values
dfStudentsVle2 = dfStudentsVle2.dropna()              # drop any rows with NA. Missing demographic data from student registration

#repeat discribe to see final data
dfStudentsVle2.describe()


# In[28]:

#Eliminate features that are no longer useful going forward
# 1. Remove attributes that just arent useful for us
del dfStudentsVle2['num_of_prev_attempts']
del dfStudentsVle2['studied_credits']

#repeat discribe to see final data
dfStudentsVle2.describe()


# # Visualize attributes graphically (at least 5)
# <hr>
# While there is some variation in the data, at this point, there is no obvious data challenge.
# * Socio-economic values have reasonable variation across the seven modules.
# * Performance variables also have reasonable variation across modules and gender. 
# * Performace attributes still exhibit outliers, * Which may require futher attention as analsys progresses.
# * The patterns of Early Acess to course materials looks a little bimodel. Patterns are very similar across genders
# Overall the data seem to be ready for further analysis.

# In[29]:

ax = sns.violinplot(y="SumAllClicks", x="code_module", hue="gender",
                    data=dfStudentsVle2, palette="muted", split=True) 
plt.show()


# # Visualize relationships amoung explanatory variables
# * There does not seem to be covariance amoung the variables (note that Num_of_prev_attempts will only be 0 in the analysis)
# * These data have numerous featues with discreet values.
# * Note the low occurance of age > 55 (will not be included in analysis)
# * There is likely high covariance with the calculated performance variables and we will see that on a later chart
# 

# In[32]:

sns.pairplot(dfStudentsVle2, vars=["gender","age_band", "disability", "final_result"])
plt.show()


# # Visualize relationship with response variable : final_result
# * Variation in Final_Result is relatively consistent across modules. No one module has an unusal pattern of grades.
# * The pariwise plot of performance variables shows some expected covariance. Note that the clusters formed by some of these pairs are elongated, which may have imlications for analysis. There is no obvious seperation of clusters by final_result evidenced on the chart.
# * The Correlation heat map also indicates expected correlations among the performance variables. Note that SumAllClicks and CNTactivity have the highest correlation to final_result, indicating that effort plays a larger role in success of students in these models that the socio-economic features represented.
# 

# In[33]:

### Visualize relationship with response variable final_result in pairplots of performance variables
sns.set(style="ticks", color_codes=True) # change style
g = sns.pairplot(dfStudentsVle2,vars=["SumClicks30","SumAllClicks", "EarlyAccess","CNTactivity"], hue="final_result")
plt.show()


# In[34]:

# Correlation Heatmap of all the Variables
sns.heatmap(dfStudentsVle2.corr(), vmax=.8, square=True)
plt.show()


# ### Summary 
# In summary, the data are prepared for analysis with a only a few areas of concern highlighted. The correlation heat map indicates the performance variables, especially SumAllClicks and CNTactivity, have the highest correlation to final_result, indicating that effort plays a larger role in success of students in this model rather than the socio-economic features represented. However, further analysis is required to understand the extent to which these various features affect course success.

# In[35]:

from sklearn.preprocessing import StandardScaler
# normalize the data - THIS IS NOT WORKING BUT I THINK I KNOW WHY. 
#I NEEDED TO PULL OUT THE NUMERIC VALUES 
# and probably create dummies for final_result. 
#LET's WORK ON THIS

#dfNormal = dfStudentsVle2
#y = dfNormal['final_result']
#dfNormal = (dfNormal-dfNormal.mean())/dfNormal.std()
#dfNormal['final_result'] = y

#dfNormal.describe()

#X = StandardScaler().fit_transform(dfStudentsVle2)
#X.describe()



# In[45]:

from sklearn.cluster import KMeans
# Extract only numeric Data
data = dfStudentsVle2._get_numeric_data().fillna(0)



# run kmeans algorithm (this is the most traditional use of k-means)
kmeans = KMeans(init='random', # initialization
        n_clusters=10,  # number of clusters
        n_init=1,       # number of different times to run k-means
        n_jobs=-1)

kmeans.fit(data)
print(kmeans.inertia_)



# In[46]:

# this 'works' but  the class i want to predict is still in the data set.
# I feel like we need to seperate the data from the target variable (final_result)
distortions = []
for i in range (1,11):
    kmeans = KMeans(init='k-means++', # initialization
        n_clusters=i,  # number of clusters
        n_init=10,       # number of different times to run k-means
        max_iter = 20,
        random_state=0)
    kmeans.fit(data)
    distortions.append(kmeans.inertia_)

plt.plot(range(1,11), distortions, marker = 'o')
plt.xlabel('number of clusters')
plt.ylabel('Distortion')
plt.show()
    


# In[56]:

#Here, I am running same with recommended # of  clusters

kmeans = KMeans(init='k-means++', # initialization
    n_clusters=4,  # number of clusters
    n_init=8,       # number of different times to run k-means
    max_iter = 20,
    random_state=0)

kmeans.fit(data)
y_km = kmeans.fit_predict(data)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_

print(centroids)
print(labels)

data.info()




# In[ ]:

#this may be helpful...but it's not quite 'right'
centers = kmeans.cluster_centers_
centers[centers<0] = 0 #the minimization function may find very small negative numbers, we threshold them to 0
centers = centers.round(2)
print('\n--------Centers of the four different clusters--------')
print('Data\t Cent1\t Cent2\t Cent3\t Cent4')
for i in range(11):
    print(i+1,'\t',centers[0,i],'\t',centers[1,i],'\t',centers[2,i],'\t',centers[3,i])


# In[ ]:



