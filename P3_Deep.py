################################################################################
#                             Python Library                                   #
################################################################################
import pandas as pd
import numpy as np

import xlsxwriter

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns


from keras import optimizers
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense


################################################################################
#                        Extract Dataset                                       #
################################################################################
def Extract_Dataset(drop_column, tar_column, filename, ori_filename):
    #read in data using pandas
    train_df = pd.read_csv('/home/park/Desktop/NumberP/P3/ANN/Dataset/tot_data.csv')

    #create a dataframe with all training data except the target column
    train_X_pure = train_df.drop(columns=drop_column)
    train_X_df=pd.DataFrame(train_X_pure)

    #create a dataframe with only the target column
    train_y_pure = train_df[tar_column]
    train_y_df=pd.DataFrame(train_y_pure)

    #extract lpa data
    lpa_pure = train_df[['cell_lpa']]
    lpa_df = pd.DataFrame(lpa_pure)

    #scale dataset in order to protect training problem
    #scaled X_train dataset
    train_X = Scale_Data(train_X_df)
    #scaled y_train dataset
    train_y = Scale_Data(train_y_df)

    #training dataset and then predict target data
    #and get Z axis(target data)
    Zaxis = Build_Network(train_X,train_y,filename)

    #convert numpy array
    Xaxis = train_df[['X']] #X axis
    Yaxis = train_df[['Y']] #Y axis
    num_Xaxis = convert_numpy(Xaxis)
    num_Yaxis = convert_numpy(Yaxis)
    num_Zaxis_original = convert_numpy(train_y)
    lpa_ori = convert_numpy(lpa_df)

    #write predicted dataset into excel file
    Write_Excel(num_Zaxis_original,Zaxis,lpa_ori,filename)

    #plotting original data
    Plotting_fun(num_Xaxis,num_Yaxis,num_Zaxis_original,ori_filename)
    #plotting result
    Plotting_fun(num_Xaxis,num_Yaxis,Zaxis,filename)

################################################################################
#                          Scale Dataset                                       #
################################################################################
def Scale_Data(pure_data):
    Data_A_step = pure_data.values.astype(float)
    min_max_scaler = preprocessing.MinMaxScaler()
    Scaled_data = min_max_scaler.fit_transform(Data_A_step)
    Data_zip = pd.DataFrame(Scaled_data)
    return Data_zip


################################################################################
#                         Build Network                                        #
################################################################################
def Build_Network(train_X, train_y, filename):
    #get number of columns in training data
    n_cols = train_X.shape[1]

    #create model
    #Architecture Nueral Network
    model = Sequential()
    model.add(Dense(50, activation='relu', input_shape=(n_cols,)))     #first layer
    model.add(Dense(25, activation='relu'))                            #second layer
    model.add(Dense(1, activation='sigmoid'))                          #output layer

    #compile model
    op_type = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=op_type, loss='mean_squared_error')

    #train model
    train_history = model.fit(train_X, train_y, validation_split=0.3, epochs=150)

    #check training loss and validation loss graph
    loss = train_history.history['loss']
    val_loss = train_history.history['val_loss']
    plt.plot(loss)
    plt.plot(val_loss)
    plt.legend(['loss','val_loss'])
    plt.savefig(filename+'_lossgraph.png', bbox_inches='tight', pad_inches=0)

    #predict target data
    test_y_predictions = model.predict(train_X)

    return test_y_predictions


################################################################################
#                      Save Excel predicted data                               #
################################################################################
def Write_Excel(y_original,y_predictions,lpa,filename):
    workbook = xlsxwriter.Workbook(filename+'.xlsx')
    worksheet = workbook.add_worksheet()

    #size of predicted dataset
    Num_row = len(y_predictions)

    #increased zgrade point
    XO_build = Point_build(y_original,y_predictions,lpa)

    worksheet.write(0,0,'Original zgrade')
    worksheet.write(0,1,'Predicted zgrade')
    worksheet.write(0,2,'new_point')

    for i in range(0,Num_row):
        worksheet.write(i+1,0,y_original[i])
        worksheet.write(i+1,1,y_predictions[i])
        worksheet.write(i+1,2,XO_build[i])


    workbook.close()

################################################################################
#                                build or not                                  #
################################################################################
def Point_build(y_original,y_predictions,lpa):

    size_of_array = len(lpa)
    #set numpy array size(return value)
    New_Inst = np.empty((0,1),int)

    #Determination of facility installation
    for j in range(0,size_of_array):
        #for 3 or 4 lpa grade
        if (lpa[j]==3)or(lpa[j]==4):
            if (y_original[j] < y_predictions[j]):
                Installation = 1 #1 means that cell predicts installation
                New_Inst = np.append(New_Inst,np.array([[Installation]]),axis=0)
            else :
                Installation = 0
                New_Inst = np.append(New_Inst,np.array([[Installation]]),axis=0)

        #for 1 or 2 lpa grade
        else :
            if (y_original[j] < y_predictions[j]):
                #the lpa grade 1 or 2, but the zgrade is increased
                Installation = 2 #2 means that zgrade is increased
                New_Inst = np.append(New_Inst,np.array([[Installation]]),axis=0)

            else :
                #the lpa grade 1 or 2, but the zgrade is decreased
                Installation = 3 #3 means that zgrade is decreased
                New_Inst = np.append(New_Inst,np.array([[Installation]]),axis=0)

    return New_Inst


################################################################################
#                        plotting predicted data                               #
################################################################################
# code reference :  https://python-graph-gallery.com/371-surface-plot/         #
################################################################################
def Plotting_fun(X,Y,Z,filename):
    #X,Y,Z dataset shape is (904,1)
    #squeeze() function help it to remove (,1)
    #Input data shape gonna be (904,)
    sam_X = X.squeeze()
    sam_Y = Y.squeeze()
    sam_Z = Z.squeeze()

    #Make the plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(sam_X, sam_Y, sam_Z, cmap=plt.cm.viridis, linewidth=0.2)

    #save plot image
    plt.savefig(filename+'_result.png', bbox_inches='tight', pad_inches=0.25)

################################################################################
#                        convert from string to int                            #
################################################################################
def convert_numpy(pandasData):
    NumpyArray=np.asarray(pandasData)
    return NumpyArray

################################################################################
#                                    main                                      #
################################################################################
#drop column
drop_dc = ['65_sen','65_2014','65_2015','65_2016','65_2017','65_2018','65_500m',
            'kin_approch','ocen_approch','old_approch','65_rate',
            'kin_app','ocen_app','old_app','bgrade_kin','bgrade_ocen',
            'z2grade_ocen', 'z2grade_kin', 'z2grade_dc', 'z2grade_old','_merge']

drop_kin = ['65_sen','65_2014','65_2015','65_2016','65_2017','65_2018','65_500m',
            'dc_approch','ocen_approch','old_approch','65_rate',
            'dc_app','ocen_app','old_app','bgrade_dc','bgrade_ocen',
            'z2grade_ocen', 'z2grade_kin', 'z2grade_dc', 'z2grade_old','_merge']

drop_ocen = ['07_sen','07_2014','07_2015','07_2016','07_2017','07_2018','07_500m',
            'dc_approch','kin_approch','old_approch','07_rate',
            'dc_app','kin_app','old_app','bgrade_kin','bgrade_dc',
            'z2grade_ocen', 'z2grade_kin', 'z2grade_dc', 'z2grade_old','_merge']

drop_old = ['07_sen','07_2014','07_2015','07_2016','07_2017','07_2018','07_500m',
            'dc_approch','kin_approch','ocen_approch','07_rate',
            'dc_app','kin_app','ocen_app','bgrade_kin','bgrade_dc',
            'z2grade_ocen', 'z2grade_kin', 'z2grade_dc', 'z2grade_old','_merge']

#target column
tar_dc = ['z2grade_dc']
tar_kin = ['z2grade_kin']
tar_ocen = ['z2grade_ocen']
tar_old = ['z2grade_old']

#filename declare
dc_file = 'dc_predict'
kin_file = 'kin_predict'
ocen_file = 'ocen_predict'
old_file = 'old_predict'

dc_file_or = 'dc_original'
kin_file_or = 'kin_original'
ocen_file_or = 'ocen_original'
old_file_or = 'old_original'

"""
Readme

Run the Code by deleting the comment "#" of the desired prediction result.

Training can only see one prediction at a time

For example, : if you want to predict for kin

#predict for kin
Extract_Dataset(drop_kin, tar_kin, kin_file, kin_file_or)
"""
#predict for dc
Extract_Dataset(drop_dc, tar_dc, dc_file,dc_file_or)

#predict for kin
#Extract_Dataset(drop_kin, tar_kin, kin_file, kin_file_or)

#predict for ocen
#Extract_Dataset(drop_ocen, tar_ocen, ocen_file,ocen_file_or)

#predict for old
#Extract_Dataset(drop_old, tar_old, old_file,old_file_or)
