# Implementing the Least Squares Classifier for MNIST dataset
### Dan Vu

In this project we will first implement a **binary least squares classifier** which will then be used as the building block for a **one vs rest multi-class classifier** and a **one vs one multi-class classifier** 

## Preparing our python environment


```python
import numpy as np

import matplotlib.pyplot as plt

from scipy.io import loadmat

import pandas as pd

#allows pd.DataFrames to be printed without truncation
pd.options.display.max_columns = None 
pd.options.display.max_rows = None
```

## Defining some functions that will help us down the line

`show_img_lbl` displays the image (handwritten digit) and label 

`show_DF` will display the dataframe of a given array/matrix


```python
def show_img_lbl(X, Y, nth):
    
    plt.figure()
    plt.imshow(X[nth,:].reshape(28,28),cmap='binary')
    print(Y[nth])
```


```python
def show_DF(X):
    return pd.DataFrame(X).head(1000)
```

## These functions will be used to build the binary classifier


```python
def preprocess(X,testX):
    
    newX = X
    newtestX = testX
    
    pixel_counter = np.zeros((1,784))
    
    for image_ind in range(len(X)):
        for pixel_ind in range(len(X[image_ind])):
            if X[image_ind,pixel_ind] != 0:
                pixel_counter[0,pixel_ind]=pixel_counter[0,pixel_ind]+1
                
    
    deletelist=[]
    for pixel_ind in range(len(X[0])):
        if pixel_counter[0,pixel_ind] < 600:
            deletelist.append(pixel_ind)
            
            
    newX = np.delete(newX, deletelist, axis=1)
    newtestX = np.delete(newtestX, deletelist, axis=1)
    
            
        
    return newX,newtestX
```


```python
def get_binary_Y(Y, num):
    binary_Y=np.zeros(np.shape(Y), dtype=int)
    for i in range(np.size(Y)):
        if Y[i,0] == num:
            binary_Y[i,0]=1
        else:
            binary_Y[i,0]=-1
        
    return binary_Y
```


```python
def getA(X):
    A=X
    A=A/255
    A = np.insert(A,0,1,axis=1)
    
    return A
```


```python
def getBeta(A,Y):
    try:
        ATA=(A.T@A)
    #     print(ATA.shape) # 494,494
        ATA_inv=np.linalg.inv(ATA)
    #     print(ATA_inv.shape) # 494,494
        ATA_inv_AT=ATA_inv @ A.T
    #     print(ATA_inv_AT.shape) # 494,60000
        beta = ATA_inv_AT @ Y # 494,60000 @ 1,60000
    except:
        AT = A.T
        AAT = A @ A.T
        AAT_inv = np.linalg.inv(AAT)
        AT_AAT_inv = AT @ AAT_inv
        beta = AT_AAT_inv @ Y

    return beta
```


```python
def testBeta(beta,ptestX):
    try:
        yhat=np.sign(beta.T @ ptestX.T)
    except:
        print(beta.shape,ptestX.shape)
        yhat=np.sign(beta @ ptestX.T)
    return yhat
```


```python
def check_binaryClassifier(yhat, btestY):
    
    yhat=yhat.T
    btestY=btestY.T
    
    nfp,nfn,ntp,ntn=0,0,0,0
    for index in range(len(yhat)):
        if yhat[index] == btestY.T[index]:
            if yhat[index]==1:
                ntp=ntp+1
            else: 
                ntn=ntn+1
        elif yhat[index]!=btestY.T[index]:
            if yhat[index]==1:
                nfp=nfp+1
            else:
                nfn=nfn+1
                
    return nfp,nfn,ntp,ntn,
    
```


```python
def generate_confusion(nfp,nfn,ntp,ntn):
    error_rate = (nfn+nfp)/(nfn+nfp+ntp+ntn)
    
    confusion = np.array([[ntp, nfn, ntp+nfn],[nfp, ntn, nfp+ntn]])
    return confusion, error_rate
```

## Binary Classifier


```python
def binaryClassifier(ptrainX,btrainY):
    
    A=getA(ptrainX)
    beta=getBeta(A,btrainY)
    
    return beta
```


```python
def test_binaryClassifier(ptestX, beta):
    
    testA=getA(ptestX)
    
    yhat=testBeta(beta,testA)
    
    return yhat
```


```python
def test_binaryClassifier2(ptestX, beta): #without the signed function
    
    testA=getA(ptestX)
    
    ytilda=beta.T @ testA.T
    
    return ytilda
```

## Now that Binary Classifier is complete, we can build the multi-class classifiers

First, we will load in and prepare our data:

`prep()` will return trainY and testY in binary format (1,-1) with the appropriate given digit


 `preprocess()` will filter out all pixels EXCEPT for the ones that have a non zero value more than 600x


```python
def prep(trainY,testY, num):
    btrainY=get_binary_Y(trainY, num)
    btestY=get_binary_Y(testY, num)
    
    return btrainY,btestY
```


```python
data = loadmat("mnist.mat")
trainX = data['trainX']
trainY = data['trainY'].transpose()
testX = data['testX']
testY = data['testY'].transpose()

ptrainX,ptestX=preprocess(trainX,testX)
```

## One vs All (One vs Rest)


```python
def onevsall(ptrainX,trainY,ptestX,testY):
    
    BET=np.array([[]])
    
    for num in range(10):
        btrainY,btestY = prep(trainY,testY,num)
        beta=binaryClassifier(ptrainX,btrainY)
        BET=np.append(BET,beta)
        
    BET=BET.reshape(10,beta.shape[0])
        
    return BET
```


```python
def test_onevsall(trainY,ptestX,testY,BET):
    
    YTILDA=np.array([])
    
    for num in range(10):
        btrainY,btestY = prep(trainY,testY,num)
        ytilda=test_binaryClassifier2(ptestX,BET[num,:])
        YTILDA=np.append(YTILDA,ytilda)
        
    YTILDA=YTILDA.reshape(10,ptestX.shape[0])
        
    return YTILDA
        
```

We call the `onevsall()` which will return BET, an array of beta matrices or weights 


```python
BET = onevsall(ptrainX,trainY,ptestX,testY)
```

then, we apply the testX dataset to our trained BET model by calling `test_onevsall()` to see how it performs


```python
YTILDA=test_onevsall(trainY,ptestX,testY,BET)
```

we then find the argmax of YTILDA (ftilda(x)) to see which digit our model predicted from the testX dataset 


```python
def getArgMax(YTilda):
    
    YHAT=np.argmax(YTilda, axis=0)
    
    YHAT=YHAT.reshape(YTilda.shape[1],1)
    
    return YHAT
```


```python
YHAT=getArgMax(YTILDA)
```

this next function checks the performance of our model


```python
def check_onevsall(YHAT,testY):
    
    confusion=np.zeros((10,10))
    
    for index in range(YHAT.shape[0]):
        confusion[testY[index],YHAT[index]]+=1
    
    error=1-np.trace(confusion)/(YHAT.shape[0])
    
    return error,confusion
```


```python
erro,conff=check_onevsall(YHAT,testY)
```

## Confusion Matrix and Error Rate(Test Set)

Confusion Matrix:
- horizontal axis = predictions
- vertical axis = reference digit


```python
print('Error rate: {}'.format(erro))
print('Percent correct: {}'.format((1-erro)*100))
show_DF(conff)
```

    Error rate: 0.13929999999999998
    Percent correct: 86.07000000000001
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>944.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>8.0</td>
      <td>13.0</td>
      <td>2.0</td>
      <td>7.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>1107.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>14.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.0</td>
      <td>54.0</td>
      <td>815.0</td>
      <td>26.0</td>
      <td>16.0</td>
      <td>0.0</td>
      <td>38.0</td>
      <td>22.0</td>
      <td>39.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.0</td>
      <td>18.0</td>
      <td>22.0</td>
      <td>884.0</td>
      <td>5.0</td>
      <td>16.0</td>
      <td>10.0</td>
      <td>22.0</td>
      <td>20.0</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>22.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>883.0</td>
      <td>3.0</td>
      <td>9.0</td>
      <td>1.0</td>
      <td>12.0</td>
      <td>46.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>24.0</td>
      <td>19.0</td>
      <td>3.0</td>
      <td>74.0</td>
      <td>24.0</td>
      <td>656.0</td>
      <td>24.0</td>
      <td>13.0</td>
      <td>38.0</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>17.0</td>
      <td>9.0</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>22.0</td>
      <td>17.0</td>
      <td>876.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>5.0</td>
      <td>43.0</td>
      <td>14.0</td>
      <td>6.0</td>
      <td>25.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>883.0</td>
      <td>1.0</td>
      <td>49.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>14.0</td>
      <td>48.0</td>
      <td>11.0</td>
      <td>31.0</td>
      <td>26.0</td>
      <td>40.0</td>
      <td>17.0</td>
      <td>13.0</td>
      <td>756.0</td>
      <td>18.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>16.0</td>
      <td>10.0</td>
      <td>3.0</td>
      <td>17.0</td>
      <td>80.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>75.0</td>
      <td>4.0</td>
      <td>803.0</td>
    </tr>
  </tbody>
</table>
</div>



## Confusion Matrix and Error Rate(Training Set)

Confusion Matrix:
- horizontal axis = predictions
- vertical axis = reference digit

here, we show the training error of our model


```python
YTILDA2=test_onevsall(trainY,ptrainX,testY,BET)
YHAT2=getArgMax(YTILDA2)
```


```python
err2,conff2=check_onevsall(YHAT2,trainY)
```


```python
print('Error rate: {}'.format(err2))
print('Percent correct: {}'.format((1-err2)*100))
show_DF(conff2)
```

    Error rate: 0.14446666666666663
    Percent correct: 85.55333333333334
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5669.0</td>
      <td>8.0</td>
      <td>21.0</td>
      <td>19.0</td>
      <td>25.0</td>
      <td>46.0</td>
      <td>65.0</td>
      <td>4.0</td>
      <td>60.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>6543.0</td>
      <td>36.0</td>
      <td>17.0</td>
      <td>20.0</td>
      <td>30.0</td>
      <td>14.0</td>
      <td>14.0</td>
      <td>60.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>99.0</td>
      <td>278.0</td>
      <td>4757.0</td>
      <td>153.0</td>
      <td>116.0</td>
      <td>17.0</td>
      <td>234.0</td>
      <td>92.0</td>
      <td>190.0</td>
      <td>22.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>38.0</td>
      <td>172.0</td>
      <td>174.0</td>
      <td>5150.0</td>
      <td>31.0</td>
      <td>122.0</td>
      <td>59.0</td>
      <td>122.0</td>
      <td>135.0</td>
      <td>128.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13.0</td>
      <td>104.0</td>
      <td>41.0</td>
      <td>5.0</td>
      <td>5189.0</td>
      <td>52.0</td>
      <td>45.0</td>
      <td>24.0</td>
      <td>60.0</td>
      <td>309.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>164.0</td>
      <td>94.0</td>
      <td>30.0</td>
      <td>448.0</td>
      <td>103.0</td>
      <td>3974.0</td>
      <td>185.0</td>
      <td>44.0</td>
      <td>237.0</td>
      <td>142.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>104.0</td>
      <td>78.0</td>
      <td>77.0</td>
      <td>2.0</td>
      <td>64.0</td>
      <td>106.0</td>
      <td>5448.0</td>
      <td>0.0</td>
      <td>36.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>55.0</td>
      <td>191.0</td>
      <td>36.0</td>
      <td>48.0</td>
      <td>165.0</td>
      <td>9.0</td>
      <td>4.0</td>
      <td>5443.0</td>
      <td>13.0</td>
      <td>301.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>69.0</td>
      <td>492.0</td>
      <td>64.0</td>
      <td>225.0</td>
      <td>102.0</td>
      <td>220.0</td>
      <td>64.0</td>
      <td>21.0</td>
      <td>4417.0</td>
      <td>177.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>67.0</td>
      <td>66.0</td>
      <td>26.0</td>
      <td>115.0</td>
      <td>365.0</td>
      <td>12.0</td>
      <td>4.0</td>
      <td>513.0</td>
      <td>39.0</td>
      <td>4742.0</td>
    </tr>
  </tbody>
</table>
</div>



# One vs One multi-class classifier

we will now implement 1v1 classifier

`label_ovo()` will translate the labels into binary (1,-1) for a specific given combination out of (10 choose 2)  


```python
def label_ovo(newTrainY, combo):
    
    binary=np.zeros(newTrainY.shape)
    
    for index in range(newTrainY.shape[0]):
        if newTrainY[index]==combo[0]:
            binary[index]=1
        else:
            binary[index]=-1
    
    return binary
```


```python
def onevsone(ptrainX,trainY):
    
    from itertools import combinations
    
    combos=np.array(list(combinations([j for j in range(10)], 2)))
    new_ptrainX=ptrainX
    
    BET=np.array([[]])
    
    for combo in combos:
        deletelist=[]
        for index in range(trainY.shape[0]):
            if trainY[index] not in combo:
                deletelist.append(index)    
                
        new_trainY = np.delete(trainY, deletelist, axis=0)
        newer_trainY = label_ovo(new_trainY,combo)
        
        new_ptrainX = np.delete(ptrainX, deletelist, axis=0)
        
        beta=binaryClassifier(new_ptrainX,newer_trainY)
        BET=np.append(BET,beta)
    
    BET=BET.reshape(combos.shape[0],beta.shape[0])
        
    
    
    return BET
```

Here, we train the 1v1 model. It will take longer than the one vs all model because we are training (10 choose 2)=45 models instead of 10


```python
BETT = onevsone(ptrainX,trainY)
```


```python
def count_votes(votes):
    from itertools import combinations
    import random
    
    combos=np.array(list(combinations([j for j in range(10)], 2)))
    
#     counts = {[j for j in range(10)][i]:([0]*10)[i] for i in range(10)}
    
    WINS=np.array([],dtype=int)
    
    
    for image_index in range(votes.shape[1]):
        counts = {[j for j in range(10)][i]:([0]*10)[i] for i in range(10)}
        for group_index in range(votes.shape[0]):
            if votes[group_index,image_index]==1:
                counts[combos[group_index,0]]+=1
            elif votes[group_index,image_index]==-1:
                counts[combos[group_index,1]]+=1
        winner= [k for k, v in counts.items() if v == max(counts.values())]
        WINS = np.append(WINS,winner)
        if len(winner)>1:
            tie_winner = random.choice(winner) #Ties are decided at random
            WINS = np.append(WINS,tie_winner)
        else:
            WINS = np.append(WINS,winner)
    WINS = WINS.reshape(votes.shape[1],1)
            
    return WINS, 
```


```python
def test_onevsone(ptestX, BETT):
    votes=np.array([])
    
    for group in range(BETT.shape[0]):
        vote = test_binaryClassifier(ptestX,BETT[group,:])
        votes = np.append(votes,vote)
        
    votes=votes.reshape(45,ptestX.shape[0])

    return votes
```


```python
print(votes.shape)
```

    (45, 10000)
    

Now, we will test our 1v1 model and then count the votes that each digit has received. The prediction output is placed into WINS



```python
votes = test_onevsone(ptestX,BETT)
```


```python
WINS=count_votes(votes)
```


```python
def check_ovo(WINS,testY):
    confusion=np.zeros((10,10))
    
    for index in range(WINS.shape[0]):
        confusion[testY[index],WINS[index]]+=1
    
    error=1-np.trace(confusion)/(WINS.shape[0])
    
    return error,confusion
```


```python
errovo,confovo=check_ovo(WINS,testY)
```

## Confusion Matrix and Error Rate 1v1(test set)

Confusion Matrix:
- horizontal axis = predictions
- vertical axis = reference digit



```python
print('Error rate: {}'.format(errovo))
print('Percent correct: {}'.format((1-errovo)*100))
show_DF(confovo)
```

    Error rate: 0.40780000000000005
    Percent correct: 59.21999999999999
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>823.0</td>
      <td>0.0</td>
      <td>75.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>73.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>91.0</td>
      <td>533.0</td>
      <td>329.0</td>
      <td>1.0</td>
      <td>11.0</td>
      <td>36.0</td>
      <td>14.0</td>
      <td>119.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>121.0</td>
      <td>3.0</td>
      <td>714.0</td>
      <td>60.0</td>
      <td>25.0</td>
      <td>3.0</td>
      <td>69.0</td>
      <td>0.0</td>
      <td>31.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6.0</td>
      <td>1.0</td>
      <td>19.0</td>
      <td>934.0</td>
      <td>0.0</td>
      <td>11.0</td>
      <td>7.0</td>
      <td>2.0</td>
      <td>24.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>2.0</td>
      <td>902.0</td>
      <td>1.0</td>
      <td>51.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>14.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>32.0</td>
      <td>2.0</td>
      <td>10.0</td>
      <td>180.0</td>
      <td>16.0</td>
      <td>300.0</td>
      <td>78.0</td>
      <td>14.0</td>
      <td>232.0</td>
      <td>28.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>54.0</td>
      <td>3.0</td>
      <td>74.0</td>
      <td>1.0</td>
      <td>126.0</td>
      <td>14.0</td>
      <td>676.0</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>13.0</td>
      <td>3.0</td>
      <td>49.0</td>
      <td>30.0</td>
      <td>53.0</td>
      <td>14.0</td>
      <td>10.0</td>
      <td>413.0</td>
      <td>16.0</td>
      <td>427.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>21.0</td>
      <td>1.0</td>
      <td>45.0</td>
      <td>94.0</td>
      <td>15.0</td>
      <td>95.0</td>
      <td>42.0</td>
      <td>15.0</td>
      <td>611.0</td>
      <td>35.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>6.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>22.0</td>
      <td>269.0</td>
      <td>12.0</td>
      <td>10.0</td>
      <td>185.0</td>
      <td>42.0</td>
      <td>458.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
votes2 = test_onevsone(ptrainX,BETT)
WINS2=count_votes(votes2)
errovo2,confovo2=check_ovo(WINS2,trainY)
```

## Confusion Matrix and Error Rate 1v1(training set)

Confusion Matrix:
- horizontal axis = predictions
- vertical axis = reference digit


```python
print('Error rate: {}'.format(errovo2))
print('Percent correct: {}'.format((1-errovo2)*100))
show_DF(confovo2)
```

    Error rate: 0.40495000000000003
    Percent correct: 59.504999999999995
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4983.0</td>
      <td>0.0</td>
      <td>406.0</td>
      <td>6.0</td>
      <td>12.0</td>
      <td>9.0</td>
      <td>482.0</td>
      <td>7.0</td>
      <td>14.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8.0</td>
      <td>460.0</td>
      <td>3396.0</td>
      <td>1849.0</td>
      <td>14.0</td>
      <td>109.0</td>
      <td>174.0</td>
      <td>57.0</td>
      <td>660.0</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>778.0</td>
      <td>23.0</td>
      <td>4020.0</td>
      <td>271.0</td>
      <td>109.0</td>
      <td>20.0</td>
      <td>517.0</td>
      <td>18.0</td>
      <td>173.0</td>
      <td>29.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>45.0</td>
      <td>5.0</td>
      <td>125.0</td>
      <td>5627.0</td>
      <td>12.0</td>
      <td>80.0</td>
      <td>37.0</td>
      <td>27.0</td>
      <td>131.0</td>
      <td>42.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17.0</td>
      <td>3.0</td>
      <td>16.0</td>
      <td>6.0</td>
      <td>5390.0</td>
      <td>6.0</td>
      <td>290.0</td>
      <td>15.0</td>
      <td>17.0</td>
      <td>82.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>244.0</td>
      <td>9.0</td>
      <td>98.0</td>
      <td>982.0</td>
      <td>96.0</td>
      <td>1894.0</td>
      <td>494.0</td>
      <td>121.0</td>
      <td>1256.0</td>
      <td>227.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>269.0</td>
      <td>21.0</td>
      <td>571.0</td>
      <td>4.0</td>
      <td>639.0</td>
      <td>128.0</td>
      <td>4213.0</td>
      <td>0.0</td>
      <td>71.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>98.0</td>
      <td>7.0</td>
      <td>194.0</td>
      <td>157.0</td>
      <td>375.0</td>
      <td>69.0</td>
      <td>66.0</td>
      <td>2680.0</td>
      <td>149.0</td>
      <td>2470.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>101.0</td>
      <td>12.0</td>
      <td>282.0</td>
      <td>714.0</td>
      <td>67.0</td>
      <td>543.0</td>
      <td>214.0</td>
      <td>49.0</td>
      <td>3703.0</td>
      <td>166.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>29.0</td>
      <td>1.0</td>
      <td>30.0</td>
      <td>117.0</td>
      <td>1446.0</td>
      <td>58.0</td>
      <td>73.0</td>
      <td>1222.0</td>
      <td>240.0</td>
      <td>2733.0</td>
    </tr>
  </tbody>
</table>
</div>



## Evaluating Performances of Both Classifiers

### 1vsALL:
- Training Error Rate: 0.144
- Testing Error Rate: 0.139

### 1vs1
- Training Error: 0.404
- Testing Error: 0.407



**In both implementations**, the generalize quite well on the Testing dataset and maintain a similar Error rate. This is favorable since we do not want to over parameterize to the training set. This is likely the result of filtering out the pixels that were not significant before training as well as adding the biased term (column of ones) to the training set. 

**Note** For 1vs1 implementation, there were many occurences of a tie. Ties were decided at random, thus, the error rates for this implementation is heavily skewed and inaccurate. Ties often happened when 2 numbers had similar structure such as 3&5 or 3&8 or 4&9. 

The ties are an interesting note because it also has some relationship with the misclassified digits from the 1vsALL implementation where these digits were often mistaken for each other. 


```python

```
