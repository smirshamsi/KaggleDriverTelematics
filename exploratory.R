

readFolder<-function(wd="./"){
  totFiles<-length(dir(wd))
  allData<-list()
  for (i in 1:totFiles){
    allData[[i]]<-read.csv(paste(wd,as.character(i),".csv",sep=''))
  }
  allData
}
#######
intervalFeature<-function(inpData,Fun=mean,step=100){
  aggregate(inpData, by=list(0:(length(inpData)-1) %/% step), mean)[,2]
}

#######
fExtractXY<-function(inpData,FUN=mean){
  FUNXY<-NULL
  for (i in 1:length(inpData)){
    tempXY<-c(FUN(inpData[[i]][,1]),FUN(inpData[[i]][,2]))
    FUNXY<-rbind(FUNXY,tempXY)
  }
  FUNXY
}

#####
vecDiff<-function(inpVec){
  lInpVec<-length(inpVec)
  outVec<-inpVec[2:lInpVec]-lInpVec[1:(lInpVec-1)]
}
####
lDiff<-function(inData){
  ####take the vecDiff of every XY for each item of list 
  lInData<-length(inData)
  outData<-list()
  for (i in 1:lInData){
    outData[[i]]<-cbind(vecDiff(inData[[i]][,1]),vecDiff(inData[[i]][,2]))
  }
  outData
}
####
lEuclid<-function(inData){
  ####take the X^2+Y^2 of every XY for each item of list 
  lInData<-length(inData)
  outData<-list()
  for (i in 1:lInData){
    outData[[i]]<-sqrt((inData[[i]][,1])^2)+((inData[[i]][,2])^2)
  }
  outData
}
####
vecHist<-function(inData,inMin,inMax,breaks=20){
  ####take the hist of all items of list 
  lInData<-length(inData)
  vecTempIn<-NULL
  outData<-NULL
  for (i in 1:lInData){
    vecTempIn<-c(VecTempIn,inData[[i]])
  }
  
  maxVecTempIn<-max(vecTempIn); minVecTempIn<-min(vecTempIn);
  breaks<-seq(minVecTempIn
  outData<-hist(VectempIn,breaks=breaks,plot=False,probability=TRUE)
  outData
}
####

###########################################################################################
wd<-"~/Downloads/drivers/1/"
allData<-readFolder(wd)
meanXY1<-fExtractXY(allData,mean)

sdXY1<-fExtractXY(allData,sd)
########
wd<-"~/Downloads/drivers/240/"
allData<-readFolder(wd)
meanXY2<-fExtractXY(allData,mean)

sdXY2<-fExtractXY(allData,sd)

plot(meanXY1,col='red')
points(meanXY2,col='blue')
maxX<-max(fExtractXY(allData,max)[,1])
minX<-min(fExtractXY(allData,min)[,1])
maxY<-max(fExtractXY(allData,max)[,2])
minY<-min(fExtractXY(allData,min)[,2])

plot(allData[[1]],col=1,xlim=c(minX,maxX),ylim=c(minY,maxY))
for (i in 2:200){
  points(allData[[i]],col=i)
}

dyOverdx<-allData
