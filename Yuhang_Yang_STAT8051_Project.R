setwd("~/Desktop/UMN FALL 2019/STAT 8051/Project/mn2019comp")
driver=read.table("drivers.csv",header = TRUE,sep = ",")
policies=read.table("policies.csv",header = TRUE,sep = ",")
vehicle=read.table("vehicles.csv",header = TRUE,sep = ",")
driver=driver[,-1]
policies=policies[,-1]
vehicle=vehicle[,-1]
mergeCols ="policy_id"
data1=merge(driver, policies, by = mergeCols)
rawdata=merge(data1, vehicle, by = mergeCols)



# read the final train dataset that processed by Xuejie, Difan, Yuhang
#finaltrain=read.table("final_all_data.csv",header = TRUE,sep = ",")
finaltrain=finaltrain[,-1]

# Split data
train=finaltrain[which(finaltrain$split=="Train"), ]
test=finaltrain[which(finaltrain$split=="Test"), ]


# fit the first model with as many predictors as possible, some variables are ignored (colors, num_owned_veh etc)
model1=glm(convert_ind~discount+Home_policy_ind+quoted_amt+Prior_carrier_grp+credit_score+Cov_package_type+CAT_zone+number_drivers+gender+age.x+safty_rating+high_education_ind+car_no+age.y+year+dependent+own+rent+home.driveway+street+unknown+parking.garage+owned+loaned+leased+AL+FL+CT+GA+MN+NJ+NY+WI,family = binomial,train)

# this model suggest some predictors have high collinearity
alias(model1)

# fit another model without predictors that gave collinearity

model2=glm(convert_ind~discount+Home_policy_ind+quoted_amt+Prior_carrier_grp+credit_score+Cov_package_type+CAT_zone+number_drivers+gender+age.x+safty_rating+high_education_ind+car_no+age.y+year+dependent+own+home.driveway+street+unknown+owned+loaned+AL+FL+CT+GA+MN+NJ+NY,family = binomial,train)



# fill the missing values in each column using median

for (i in 1:dim(test)[2]) {
  if (any(is.na(test[,i]))==TRUE) {
    test[which(is.na(test[,i])),i]=median(test[,i],na.rm = TRUE)
  }
  else {
    test[,i]=test[,i]
  }
}

# make prediction

b=predict(model2,test,type = "response")

# organize the result 
result=data.frame(test$policy_id,b)
result$test.policy_id=as.character(result$test.policy_id)
result$test.policy_id=paste0("policy_", result$test.policy_id)

# group the data that has same policy id

finalresult=aggregate(b~test.policy_id,result,mean)

# generate final result
colnames(finalresult)=c("policy_id","conv_prob")
write.csv(finalresult,"result.csv",row.names = F)


## xgboost 

library("xgboost")

# using convert_ind as the label
train_label=train[,15]

# get rid of some columns like num_loaned_veh, split, etc.
train_xg=train[,-c(10,11,12,13,14,15,16)]
resultxg=xgboost(data = as.matrix(train_xg), label = train_label, max.depth = 2, eta = 1, nthread = 2, nrounds = 2, objective = "binary:logistic")

## fill the missing values in test set using median

for (i in 1:dim(test)[2]) {
  if (any(is.na(test[,i]))==TRUE) {
    test[which(is.na(test[,i])),i]=median(test[,i],na.rm = TRUE)
  }
  else {
    test[,i]=test[,i]
  }
}
# do the same with test set 
test_xg=test[,-c(10,11,12,13,14,15,16)]

# make predictions
pred=predict(resultxg,as.matrix(test_xg))
result_xg=data.frame(test$policy_id,pred)
result_xg$test.policy_id=as.character(result_xg$test.policy_id)
result_xg$test.policy_id=paste0("policy_", result_xg$test.policy_id)
finalresultxg=aggregate(pred~test.policy_id,result_xg,mean)
colnames(finalresultxg)=c("policy_id","conv_prob")

# cross validation using xgboost
xgb.cv(data = as.matrix(train_xg), label = train_label, max.depth = 6, eta = 0.3, nround = 2, objective = "binary:logistic", nfold = 5)


# importance plot
finall=read.table("data_final_xj.csv",header = TRUE,sep = ",")
Finaltrain=finall[which(finall$split=="Train"), ]
Finaltest=finall[which(finall$split=="Test"), ]
#Finaltrain=Finaltrain[,-c(1,2)]
train_label=Finaltrain$convert_ind
drops=c('X','policy_id','convert_ind','split','Unnamed..0')
train_xg=Finaltrain[,!(names(Finaltrain) %in% drops)]
train_xg=train_xg[,-c(37,39,40,41,42,43,44,45,46,47,48,49,50,51)]
library("Matrix")
A=as(as.matrix(train_xg),"sparseMatrix")
library("xgboost")
bst=xgboost(data = A, label = train_label, max.depth = 6, eta = 0.3, nthread = 4, nrounds = 25, objective = "binary:logistic")
importance_matrix <- xgb.importance(A@Dimnames[[2]], model = bst)
xgb.plot.importance(importance_matrix)
