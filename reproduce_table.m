%  Reproduce table 3, row 3, column 4 of
%
%  Witten, D. M. & Tibshirani, R. Penalized classification using
%  Fisher’s linear discriminant. Journal of the Royal Statistical
%  Society: Series B (Statistical Methodology) 73, 753–772 (2011).
%
nRepetitions = 10;
lambdas = [0.006 : 0.0001 : 0.009];
nSubsets = 4;
nDiscriminants = 2;
unequalPriorsFlag = true;
maxIter = 30;

%  load data 
%
x = hdf5read('sun.h5', '/x');
y = hdf5read('sun.h5', '/y');
nObservations = size(x,1);

%  run test with nRepetitions
%
testErrors = zeros(nRepetitions, 1);
testUsedFeatures = zeros( nRepetitions,1);
for iter = 1:nRepetitions
  %  random split into training and test set
  % 
  idx = randperm( nObservations);
  trainIdx = idx(1:0.75*nObservations);
  testIdx = idx(0.75*nObservations+1:end);
  
  xTrain = x(trainIdx,:);
  yTrain = y(trainIdx);
  
  xTest = x( testIdx,:);
  yTest = y( testIdx);
  
  %  do cross validation on training set to find best lambda
  %
  bestLambdaIdx = 1;
  bestError = 1e10;
  bestUsedFeatures = 1e10;
  for lambdaIdx = 1:length( lambdas)
	lambda = lambdas(lambdaIdx);
	%  sort training data by labels to have balanced classes in
    %  cross validation
	%
	[yTrain idx] = sort( yTrain);
	xTrain = xTrain(idx,:);
	[errors nUsedFeatures] = crossVal_penalizedLDA_L1(...
		xTrain, yTrain, lambda, nDiscriminants, nSubsets, unequalPriorsFlag, maxIter);
	meanError = mean(errors);
	meanUsedFeatures = mean(nUsedFeatures);
	
	%  display the errors
	%
	disp( ['lambda = ' num2str(lambda)...
		   ': crossval error = ' num2str(meanError) ' (' ...
		   num2str(std(errors)) ...
		   '); num features = ' num2str(meanUsedFeatures)]);
	
	% check if the result improved
	%
	if (meanError < bestError || ...
		(meanError == bestError ...
		 && meanUsedFeatures < bestUsedFeatures))
	  bestLambdaIdx = lambdaIdx;
	  bestError = meanError;
	  bestUsedFeatures = meanUsedFeatures;
	end
  end
	
  %  compute error on test set with best lambda
  %
  bestLambda = lambdas(bestLambdaIdx);
  ldaResult = penalizedLDA_L1( xTrain, yTrain, bestLambda, nDiscriminants, maxIter);
  predResult = predict_NN_LDA( xTest, ldaResult, unequalPriorsFlag);
  yPred = predResult.yPred;  
  testErrors(iter) = sum(yTest ~= yPred) ;
  testUsedFeatures(iter) = sum(sum( abs(ldaResult.discrim) > 0, 2) > 0); 

  %  display test error
  %
  disp('------------------------------')
  disp( ['best lambda = ' num2str(bestLambda) ...
		 ': test error = ' num2str( testErrors(iter)) ...
		 '; num features = ' num2str(testUsedFeatures(iter))]);
  disp('==============================')
end

%  display final results
%
disp(['Results over ' num2str(nRepetitions) ' training-test splits'])
disp(['Errors:   ' num2str(mean( testErrors)) ...
	  ' (' num2str(std(testErrors)) ')'])
disp(['Features: ' num2str(mean( testUsedFeatures)) ...
	  ' (' num2str(std(testUsedFeatures)) ')'])
