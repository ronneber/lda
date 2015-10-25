function [errors nUsedFeatures] = crossVal_penalizedLDA_L1(...
	x, y, lambda, nDiscriminants, nSubsets, unequalPriorsFlag, maxiter)
%% compute n-fold cross validation using penalizedLDA_L1()
%
%  Parameters:
%  x:                 data matrix (n observations, n features)
%  y:                 class vector (n oservations) containing class indices
%                     1,2,...,nclasses. 
%  lambda:            weight for L1 sparseness term
%  nDiscriminants:    number of desired discriminant vectors
%  nSubsets:          number of subsets in cross validation
%  unequalPriorsFlag: flag if the priors from the training data
%                     shall be used in the prediction
%  maxIter:           maximal number of iterations passed to
%                     penalizedLDA_L1() 
%
%  Return values:
%  errors:        number(!) of incorrectly classified samples
%                 for each subset
%  nUsedFeatures: number of used features for each subset (counting
%                 features that are used in at least one of the
%                 discriminant vectors) 
%
%  The subsets are extracted interleaved from the data, e.g. for
%  10-fold cross validation, the subset indices are
%    subset 1:  1, 11, 21, 31, ...
%    subset 2:  2, 12, 22, 32, ...
%    ...
%  This allows easily to achieve subsets with balanced classes,
%  just by passing the vectors ordered by class label.  
%
nObservations = size(x,1);
nClasses = max(y);

errors = zeros(nSubsets,1);
nUsedFeatures = zeros(nSubsets,1);

for si = 1:nSubsets
  %  compute indices for validation set and training set
  %
  valSetIndices = si:nSubsets:nObservations;
  trainSetIndices = 1:nObservations;
  trainSetIndices(valSetIndices) = [];
  
  %  do training
  %
  xTrain = x(trainSetIndices,:);
  yTrain = y(trainSetIndices);
  ldaResult = penalizedLDA_L1( xTrain, yTrain, lambda, nDiscriminants, ...
							   maxiter);
  
  %  do prediction
  %
  xVal = x(valSetIndices,:);
  yVal = y(valSetIndices);
  predResult = predict_NN_LDA( xVal, ldaResult, unequalPriorsFlag);
  yPred = predResult.yPred;
  
  %  compute the number of wrong predictions and number of used
  %  features 
  %
  errors(si) = sum(yVal ~= yPred);
  nUsedFeatures(si) = sum(sum( abs(ldaResult.discrim) > 0, 2) > 0); 
end
