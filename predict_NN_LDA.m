function predResult = predict_NN_LDA( xTest, ldaResult, unequalPriorsFlag)
%% predict the class label using nearest neighbor classifier to the
%  class centroids in the space spanned by the discriminant vectors.
%
%  Paraemters:
%  xTest:             data matrix (n observations, n features)
%  ldaResult:         result from penalizedLDA_L1() on training data
%  unequalPriorsFlag: flag if the priors from the training data
%                     shall be used
%
%  Return values:
%  predResult.xTestProj: the test vectors projected to the space
%                        spanned by the discriminant vectors
%  predResult.yPred:     predicted class for each test vector
%

%  Map the test vectors into the space spanned by the discriminant
%  vectors 
%
nTestVecs = size(xTest,1);
xTestUnitStd = (xTest - repmat(ldaResult.xOrigMean,[nTestVecs,1]))...
	./ repmat(ldaResult.wcsd', [nTestVecs,1]);
xTestProj = xTestUnitStd * ldaResult.discrim;
  
%  compute distances to the centroids
%
nClasses = size(ldaResult.classCentroids,1);
dists = zeros(nTestVecs, nClasses);
for ci = 1:nClasses
  centerRep = repmat(ldaResult.classCentroids(ci,:), [nTestVecs,1]);
  dists(:,ci) = sqrt( sum( (xTestProj - centerRep).^2, 2));
end

%  find closest distance for each vector
%
if (unequalPriorsFlag)
  negdists = -0.5 * dists.^2 + repmat( log(ldaResult.classPriors)', ...
									[nTestVecs,1]);
  [~, yPred] = max(negdists, [], 2);
else
  [~, yPred] = min(dists, [], 2);
end

predResult.xTestProj = xTestProj;
predResult.yPred = yPred;
