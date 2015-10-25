function ldaResult = penalizedLDA_L1( x, y, lambda, nDiscriminants, maxIter)
%% penalizedLDA 
%  computes a LDA with L1-penalization on the discriminant vectors
%
%  Parameters:
%  x:              data matrix (n observations, n features)
%  y:              class vector (n oservations) containing class indices
%                  1,2,...,nclasses. 
%  lambda:         weight for L1 sparseness term
%  nDiscriminants: number of desired discriminant vectors
%  maxIter:        maximal number of iterations to optimize each
%                  discriminant vector 
%
%  Return Values:
%  ldaResult.xOrigMean:      mean of training vectors. E.g. for shifting
%                            new test vectors 
%  ldaResult.wcsd:           average within-class standard deviation,
%                            e.g. for scaling new test vectors
%  ldaResult.discrim:        discriminant vectors (in shifted and scaled
%                            space) 
%  ldaResult.xproj:          training vectors mapped to discriminant space
%  ldaResult.classCentroids: class centroids in discriminant space
%  ldaResult.classPriors:    class priors inferred from training data
%
%  Reference: Witten, D. M. & Tibshirani, R. Penalized classification
%  using Fisher’s linear discriminant. Journal of the Royal Statistical
%  Society: Series B (Statistical Methodology) 73, 753–772 (2011).
%

[nObservations, nFeatures] = size(x);
nClasses = max(y);

%  encode class labels as indicator matrix (nObservations x nClasses) 
%
yMat = full(sparse(1:nObservations, y, ones(nObservations,1)));

%  compute the average within-class standard deviation.
%  Due to high-dim feature space, use the diagonal estimate.
%  I.e. wcsd only contains the diagonal components
%
sqsum = zeros(nFeatures,1);
for k = 1:nClasses
  v = x(y==k, :);
  sqsum = sqsum + sum((v - repmat(mean(v),[size(v,1) 1])).^2)';
end
wcsd = sqrt(sqsum / nObservations);

%  Make features mean free and transform them to the space with unit
%  within-class standard deviation (this is a simple scaling here,
%  because we use the diagonal estimate, see above)
%  Working in this space reduces Fisher's discriminant problem to a
%  standard Eigenproblem (see page 755)
%
xOrig = x;
x = (x - repmat(mean(x),[size(x,1),1])) ./ repmat(wcsd', [size(x,1),1]);

%  compute the sqrt of the between-class covariance matrix to "inject" the
%  orthogonal projection matrix for each discriminant vector 
%  (see equation (7) of Witten et al)
%
yMatScaled = yMat ./ repmat( sqrt( sum(yMat)), [nObservations,1]);
sqrtCovarBetween = yMatScaled' * x / sqrt(nObservations);

%  Algorithm 1 from Witten et al
%  due to space transformation (see above) we estimate the
%  substituted betas here.
%
%  setup matrix for discriminant vectors (initialized to zero) 
%
betas = zeros( nFeatures, nDiscriminants);

for k = 1:nDiscriminants
  %  Algorithm step (a):
  %  Define an orthogonal projection matrix P that projects onto the space
  %  that is orthogonal to all existing discriminant vectors
  %
  if( k == 1) 
	P = eye( nClasses);
  else
	[U,S,V] = svd( sqrtCovarBetween * betas);
	% eigenvectors with an eigenvalue < 1e-10 are considered valid
	%
	validEigenVectors = U(:, diag(S) > 1e-10);
	P = eye( nClasses) - validEigenVectors * validEigenVectors';
  end
  
  %  Algorithm step (b) and (c): 
  %  due to the substitution the within-class-covariance is 1 and can
  %  be omitted.  For fast computation we the use "economy" version of
  %  SVD -- we need only the first eigenvalue and eigenvector.
  %  (use trick from R implementation to keep matrix for svd small)
  %
  [U,S,V] = svd( sqrtCovarBetween' * P, 'econ');
  beta = U(:,1);
  
  %  ensure, that all components are penalized equally
  %  (see section 4.2.4)
  %
  lambdaFactor = S(1,1)^2;
  
  %  Algorithm step (d)
  %  (solution from proposition 2 in section 4.3.2)
  %
  objective = 0;
  for iter = 1:maxIter
	% equation (23)
	% soft thresholding
	%
	tmp = (sqrtCovarBetween' * P) * (sqrtCovarBetween * beta);
	beta = sign(tmp) .* max(abs(tmp) - lambdaFactor*lambda/2, 0);
	
	% equation (21)
	%
	beta = beta ./ sqrt(sum(beta.^2));
	beta(isnan(beta)) = 0;

	% compute objective and check for termination criterion
	% (relaitve change smaller than 1e-6 or beta equal to 0)
	%
	oldObjective = objective;
	objective = beta' * sqrtCovarBetween' * P * sqrtCovarBetween * beta ...
		- lambdaFactor * lambda * sum(abs(beta));
	if( iter > 4 && ...
		(abs( objective - oldObjective) / max(1e-3, objective) < 1e-6 ...
		 || sum( abs( beta)) == 0))
	  break;
	end
  end
  betas(:,k) = beta;
end

%  map the training vectors into the space spanned by the
%  discriminant vectors
%
xproj = x * betas;

%  compute the class centroids in the projected space
%
classCentroids = zeros( nClasses, nDiscriminants);
for ci = 1:nClasses
  classCentroids(ci,:) = mean( xproj(y == ci, :));
end

%  compute the class priors
%
priors = zeros(nClasses,1);
for ci = 1:nClasses
  priors(ci) = mean( y == ci);
end

% put all return values into ldaResultput struct
%
ldaResult.discrim = betas;
ldaResult.xproj = xproj;
ldaResult.xOrigMean = mean(xOrig);
ldaResult.wcsd = wcsd;
ldaResult.classCentroids = classCentroids;
ldaResult.classPriors = priors;
