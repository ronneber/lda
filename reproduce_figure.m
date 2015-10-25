%  Reproduce Figure 2 of
%
%  Witten, D. M. & Tibshirani, R. Penalized classification using
%  Fisher’s linear discriminant. Journal of the Royal Statistical
%  Society: Series B (Statistical Methodology) 73, 753–772 (2011).
%

%  load data 
%
x = hdf5read('sun.h5', '/x');
y = hdf5read('sun.h5', '/y');

%  do the LDA with L1 penalization
%  we use lambda = .008 and 2 discriminant vectors
tic
lambda = 0.008;
nDiscriminants = 2;
maxIter = 30;
out = penalizedLDA_L1(x, y, lambda, nDiscriminants, maxIter);
toc

%  create the plot
%
xproj = out.xproj;
plot( xproj(y==1,1), xproj(y==1,2), 'xr', ...
	  xproj(y==2,1), xproj(y==2,2), 'ok', ...
	  xproj(y==3,1), xproj(y==3,2), '*b', ...
	  xproj(y==4,1), xproj(y==4,2), '+g')

legend( {'astrocytomas', 'glioblastomas', 'non-tumor', ...
		 'oligodendrogliomas'})
xlabel('1st Discriminant Vector')
ylabel('2nd Discriminant Vector')
