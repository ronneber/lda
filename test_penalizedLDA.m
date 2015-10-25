%
%  some tests for the penalizedLDA implementation
%  simple example from R package. Compare results to R
%  implementation 

%% Test 1: within class standard deviation
%
x       = load('test1_x.csv');
y       = load('test1_y.csv');
wcsd    = load('test1_wscd.csv');
discrim = load('test1_discrim.csv');
xproj   = load('test1_xproj.csv');

out = penalizedLDA_L1(x, y, .14, 2, 30);

assert(all(abs(out.wcsd - wcsd) < 1e-6))

%% Test 2: discriminant vectors
%
x       = load('test1_x.csv');
y       = load('test1_y.csv');
wcsd    = load('test1_wscd.csv');
discrim = load('test1_discrim.csv');
xproj   = load('test1_xproj.csv');

out = penalizedLDA_L1(x, y, .14, 2, 30);
success = (abs(out.discrim - discrim) < 1e-2);
assert(all(success(:)))

%% Test 3: projections of x
%
x       = load('test1_x.csv');
y       = load('test1_y.csv');
wcsd    = load('test1_wscd.csv');
discrim = load('test1_discrim.csv');
xproj   = load('test1_xproj.csv');

out = penalizedLDA_L1(x, y, .14, 2, 30);
success = (abs(out.xproj - xproj) < 0.03);
assert(all(success(:)))


