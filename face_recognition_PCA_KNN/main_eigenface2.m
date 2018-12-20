%% Face recognition using PCA and KNN
% Author: Xiaocan Li
% last modified: Dec, 20, 2018
% courtesy of yalefaces dataset
% www.github.com/hsiaotsan
%% preparation
% change the hardcoded path
path='J:\Graduate\Grad1Spring\Other\FacialRecognition\yalefaces\';
H = 100; W = 100;
num_face = 165;

%% read face image and save to matrix
X_with_mean = zeros(H*W, num_face);
for i=1:165
    img_path = strcat(path,'s',num2str(i),'.bmp');
    I = imread(img_path);
    I = I(:);
    X_with_mean(:, i) = I;
end

%% get average face and show
face_avg = mean(X_with_mean, 2);
face_avg_image = uint8(reshape(face_avg, W, H));
imshow(face_avg_image); title('Average Face')

%% eigendecomposition of XX'
X_centered = bsxfun(@minus, X_with_mean, face_avg); % subtract mean

% using the property that
% if X'X*u=lambda*u
% then (XX')X*u = lambda*(Xu)
% ie. X*(eigenvector of X'X) = (eigenvector of XX')
% they share the same eigenvalues (one of them has more eigenvalues, a subset)
% the equal sign is not strict but you know that I mean

M = X_centered'*X_centered;
[V, D] = eig(M); % D is in ascending order, d1 < d2 < ...

% select the number of eigens preserved to keep 95% of original data
d = diag(D);
d = d(end:-1:1); % put d in descending order, d1 > d2 > ...
cumsum_d = cumsum(d);
percentage = cumsum_d / cumsum_d(end);
location = find(percentage>0.95);
K = location(1); % the first value that satisfies percentage > 0.95

% plot #components - percentage curve
plot(percentage);
hold on
scatter(K, percentage(K));
text(K, percentage(K), strcat('(',num2str(K),', ',num2str(percentage(K)), ')'))
title('#components -- percentage');xlabel('#components');ylabel('percentage')

% get eigendecomposition of XX'
V = fliplr(V); % make sure it's consistent with D in descending order
V_pca = V(:, 1:K);
U = X_centered*V_pca; % U is the eigendecomposition of XX'
U = normc(U);
% compute weights under new basis U
weights = U'*X_centered;
% recover face
faces_recover = bsxfun(@plus, U*weights, face_avg); % U*weights + face_avg

% show recovered face
whichone = 1;
figure;
face_rec = uint8(reshape(faces_recover(:, whichone), W, H));
imshow(face_rec); title(strcat('Recovered face #',num2str(whichone)))

%% predict using KNN, the best K is chosen in next section
% dataset split = 80% train + 20% test
%******* warning: the following split is not accurate *******
%******* to be accurate, the split should consider the portion of each class *******
load('ground_truth.mat');
n = randperm(size(weights, 2));
m = ceil(n*0.2); %  20% test
train_gnd = gnd(1:n(1:m));
test_gnd = gnd(n(1:m)+1:end);
train_data = weights(:, 1:n(1:m));
test_data = weights(:, n(1:m)+1:end);

% train KNN model, the optimal K = 1.
% fitcknn(#example*#feature, #example)
model_knn = fitcknn(train_data', train_gnd, 'NumNeighbors', 1);

% predict on given new image X, decenter:
i = 100;
img_path = strcat(path,'s',num2str(i),'.bmp');
X_new = imread(img_path);
X_new = double(X_new(:));
X = X_new - face_avg*ones(1, size(X_new, 2));
%    calculate the coefficient under the basis of eigenfaces:
w_new = U'*X; % size(w_new) = K * #X_new
%     use KNN to find closest point w_new as the prediction    
%     predict
label_predict = predict(model_knn, w_new');


%% find the optimal value of K for KNN, result k = 1.
max_iter = length(test_gnd);
accuracy=zeros(max_iter, 1);
for iter=1:max_iter
    model_knn = fitcknn(train_data', train_gnd, 'NumNeighbors', iter);
    label_predict = predict(model_knn, test_data');
    accuracy(iter) = mean(label_predict == test_gnd);
end

[max_acc, bestK] = max(accuracy);
figure;
plot(accuracy);title('KNN: K - Accuracy')