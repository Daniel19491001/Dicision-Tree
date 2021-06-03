clc
clear all
close all;
load fisheriris;  % 加载数据集
% 数据可视化
x = meas;
y = species;
class = unique(y);
attr = {'萼片长度', '萼片宽度', '花瓣长度', '花瓣宽度'};
ind1 = ismember(y, class{1});
ind2 = ismember(y, class{2});
ind3 = ismember(y, class{3});
s=10;
for i=1:4
   for j=1:4
      subplot(4,4,4*(i-1)+j);
      if i==j
          set(gca, 'xtick', [], 'ytick', []);
          text(.2, .5, attr{i});
          set(gca, 'box', 'on');
          continue;
       end
      scatter(x(ind1,i), x(ind1,j), s, 'r', 'MarkerFaceColor', 'r');
      hold on
      scatter(x(ind2,i), x(ind2,j), s, 'b', 'MarkerFaceColor', 'b');
      hold on
      scatter(x(ind3,i), x(ind3,j), s, 'g', 'MarkerFaceColor', 'g');
       set(gca, 'box', 'on');
   end
end
% 随机划分训练集和测试集
ratio=8/3;
num = length(x);
num_test = round(num/(1+ratio));
num_train = num - num_test;
index = randperm(num);
x_train = x(index(1:num_train),:);
y_train = y(index(1:num_train));
x_test = x(index(num_train+1:end),:);
y_test = y(index(num_train+1:end));
% 构建决策树并预测结果
tree = fitctree(x_train, y_train);
y_test_p = predict(tree, x_test);
% 查看决策树视图
view(tree);
view(tree,'mode','graph');
% 计算预测准确率
acc = sum(strcmp(y_test,y_test_p))/num_test;
disp(['识别准确率 ', num2str(acc)]);

%剪枝
[~,~,~,bestlevel] = cvLoss(tree,'subtrees','all','treesize','min')
ptree = prune(tree,'Level',bestlevel);
view(ptree,'mode','graph')
%计算剪枝后决策树的重采样误差和交叉验证误差
resubPrune = resubLoss(ptree)
lossPrune = kfoldLoss(crossval(ptree))
