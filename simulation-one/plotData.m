function plotData(class1, class2)

scatter(class1(:, 1), class1(:, 2), 'r', 'o'); hold;
scatter(class2(:, 1), class2(:, 2), 'b', '+'); hold;
legend('Class 1', 'Class 2');
xlabel('X_1'); ylabel('X_2');
title('Visualize generated data');

end
