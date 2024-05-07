%% Binary Array Visualization

% Load the .mat file containing the binary array
data = load('15_stacked_masks.mat');
stacked_masks = data.stacked_masks;

% Create an isosurface plot
figure;
iso = isosurface(stacked_masks, 0.1);

% Set plot properties
patch(iso, 'FaceColor', 'blue', 'EdgeColor', 'none', 'FaceAlpha', 0.5);
axis equal;         % Make all axes equal
view(3);            % Set view angle
axis tight;         % Fit axis tightly to data

% Set labels
xlabel('Height', 'FontSize', 14);
ylabel('Width', 'FontSize', 14);
zlabel('Depth', 'FontSize', 14);

% Add grid
grid on;

% Title
title('3D Visualization at 10% Confidence Threshold', 'FontSize', 16);

% Show plot

