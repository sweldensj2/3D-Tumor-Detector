% Load the .mat file containing the binary array
data = load('stacked_masks.mat');
stacked_masks = data.stacked_masks;

% Create an isosurface plot
figure;
iso = isosurface(stacked_masks, 0.5);

% Set plot properties
patch(iso, 'FaceColor', 'blue', 'EdgeColor', 'none', 'FaceAlpha', 0.5);
axis equal;         % Make all axes equal

% Set axis limits
xlim([0 size(stacked_masks, 1)]);
ylim([0 size(stacked_masks, 2)]);
zlim([0 size(stacked_masks, 3)]);

% Set labels
xlabel('X');
ylabel('Y');
zlabel('Z');

% Add grid
grid on;

% Title
title('3D Visualization of Binary Array');

% Show plot
