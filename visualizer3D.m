%% Binary Array Visualization

% Load the .mat file containing the binary array
data = load('10_stacked_masks.mat');
stacked_masks = data.stacked_masks;

% Create an isosurface plot
figure;
iso = isosurface(stacked_masks, 0.10);

% Set plot properties
patch(iso, 'FaceColor', 'blue', 'EdgeColor', 'none', 'FaceAlpha', 0.5);
axis equal;         % Make all axes equal
view(3);            % Set view angle
axis tight;         % Fit axis tightly to data

% Set labels
xlabel('Height (mm)', 'FontSize', 14);
ylabel('Width (mm)', 'FontSize', 14);
zlabel('Depth (mm)', 'FontSize', 14);

% Add grid
grid on;

% Title
title('3D Visualization at 10% Confidence Threshold', 'FontSize', 16);

% Show plot

%% Lets make a video out of it

% Load the .mat file containing the binary array
data = load('10_stacked_masks.mat');
stacked_masks = data.stacked_masks;

% Create an isosurface plot
fig = figure;
iso = isosurface(stacked_masks, 0.1);

% Set plot properties
patch(iso, 'FaceColor', 'blue', 'EdgeColor', 'none', 'FaceAlpha', 0.5);
axis equal;         % Make all axes equal
axis tight;         % Fit axis tightly to data

% Set labels
xlabel('Height (mm)', 'FontSize', 14);
ylabel('Width (mm)', 'FontSize', 14);
zlabel('Depth (mm)', 'FontSize', 14);

% Add grid
grid on;

% Title
title('3D Visualization at 10% Confidence Threshold', 'FontSize', 16);

% Create a video writer object
writerObj = VideoWriter('3D_Visualization.mp4', 'MPEG-4');
writerObj.FrameRate = 15;  % Set frame rate (frames per second)
open(writerObj);

% Capture frames while rotating the plot
for angle = 0:2:360  % Rotate plot by 2 degrees at a time
    view(angle, 30);  % Set the view angle (rotate around the y-axis)
    drawnow;          % Force MATLAB to update the plot
    frame = getframe(fig);
    writeVideo(writerObj, frame);  % Write current frame to the video
end

% Close the video writer object
close(writerObj);


