file_path = '\\psyger-stor02.d.uzh.ch/methlab/Neurometric/Anti_new/data/flatiron/data_flatiron/filtered/filteredData_proleft_young.mat';
load(file_path);

num_epochs = numel(filteredData);
num_electrodes = size(filteredData{1}, 1);

%x coordinate is electrode 131
sensor_electrode = 131;

pre_peak_length = 5; % The number of timepoints before the peak to include in the pre-peak epoch
post_peak_length = 10; % The number of timepoints after the peak to include in the post-peak epoch

pre_peak_means = zeros(num_electrodes, 1);
post_peak_means = zeros(num_electrodes, 1);

peak_times = []; 
kept_epochs = 0; 

for e = 1:num_epochs
    epoch_data = double(filteredData{e});

    baseline_period = 400:500; 
    baseline_mean = mean(epoch_data(:, baseline_period), 2);
    epoch_data = epoch_data - baseline_mean; %baselinee correction

    % Focus on timepoints 500:530
    epoch_data = epoch_data(:, 500:530);

    sensor_data = epoch_data(sensor_electrode, :);
    sensor_velocity = diff(sensor_data); % First deriv to get the velocity data
    [~, peak_index] = max(abs(sensor_velocity)); %  timepoint of peak velocity

    % Check that we have a full pre-peak and post-peak epoch
    if peak_index <= pre_peak_length || peak_index >= size(epoch_data, 2) - post_peak_length
        continue; % Skip this epoch if peak occurs too early or too late
    end

    kept_epochs = kept_epochs + 1;

    peak_times = [peak_times, peak_index];

    pre_peak_data = epoch_data(:, (peak_index-pre_peak_length+1):peak_index);
    post_peak_data = epoch_data(:, (peak_index+1):(peak_index+post_peak_length));

    pre_peak_means = pre_peak_means + mean(pre_peak_data, 2);
    post_peak_means = post_peak_means + mean(post_peak_data, 2);
end
pre_peak_means = pre_peak_means / kept_epochs;
post_peak_means = post_peak_means / kept_epochs;

% Absolute difference threshold 
diff_threshold = 0.5;

% Compare pre and post peak means and identify significant electrodes
significant_electrodes = [];
for i = 1:128 
    if abs(pre_peak_means(i) - post_peak_means(i)) > diff_threshold
        significant_electrodes = [significant_electrodes, i];
    end
end

disp('Significant electrodes:');
disp(significant_electrodes);

figure;
hist(peak_times, 1:max(peak_times));
xlabel('Timepoint of peak velocity');
ylabel('Number of epochs');
title('Histogram of peak velocity timepoints');

plot_ERPs_of_electrodes(significant_electrodes, filteredData, kept_epochs, pre_peak_length, post_peak_length);


function plot_ERPs_of_electrodes(electrodes, filteredData, kept_epochs, pre_peak_length, post_peak_length)
    figure; 
    hold on; 
    colors = hsv(length(electrodes)); 
    legendInfo = cell(1, length(electrodes));
    
    for idx = 1:length(electrodes)
        electrode = electrodes(idx);
        erp_data = zeros(kept_epochs, 31); % Initialize with 31 columns because of the slicing from 500:530
        count_epochs = 0; 
        peak_times = []; 

        % Calculate ERP for each epoch
        for epoch = 1:numel(filteredData)
            epoch_data = double(filteredData{epoch});

            % Focus on timepoints 500:530
            epoch_data = epoch_data(:, 500:530);

            sensor_data = epoch_data(electrode, :);  % Use the specific electrode data here
            sensor_velocity = diff(sensor_data); % First derivative to get the velocity data
            [~, peak_index] = max(abs(sensor_velocity)); % Identify the timepoint of peak velocity

            % Check that we have a full pre-peak and post-peak epoch
            if peak_index <= pre_peak_length || peak_index >= size(epoch_data, 2) - post_peak_length
                continue; % Skip this epoch if peak occurs too early or too late
            end

            peak_times = [peak_times, peak_index]; % Add peak time to the array for this electrode

            count_epochs = count_epochs + 1;
            erp_data(count_epochs, :) = epoch_data(electrode, :).';
        end

        erp_mean = mean(erp_data, 1);
        erp_sem = std(erp_data, 0, 1) / sqrt(kept_epochs);

        shadedErrorBar(1:31, erp_mean, erp_sem, 'lineprops', {'Color', colors(idx, :)});
        legendInfo{idx} = ['Electrode ' num2str(electrode)]; 

        % Plot vertical line at average peak time for this electrode
        avg_peak_time = mean(peak_times);
        yl = ylim; % Get the current y-axis limits
        plot([avg_peak_time, avg_peak_time], yl, 'Color', colors(idx, :), 'LineStyle', '--'); % Plot the vertical line
    end
    
    title('ERP for Significant Electrodes');
    xlabel('Timepoints');
    ylabel('Amplitude (uV)');
    legend(legendInfo); 
    hold off; 
end

