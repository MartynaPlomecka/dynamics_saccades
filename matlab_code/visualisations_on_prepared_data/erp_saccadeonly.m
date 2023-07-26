restoredefaultpath
rehash toolboxcache
savepath

addpath('\\psyger-stor02.d.uzh.ch\methlab\Neurometric\Anti_new\toolboxes\eeglab14_1_2b/')
eeglab

addpath('\\psyger-stor02.d.uzh.ch\methlab\Neurometric\Anti_new\data\flatiron/')
addpath('\\psyger-stor02.d.uzh.ch\methlab\Neurometric\Anti_new\plot_tools')
mycolormap = customcolormap_preset('red-white-blue');

dir = '//psyger-stor02.d.uzh.ch/methlab/Neurometric/Anti_new/data/flatiron/data_flatiron/filtered';
files = {'filteredData_antileft_old.mat', 'filteredData_antileft_young.mat', 'filteredData_antiright_old.mat', 'filteredData_antiright_young.mat', 'filteredData_proleft_old.mat', 'filteredData_proleft_young.mat', 'filteredData_proright_old.mat', 'filteredData_proright_young.mat'};
file_paths = strcat(dir, '\', files);
%%

electrodes = {
    'FEF', [24,19,11,4,124,20,12,5,118];
    'OCC', [70,75,83,74,82,71,76];
    'EOG', [127,126,25,21,14,8];
};

Fs = 500;

[b_high, a_high] = butter(6, 3/(Fs/2), 'high');
[b_low, a_low] = butter(6, 40/(Fs/2), 'low');

colors = ['b', 'r', 'g'];

max_length = 0;
for i = 1:8
    load(file_paths{i});
    num_epochs = numel(filteredData);
    for e = 1:num_epochs
        epoch_data = filteredData{e};
        for k = 1:size(electrodes, 1)
            electrode_ids = electrodes{k, 2};
            epoch_data_k = double(epoch_data(electrode_ids, :));
            epoch_data_k = epoch_data_k(:, 10:end);
            baseline_period = 391:491;
            baseline_mean = mean(epoch_data_k(:, baseline_period), 2);
            epoch_data_k = epoch_data_k - baseline_mean;
            epoch_data_k = epoch_data_k(:, 501:end-500);
            if size(epoch_data_k, 2) > max_length
                max_length = size(epoch_data_k, 2);
            end
        end
    end
end

for i = 1:8
    load(file_paths{i});

    num_epochs = numel(filteredData);

    fig = figure('Position', [100 100 1200 800]);
    set(gca, 'FontName', 'Helvetica', 'FontSize', 12);
    set(gcf, 'Color', 'w');
    set(gca, 'FontWeight', 'light');

    for row = 1:2
        subplot(2, 1, row);
        hold on;

        for j = 1:size(electrodes, 1)
            electrode_ids = electrodes{j, 2};
            averaged_data = zeros(length(electrode_ids), max_length);
            data_per_epoch = zeros(length(electrode_ids), max_length, num_epochs);

            for e = 1:num_epochs
                epoch_data = filteredData{e};
                epoch_data = double(epoch_data(electrode_ids, :));

                epoch_data = epoch_data(:, 10:end);

                baseline_period = 391:491;
                baseline_mean = mean(epoch_data(:, baseline_period), 2);
                epoch_data = epoch_data - baseline_mean;

                epoch_data = epoch_data(:, 501:end-500);

                epoch_data = interp1(1:size(epoch_data, 2), epoch_data', linspace(1, size(epoch_data, 2), max_length))';

                if row == 2
                    epoch_data = filtfilt(b_high, a_high, epoch_data')';
                    epoch_data = filtfilt(b_low, a_low, epoch_data')';
                end

                if max_length <= size(epoch_data, 2)
                    averaged_data = averaged_data + epoch_data(:, 1:max_length);
                    data_per_epoch(:,:,e) = epoch_data(:, 1:max_length);
                else
                    disp('Warning: Time range exceeds epoch size.')
                end
            end

            averaged_data = averaged_data / num_epochs;

            SEM = std(squeeze(mean(data_per_epoch,1)),0,2)/sqrt(num_epochs);

            shadedErrorBar(1:max_length, mean(averaged_data,1), SEM, 'lineprops', colors(j));
        end

        title(sprintf('Epochs normalized to longest saccade'), 'FontSize', 12);
        if row == 1
            ylabel('Amplitude (uV)')
        else
            ylabel('Amplitude (HPF 3Hz, LPF 40Hz)')
        end
        xlabel('Normalized Time')
        ylim([-2 2]);
        xlim([1 max_length]);

        line([max_length/2 max_length/2], ylim, 'Color', 'k', 'LineStyle', '--');
        legend(electrodes(:,1), 'Location', 'northeast');
        hold off;
    end

    sgtitle(strrep(files{i}, '_', ' '));
    saveas(fig, ['saccadeonly_erp_01_40_normalized_' strrep(files{i}, '.mat', '.png')]);
end
