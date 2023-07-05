%%
gsp_start

%% Creating Weight Matrix

W_G = zeros(8, 8);

W_G(1, :) = [0, 1, 1, 0, 0, 0, 0, 1];
W_G(2, :) = [1, 0, 1, 1, 1, 0, 0, 1];
W_G(3, :) = [1, 1, 0, 1, 0, 0, 0, 0];
W_G(4, :) = [0, 1, 1, 0, 1, 1, 0, 1];
W_G(5, :) = [0, 1, 0, 1, 0, 1, 1, 1];
W_G(6, :) = [0, 0, 0, 1, 1, 0, 1, 0];
W_G(7, :) = [0, 0, 0, 0, 1, 1, 0, 0];
W_G(8, :) = [1, 1, 0, 1, 1, 0, 0, 0];

%% Creating Graph by Weight Matrix

G = gsp_graph(W_G);
coords = gsp_compute_coordinates(G , 3);
G.coords = coords;

figure;
gsp_plot_graph(G)
title("G graph")

%% finding EigenVectors

[eV, LAMBDA] = eig(G.L);

%% generating original X signal:

x = 2 * eV(:, 1) + eV(:, 2);

figure;
gsp_plot_signal(G, x)
title("Original Signal on the graph")

%% adding white noise to signal!

n_ = normrnd(0, 1, [1, length(x)]);
NSR = 0.5 * sum(x .^ 2) / sum(n_ .^ 2);
x_noisy = x + n_' * NSR;

figure;
gsp_plot_signal(G, x_noisy)
title("Noisy Signal on the graph (SNR = 10dB)")

n_NF_ = x - x_noisy;
SNR_NF = 10 * log10(sum(x .^ 2) / sum(n_NF_ .^ 2));

%% plot in a line!

figure;
subplot(2, 1, 1)
stem(1 : 8, x_noisy)
title("Noisy signal value on each vertex")
subplot(2, 1, 2)
stem(1 : 8, x)
title("Original signal value on each vertex")
hold off

%% in Spectral Domain (L):

x_original_hat_L = eV' * x;
x_noisy_hat_L = eV' * x_noisy;

figure;
subplot(2, 1, 1)
stem(1 : 8, x_noisy_hat_L)
title("Noisy signal spectrum (L)")
subplot(2, 1, 2)
stem(1 : 8, x_original_hat_L)
title("Original signal spectrum (L)")
hold off

%% in Spectral Domain (W_norm):

D = diag(G.d);

W_G_norm = D ^ (-0.5) * W_G * D ^ (-0.5);

[eV_WN, LAMBDA_WN] = eig(W_G_norm);

x_original_hat_WN = eV_WN' * x;
x_noisy_hat_WN = eV_WN' * x_noisy;

figure;
subplot(2, 1, 1)
stem(1 : 8, x_noisy_hat_WN)
title("Noisy signal spectrum (WN)")
subplot(2, 1, 2)
stem(1 : 8, x_original_hat_WN)
title("Original signal spectrum (WN)")
hold off

%% Filtering in L sense:

h_L = [1, 1, 0, 0, 0, 0, 0, 0];
F_L = diag(h_L);

x_denoised_L = eV * F_L * x_noisy_hat_L;

figure;
stem(1 : 8, h_L)
title("LPF response in spectral domain in L sense")

figure;
subplot(3, 1, 1)
stem(1 : 8, x_noisy)
title("Noisy signal value on each vertex")
subplot(3, 1, 2)
stem(1 : 8, x)
title("Original signal value on each vertex")
subplot(3, 1, 3)
stem(1 : 8, x_denoised_L)
title("Denoised signal by L sense value on each vertex")
hold off

%% Filtering in WN sense:

h_WN = [0, 0, 0, 0, 0, 0, 1, 1];
F_WN = diag(h_WN);

x_denoised_WN = eV_WN * F_WN * x_noisy_hat_WN;

figure;
stem(1 : 8, h_WN)
title("LPF response in spectral domain in WN sense")

figure;
subplot(3, 1, 1)
stem(1 : 8, x_noisy)
title("Noisy signal value on each vertex")
subplot(3, 1, 2)
stem(1 : 8, x)
title("Original signal value on each vertex")
subplot(3, 1, 3)
stem(1 : 8, x_denoised_WN)
title("Denoised signal by WN sense value on each vertex")
hold off

%% Computing SNR for filtered Signals!

% n_NF_ = x - x_noisy; %non filtered noisy x noise
% SNR_NF = 10 * log10(sum(x .^ 2) / sum(n_NF_ .^ 2));

n_F_L = x - x_denoised_L; %filtered signal in sense of laplacian
SNR_F_L = 10 * log10(sum(x .^ 2) / sum(n_F_L .^ 2));

n_F_WN = x - x_denoised_WN; %filtered signal in sense of laplacian
SNR_F_WN = 10 * log10(sum(x .^ 2) / sum(n_F_WN .^ 2));

%% Check if filters are LSI!

% L sense:

d_L = abs(F_L * G.L - G.L * F_L);

% WN sense:

d_WN = abs(F_WN * W_G_norm - W_G_norm * F_WN);

% no! They are not LSI!

%% Generating LSI Filters - step 1, laplacian:

tmp = fliplr(vander(diag(LAMBDA)));
Van = tmp(:, 1 : 3);

frequency_response = [1, 1, 0, 0, 0, 0, 0, 0]';

h_l_si = pinv(Van) * frequency_response;

F_L_lsi = h_l_si(1) * LAMBDA ^ 0 + h_l_si(2) * LAMBDA ^ 1 + h_l_si(3) * LAMBDA ^ 2;

x_denoised_L_lsi = eV * F_L_lsi * x_noisy_hat_L;

figure;
stem(1 : 8, diag(F_L_lsi))
title("LPF response in spectral domain in L sense")

figure;
subplot(3, 1, 1)
stem(1 : 8, x_noisy)
title("Noisy signal value on each vertex")
subplot(3, 1, 2)
stem(1 : 8, x)
title("Original signal value on each vertex")
subplot(3, 1, 3)
stem(1 : 8, x_denoised_L_lsi)
title("Denoised signal by L sense value on each vertex")
hold off

%% Generating LSI Filters - step 1, normalized W:

tmp = fliplr(vander(diag(LAMBDA_WN)));
Van_WN = tmp(:, 1 : 3);

frequency_response = [0, 0, 0, 0, 0, 0, 1, 1]';

h_WN_si = pinv(Van_WN) * frequency_response;

F_WN_lsi = h_WN_si(1) * LAMBDA_WN ^ 0 + h_WN_si(2) * LAMBDA_WN ^ 1 + h_WN_si(3) * LAMBDA_WN ^ 2;

x_denoised_WN_lsi = eV * F_WN_lsi * x_noisy_hat_WN;

figure;
stem(1 : 8, diag(F_WN_lsi))
title("LPF response in spectral domain in WN sense")

figure;
subplot(3, 1, 1)
stem(1 : 8, x_noisy)
title("Noisy signal value on each vertex")
subplot(3, 1, 2)
stem(1 : 8, x)
title("Original signal value on each vertex")
subplot(3, 1, 3)
stem(1 : 8, x_denoised_WN_lsi)
title("Denoised signal by WN sense value on each vertex")
hold off

%% Computing SNR for filtered Signals!

% n_NF_ = x - x_noisy; %non filtered noisy x noise
% SNR_NF = 10 * log10(sum(x .^ 2) / sum(n_NF_ .^ 2));

n_F_L_lsi = x - x_denoised_L_lsi; %filtered signal in sense of laplacian
SNR_F_L_lsi = 10 * log10(sum(x .^ 2) / sum(n_F_L_lsi .^ 2));

n_F_WN_lsi = x - x_denoised_WN_lsi; %filtered signal in sense of laplacian
SNR_F_WN_lsi = 10 * log10(sum(x .^ 2) / sum(n_F_WN_lsi .^ 2));

%% plotting in spectral Domain! - laplacian sense filter

figure;
subplot(2, 1, 1)
stem(1 : 8, h_L)
title("Ideal Shift variant Low pass filter (based on laplacian matrix)")

subplot(2, 1, 2)
stem(1 : 8, diag(F_L_lsi))
title("Shift invariant Low pass filter (based on laplacian matrix)")

%% plotting in spectral domain! - WN sense filter

figure;
subplot(2, 1, 1)
stem(1 : 8, h_WN)
title("Ideal Shift variant Low pass filter (based on normalized weight matrix)")

subplot(2, 1, 2)
stem(1 : 8, diag(F_WN_lsi))
title("Shift invariant Low pass filter (based on normalized weight matrix)")

%% Generating LSI Filters - m = 2, laplacian:

tmp = fliplr(vander(diag(LAMBDA)));
Van = tmp(:, 1 : 2);

frequency_response = [1, 1, 0, 0, 0, 0, 0, 0]';

h_l_si = pinv(Van) * frequency_response;

F_L_lsi = h_l_si(1) * LAMBDA ^ 0 + h_l_si(2) * LAMBDA ^ 1;

x_denoised_L_lsi = eV * F_L_lsi * x_noisy_hat_L;

figure;
stem(1 : 8, diag(F_L_lsi))
title("LPF response in spectral domain in L sense")

figure;
subplot(3, 1, 1)
stem(1 : 8, x_noisy)
title("Noisy signal value on each vertex")
subplot(3, 1, 2)
stem(1 : 8, x)
title("Original signal value on each vertex")
subplot(3, 1, 3)
stem(1 : 8, x_denoised_L_lsi)
title("Denoised signal by L sense value on each vertex")
hold off

n_F_L_lsi = x - x_denoised_L_lsi; %filtered signal in sense of laplacian
SNR_F_L_lsi = 10 * log10(sum(x .^ 2) / sum(n_F_L_lsi .^ 2));

%% Generating LSI Filters - m = 5, laplacian:

tmp = fliplr(vander(diag(LAMBDA)));
Van = tmp(:, 1 : 5);

frequency_response = [1, 1, 0, 0, 0, 0, 0, 0]';

h_l_si = pinv(Van) * frequency_response;

F_L_lsi = h_l_si(1) * LAMBDA ^ 0 + h_l_si(2) * LAMBDA ^ 1 + h_l_si(3) * LAMBDA ^ 2 + h_l_si(4) * LAMBDA ^ 3 + h_l_si(5) * LAMBDA ^ 4;

x_denoised_L_lsi = eV * F_L_lsi * x_noisy_hat_L;

figure;
stem(1 : 8, diag(F_L_lsi))
title("LPF response in spectral domain in L sense")

figure;
subplot(3, 1, 1)
stem(1 : 8, x_noisy)
title("Noisy signal value on each vertex")
subplot(3, 1, 2)
stem(1 : 8, x)
title("Original signal value on each vertex")
subplot(3, 1, 3)
stem(1 : 8, x_denoised_L_lsi)
title("Denoised signal by L sense value on each vertex")
hold off

n_F_L_lsi = x - x_denoised_L_lsi; %filtered signal in sense of laplacian
SNR_F_L_lsi = 10 * log10(sum(x .^ 2) / sum(n_F_L_lsi .^ 2));

%% Generating LSI Filters - m = 8, laplacian:

tmp = fliplr(vander(diag(LAMBDA)));
Van = tmp(:, 1 : 8);

frequency_response = [1, 1, 0, 0, 0, 0, 0, 0]';

h_l_si = pinv(Van) * frequency_response;

F_L_lsi = h_l_si(1) * LAMBDA ^ 0 + h_l_si(2) * LAMBDA ^ 1 + h_l_si(3) * LAMBDA ^ 2 + h_l_si(4) * LAMBDA ^ 3 + h_l_si(5) * LAMBDA ^ 4 + h_l_si(6) * LAMBDA ^ 5 + h_l_si(7) * LAMBDA ^ 6 + h_l_si(8) * LAMBDA ^ 7;

x_denoised_L_lsi = eV * F_L_lsi * x_noisy_hat_L;

figure;
stem(1 : 8, diag(F_L_lsi))
title("LPF response in spectral domain in L sense")

figure;
subplot(3, 1, 1)
stem(1 : 8, x_noisy)
title("Noisy signal value on each vertex")
subplot(3, 1, 2)
stem(1 : 8, x)
title("Original signal value on each vertex")
subplot(3, 1, 3)
stem(1 : 8, x_denoised_L_lsi)
title("Denoised signal by L sense value on each vertex")
hold off

n_F_L_lsi = x - x_denoised_L_lsi; %filtered signal in sense of laplacian
SNR_F_L_lsi = 10 * log10(sum(x .^ 2) / sum(n_F_L_lsi .^ 2));