function vector = JRM(image, quant_table)
% July 2014: Corrected a mistake discovered by Yi Zhang, feature sets Ax_T5
% and Ax_T5_ref were always zeros due to use of incorrect function.
% -------------------------------------------------------------------------
% Copyright (c) 2011 DDE Lab, Binghamton University, NY.
% All Rights Reserved.
% -------------------------------------------------------------------------
% Permission to use, copy, modify, and distribute this software for
% educational, research and non-profit purposes, without fee, and without a
% written agreement is hereby granted, provided that this copyright notice
% appears in all copies. The program is supplied "as is," without any
% accompanying services from DDE Lab. DDE Lab does not warrant the
% operation of the program will be uninterrupted or error-free. The
% end-user understands that the program was developed for research purposes
% and is advised not to rely exclusively on the program for any reason. In
% no event shall Binghamton University or DDE Lab be liable to any party
% for direct, indirect, special, incidental, or consequential damages,
% including lost profits, arising out of the use of this software. DDE Lab
% disclaims any warranties, and has no obligations to provide maintenance,
% support, updates, enhancements or modifications.
% -------------------------------------------------------------------------
% Contact: jan@kodovsky.com | fridrich@binghamton.edu | November 2011
%          http://dde.binghamton.edu/download/feature_extractors
% -------------------------------------------------------------------------
% Extracts ALL submodels for the rich image model for DCT domain
% steganalysis [1] from the given IMAGE. The output is stored in a
% structured variable 'f'. The first half of the features are
% non-calibrated features. The second half are the reference values that
% are extracted from the reference image (decompressed, cropped by 4x4,
% re-compressed with JPEG quality factor QF). In [1], we used QF=75 as both
% the original and the reference quality factor.  Total dimensionality of
% the features: 22510.
% -------------------------------------------------------------------------
% Input:  FILE ... path to the JPEG image
%         QF ..... JPEG quality factor of the reference image (we
%                   recommend to keep it the same as the original image QF)
% Output: F ...... extracted features in a structured format
% -------------------------------------------------------------------------
% Note: Extracted features follow the structure (and logic) of Figure 1 in
% [1] and are well commented in the code (lines 41-254).
% -------------------------------------------------------------------------
% IMPORTANT: For accessing DCT coefficients of JPEG images, we use Phil
% Sallee's Matlab JPEG toolbox (function jpeg_read) available at
% http://philsallee.com/jpegtbx/
% -------------------------------------------------------------------------
% [1] Steganalysis of JPEG Images Using Rich Models, J. Kodovsky and J.
% Fridrich, to appear in Proc. SPIE, Electronic Imaging, Media
% Watermarking, Security, and Forensics XIV, San Francisco, CA, January
% 23–25, 2012.
% -------------------------------------------------------------------------
% JPEG_STRUCT = jpeg_read(IMAGE);
% quant_table =  JPEG_STRUCT.quant_tables{1};
QF = QFfromQMatrix(quant_table);

X = image; absX = abs(X);
[M,N] = size(X); M = floor(M/8)*8; N = floor(N/8)*8;
absXDh = absX(:,1:N-8)-absX(:,2:N-7);
absXDv = absX(1:M-8,:)-absX(2:M-7,:);
absXDd = absX(1:M-8,1:N-8)-absX(2:M-7,2:N-7);
absXDih = absX(:,1:N-8)-absX(:,9:N);
absXDiv = absX(1:M-8,:)-absX(9:M,:);

% DCT-mode specific co-occurrences of absolute values
f.Ah_T3  = extract_submodels_abs(absX,1:20,3,'hor'); % hor/ver neighbouring pairs | -
f.Ad_T3  = [extract_submodels_abs(absX,[1:5 7:10 13:14],3,'diag');extract_submodels_abs(absX,[6:10 12:14 17],3,'semidiag')];  % diagonally and semidiagonally neighbouring pairs \
f.Aoh_T3 = extract_submodels_abs(absX,[1:4 6:9 11:13 15 16 18],3,'hor_skip'); % horizontally skip one
f.Ax_T3  = extract_submodels_abs(absX,[1:5 8:10 14],3,'sym_8x8');  % symmetric wrt diagonal
f.Aod_T3 = [extract_submodels_abs(absX,[1:4 7:9 13],3,'diag_skip');extract_submodels_abs(absX,[11:14 22 16 17 23 24],3,'semidiag_skip')]; % diagonally and semidiagonally skip one neighbouring pairs \
f.Am_T3  = extract_submodels_abs(absX,6:20,3,'horse'); % intra horse move
f.Aih_T3 = extract_submodels_abs(absX,1:20,3,'inter_hor'); % inter-block hor/ver
f.Aid_T3 = extract_submodels_abs(absX,[1:5 7:10 13 14],3,'inter_diag'); % inter-block diagonally
f.Ais_T3 = extract_submodels_abs(absX,[1:5 7:10 13 14],3,'inter_semidiag'); % inter-block semidiagonally
f.Aix_T3 = extract_submodels_abs(absX,1:20,3,'inter_symm'); % inter, neighbouring diag. symmetric

% DCT-mode specific co-occurrences of differences of absolute values (horizontal/vertical)
f.Dh1_T2  = extract_submodels(absXDh,absXDv,1:20,2,'hor'); % hor/ver neighbouring pairs | -
f.Dd1_T2  = [extract_submodels(absXDh,absXDv,[1:5 7:10 13:14],2,'diag');extract_submodels(absXDh,absXDv,[6:10 12:14 17],2,'semidiag')];  % diagonally and semidiagonally neighbouring pairs \
f.Doh1_T2 = extract_submodels(absXDh,absXDv,[1:4 6:9 11:13 15 16 18],2,'hor_skip'); % horizontally skip one
f.Dx1_T2  = extract_submodels(absXDh,absXDv,[1:5 8:10 14],2,'sym_8x8');  % symmetric wrt diagonal
f.Dod1_T2 = [extract_submodels(absXDh,absXDv,[1:4 7:9 13],2,'diag_skip');extract_submodels(absXDh,absXDv,[11:14 22 16 17 23 24],2,'semidiag_skip')];      % diagonally and semidiagonally skip one neighbouring pairs \
f.Dm1_T2  = extract_submodels(absXDh,absXDv,6:20,2,'horse'); % intra horse move
f.Dih1_T2 = extract_submodels(absXDh,absXDv,1:20,2,'inter_hor'); % inter-block hor/ver
f.Did1_T2 = extract_submodels(absXDh,absXDv,[1:5 7:10 13 14],2,'inter_diag'); % inter-block diagonally
f.Dis1_T2 = extract_submodels(absXDh,absXDv,[1:5 7:10 13 14],2,'inter_semidiag'); % inter-block semidiagonally
f.Dix1_T2 = extract_submodels(absXDh,absXDv,1:20,2,'inter_symm'); % inter, neighbouring diag. symmetric
% DCT-mode specific co-occurrences of differences of absolute values (diagonal)
f.Dh2_T2  = extract_submodels(absXDd,absXDd,1:20,2,'hor'); % hor/ver neighbouring pairs | -
f.Dd2_T2  = [extract_submodels(absXDd,absXDd,[1:5 7:10 13:14],2,'diag');extract_submodels(absXDd,absXDd,[6:10 12:14 17],2,'semidiag')];  % diagonally and semidiagonally neighbouring pairs \
f.Doh2_T2 = extract_submodels(absXDd,absXDd,[1:4 6:9 11:13 15 16 18],2,'hor_skip'); % horizontally skip one
f.Dx2_T2  = extract_submodels(absXDd,absXDd,[1:5 8:10 14],2,'sym_8x8');  % symmetric wrt diagonal
f.Dod2_T2 = [extract_submodels(absXDd,absXDd,[1:4 7:9 13],2,'diag_skip');extract_submodels(absXDd,absXDd,[11:14 22 16 17 23 24],2,'semidiag_skip')];      % diagonally and semidiagonally skip one neighbouring pairs \
f.Dm2_T2  = extract_submodels(absXDd,absXDd,6:20,2,'horse'); % intra horse move
f.Dih2_T2 = extract_submodels(absXDd,absXDd,1:20,2,'inter_hor'); % inter-block hor/ver
f.Did2_T2 = extract_submodels(absXDd,absXDd,[1:5 7:10 13 14],2,'inter_diag'); % inter-block diagonally
f.Dis2_T2 = extract_submodels(absXDd,absXDd,[1:5 7:10 13 14],2,'inter_semidiag'); % inter-block semidiagonally
f.Dix2_T2 = extract_submodels(absXDd,absXDd,1:20,2,'inter_symm'); % inter, neighbouring diag. symmetric
% DCT-mode specific co-occurrences of differences of absolute values (inter-block horizontal/vertical)
f.Dh3_T2  = extract_submodels(absXDih,absXDiv,1:20,2,'hor'); % hor/ver neighbouring pairs | -
f.Dd3_T2  = [extract_submodels(absXDih,absXDiv,[1:5 7:10 13:14],2,'diag');extract_submodels(absXDih,absXDiv,[6:10 12:14 17],2,'semidiag')];  % diagonally and semidiagonally neighbouring pairs \
f.Doh3_T2 = extract_submodels(absXDih,absXDiv,[1:4 6:9 11:13 15 16 18],2,'hor_skip'); % horizontally skip one
f.Dx3_T2  = extract_submodels(absXDih,absXDiv,[1:5 8:10 14],2,'sym_8x8');  % symmetric wrt diagonal
f.Dod3_T2 = [extract_submodels(absXDih,absXDiv,[1:4 7:9 13],2,'diag_skip');extract_submodels(absXDih,absXDiv,[11:14 22 16 17 23 24],2,'semidiag_skip')];      % diagonally and semidiagonally skip one neighbouring pairs \
f.Dm3_T2  = extract_submodels(absXDih,absXDiv,6:20,2,'horse'); % intra horse move
f.Dih3_T2 = extract_submodels(absXDih,absXDiv,1:20,2,'inter_hor'); % inter-block hor/ver
f.Did3_T2 = extract_submodels(absXDih,absXDiv,[1:5 7:10 13 14],2,'inter_diag'); % inter-block diagonally
f.Dis3_T2 = extract_submodels(absXDih,absXDiv,[1:5 7:10 13 14],2,'inter_semidiag'); % inter-block semidiagonally
f.Dix3_T2 = extract_submodels(absXDih,absXDiv,1:20,2,'inter_symm'); % inter, neighbouring diag. symmetric
% Integral co-occurrences from absolute values Ax
f.Ax_T5 = extract_specific_integral_features_abs(absX,5,'MXh');
f.Ax_T5 = [f.Ax_T5;extract_specific_integral_features_abs(absX,5,'MXd')];
f.Ax_T5 = [f.Ax_T5;extract_specific_integral_features_abs(absX,5,'MXs')];
f.Ax_T5 = [f.Ax_T5;extract_specific_integral_features_abs(absX,5,'MXih')];
f.Ax_T5 = [f.Ax_T5;extract_specific_integral_features_abs(absX,5,'MXid')];
% Integral co-occurrences from differences of abs. values
% DfH - frequency dependencies, HORIZONTAL difference
f.DfH_T5 = extract_specific_integral_features_diff2(absXDh,5,'MXh');
f.DfH_T5 = [f.DfH_T5;extract_specific_integral_features_diff2(absXDh,5,'MXv')];
f.DfH_T5 = [f.DfH_T5;extract_specific_integral_features_diff2(absXDh,5,'MXd')];
f.DfH_T5 = [f.DfH_T5;extract_specific_integral_features_diff2(absXDh,5,'MXs')];
% DsH - spatial dependencies, HORIZONTAL differences
f.DsH_T5 = extract_specific_integral_features_diff2(absXDh,5,'MXih');
f.DsH_T5 = [f.DsH_T5;extract_specific_integral_features_diff2(absXDh,5,'MXiv')];
f.DsH_T5 = [f.DsH_T5;extract_specific_integral_features_diff2(absXDh,5,'MXid')];
f.DsH_T5 = [f.DsH_T5;extract_specific_integral_features_diff2(absXDh,5,'MXis')];
% DfV - frequency dependencies, VERTICAL differences
f.DfV_T5  = extract_specific_integral_features_diff2(absXDv,5,'MXh');
f.DfV_T5  = [f.DfV_T5;extract_specific_integral_features_diff2(absXDv,5,'MXv')];
f.DfV_T5  = [f.DfV_T5;extract_specific_integral_features_diff2(absXDv,5,'MXd')];
f.DfV_T5  = [f.DfV_T5;extract_specific_integral_features_diff2(absXDv,5,'MXs')];
% DsV - spatial dependencies, VERTICAL differences
f.DsV_T5 = extract_specific_integral_features_diff2(absXDv,5,'MXih');
f.DsV_T5 = [f.DsV_T5;extract_specific_integral_features_diff2(absXDv,5,'MXiv')];
f.DsV_T5 = [f.DsV_T5;extract_specific_integral_features_diff2(absXDv,5,'MXid')];
f.DsV_T5 = [f.DsV_T5;extract_specific_integral_features_diff2(absXDv,5,'MXis')];
% DfD - frequency dependencies, DIAGONAL differences
f.DfD_T5  = extract_specific_integral_features_diff2(absXDd,5,'MXh');
f.DfD_T5  = [f.DfD_T5;extract_specific_integral_features_diff2(absXDd,5,'MXv')];
f.DfD_T5  = [f.DfD_T5;extract_specific_integral_features_diff2(absXDd,5,'MXd')];
f.DfD_T5  = [f.DfD_T5;extract_specific_integral_features_diff2(absXDd,5,'MXs')];
% DsD - spatial dependencies, DIAGONAL differences
f.DsD_T5 = extract_specific_integral_features_diff2(absXDd,5,'MXih');
f.DsD_T5 = [f.DsD_T5;extract_specific_integral_features_diff2(absXDd,5,'MXiv')];
f.DsD_T5 = [f.DsD_T5;extract_specific_integral_features_diff2(absXDd,5,'MXid')];
f.DsD_T5 = [f.DsD_T5;extract_specific_integral_features_diff2(absXDd,5,'MXis')];
% DfIH - frequency dependencies, INTER-BLOCK HORIZONTAL differences
f.DfIH_T5  = extract_specific_integral_features_diff2(absXDih,5,'MXh');
f.DfIH_T5  = [f.DfIH_T5;extract_specific_integral_features_diff2(absXDih,5,'MXv')];
f.DfIH_T5  = [f.DfIH_T5;extract_specific_integral_features_diff2(absXDih,5,'MXd')];
f.DfIH_T5  = [f.DfIH_T5;extract_specific_integral_features_diff2(absXDih,5,'MXs')];
% DsIH - spatial dependencies, INTER-BLOCK HORIZONTAL differences
f.DsIH_T5 = extract_specific_integral_features_diff2(absXDih,5,'MXih');
f.DsIH_T5 = [f.DsIH_T5;extract_specific_integral_features_diff2(absXDih,5,'MXiv')];
f.DsIH_T5 = [f.DsIH_T5;extract_specific_integral_features_diff2(absXDih,5,'MXid')];
f.DsIH_T5 = [f.DsIH_T5;extract_specific_integral_features_diff2(absXDih,5,'MXis')];
% DfIV - frequency dependencies, INTER-BLOCK VERTICAL differences
f.DfIV_T5  = extract_specific_integral_features_diff2(absXDiv,5,'MXh');
f.DfIV_T5  = [f.DfIV_T5;extract_specific_integral_features_diff2(absXDiv,5,'MXv')];
f.DfIV_T5  = [f.DfIV_T5;extract_specific_integral_features_diff2(absXDiv,5,'MXd')];
f.DfIV_T5  = [f.DfIV_T5;extract_specific_integral_features_diff2(absXDiv,5,'MXs')];
% DsIV - spatial dependencies, INTER-BLOCK VERTICAL differences
f.DsIV_T5 = extract_specific_integral_features_diff2(absXDiv,5,'MXih');
f.DsIV_T5 = [f.DsIV_T5;extract_specific_integral_features_diff2(absXDiv,5,'MXiv')];
f.DsIV_T5 = [f.DsIV_T5;extract_specific_integral_features_diff2(absXDiv,5,'MXid')];
f.DsIV_T5 = [f.DsIV_T5;extract_specific_integral_features_diff2(absXDiv,5,'MXis')];

% % reference version of the features (for calibration)
% X = DCTPlane_reference(IMAGE,QF); absX = abs(X);
% [M,N] = size(X); M = floor(M/8)*8; N = floor(N/8)*8;
% absXDh = absX(:,1:N-8)-absX(:,2:N-7);
% absXDv = absX(1:M-8,:)-absX(2:M-7,:);
% absXDd = absX(1:M-8,1:N-8)-absX(2:M-7,2:N-7);
% absXDih = absX(:,1:N-8)-absX(:,9:N);
% absXDiv = absX(1:M-8,:)-absX(9:M,:);
% 
% % DCT-mode specific co-occurrences of absolute values
% f.Ah_T3_ref  = extract_submodels_abs(absX,1:20,3,'hor'); % hor/ver neighbouring pairs | -
% f.Ad_T3_ref  = [extract_submodels_abs(absX,[1:5 7:10 13:14],3,'diag');extract_submodels_abs(absX,[6:10 12:14 17],3,'semidiag')];  % diagonally and semidiagonally neighbouring pairs \
% f.Aoh_T3_ref = extract_submodels_abs(absX,[1:4 6:9 11:13 15 16 18],3,'hor_skip'); % horizontally skip one
% f.Ax_T3_ref  = extract_submodels_abs(absX,[1:5 8:10 14],3,'sym_8x8');  % symmetric wrt diagonal
% f.Aod_T3_ref = [extract_submodels_abs(absX,[1:4 7:9 13],3,'diag_skip');extract_submodels_abs(absX,[11:14 22 16 17 23 24],3,'semidiag_skip')]; % diagonally and semidiagonally skip one neighbouring pairs \
% f.Am_T3_ref  = extract_submodels_abs(absX,6:20,3,'horse'); % intra horse move
% f.Aih_T3_ref = extract_submodels_abs(absX,1:20,3,'inter_hor'); % inter-block hor/ver
% f.Aid_T3_ref = extract_submodels_abs(absX,[1:5 7:10 13 14],3,'inter_diag'); % inter-block diagonally
% f.Ais_T3_ref = extract_submodels_abs(absX,[1:5 7:10 13 14],3,'inter_semidiag'); % inter-block semidiagonally
% f.Aix_T3_ref = extract_submodels_abs(absX,1:20,3,'inter_symm'); % inter, neighbouring diag. symmetric
% 
% % DCT-mode specific co-occurrences of differences of absolute values (horizontal/vertical)
% f.Dh1_T2_ref  = extract_submodels(absXDh,absXDv,1:20,2,'hor'); % hor/ver neighbouring pairs | -
% f.Dd1_T2_ref  = [extract_submodels(absXDh,absXDv,[1:5 7:10 13:14],2,'diag');extract_submodels(absXDh,absXDv,[6:10 12:14 17],2,'semidiag')];  % diagonally and semidiagonally neighbouring pairs \
% f.Doh1_T2_ref = extract_submodels(absXDh,absXDv,[1:4 6:9 11:13 15 16 18],2,'hor_skip'); % horizontally skip one
% f.Dx1_T2_ref  = extract_submodels(absXDh,absXDv,[1:5 8:10 14],2,'sym_8x8');  % symmetric wrt diagonal
% f.Dod1_T2_ref = [extract_submodels(absXDh,absXDv,[1:4 7:9 13],2,'diag_skip');extract_submodels(absXDh,absXDv,[11:14 22 16 17 23 24],2,'semidiag_skip')];      % diagonally and semidiagonally skip one neighbouring pairs \
% f.Dm1_T2_ref  = extract_submodels(absXDh,absXDv,6:20,2,'horse'); % intra horse move
% f.Dih1_T2_ref = extract_submodels(absXDh,absXDv,1:20,2,'inter_hor'); % inter-block hor/ver
% f.Did1_T2_ref = extract_submodels(absXDh,absXDv,[1:5 7:10 13 14],2,'inter_diag'); % inter-block diagonally
% f.Dis1_T2_ref = extract_submodels(absXDh,absXDv,[1:5 7:10 13 14],2,'inter_semidiag'); % inter-block semidiagonally
% f.Dix1_T2_ref = extract_submodels(absXDh,absXDv,1:20,2,'inter_symm'); % inter, neighbouring diag. symmetric
% % DCT-mode specific co-occurrences of differences of absolute values (diagonal)
% f.Dh2_T2_ref  = extract_submodels(absXDd,absXDd,1:20,2,'hor'); % hor/ver neighbouring pairs | -
% f.Dd2_T2_ref  = [extract_submodels(absXDd,absXDd,[1:5 7:10 13:14],2,'diag');extract_submodels(absXDd,absXDd,[6:10 12:14 17],2,'semidiag')];  % diagonally and semidiagonally neighbouring pairs \
% f.Doh2_T2_ref = extract_submodels(absXDd,absXDd,[1:4 6:9 11:13 15 16 18],2,'hor_skip'); % horizontally skip one
% f.Dx2_T2_ref  = extract_submodels(absXDd,absXDd,[1:5 8:10 14],2,'sym_8x8');  % symmetric wrt diagonal
% f.Dod2_T2_ref = [extract_submodels(absXDd,absXDd,[1:4 7:9 13],2,'diag_skip');extract_submodels(absXDd,absXDd,[11:14 22 16 17 23 24],2,'semidiag_skip')];      % diagonally and semidiagonally skip one neighbouring pairs \
% f.Dm2_T2_ref  = extract_submodels(absXDd,absXDd,6:20,2,'horse'); % intra horse move
% f.Dih2_T2_ref = extract_submodels(absXDd,absXDd,1:20,2,'inter_hor'); % inter-block hor/ver
% f.Did2_T2_ref = extract_submodels(absXDd,absXDd,[1:5 7:10 13 14],2,'inter_diag'); % inter-block diagonally
% f.Dis2_T2_ref = extract_submodels(absXDd,absXDd,[1:5 7:10 13 14],2,'inter_semidiag'); % inter-block semidiagonally
% f.Dix2_T2_ref = extract_submodels(absXDd,absXDd,1:20,2,'inter_symm'); % inter, neighbouring diag. symmetric
% % DCT-mode specific co-occurrences of differences of absolute values (inter-block horizontal/vertical)
% f.Dh3_T2_ref  = extract_submodels(absXDih,absXDiv,1:20,2,'hor'); % hor/ver neighbouring pairs | -
% f.Dd3_T2_ref  = [extract_submodels(absXDih,absXDiv,[1:5 7:10 13:14],2,'diag');extract_submodels(absXDih,absXDiv,[6:10 12:14 17],2,'semidiag')];  % diagonally and semidiagonally neighbouring pairs \
% f.Doh3_T2_ref = extract_submodels(absXDih,absXDiv,[1:4 6:9 11:13 15 16 18],2,'hor_skip'); % horizontally skip one
% f.Dx3_T2_ref  = extract_submodels(absXDih,absXDiv,[1:5 8:10 14],2,'sym_8x8');  % symmetric wrt diagonal
% f.Dod3_T2_ref = [extract_submodels(absXDih,absXDiv,[1:4 7:9 13],2,'diag_skip');extract_submodels(absXDih,absXDiv,[11:14 22 16 17 23 24],2,'semidiag_skip')];      % diagonally and semidiagonally skip one neighbouring pairs \
% f.Dm3_T2_ref  = extract_submodels(absXDih,absXDiv,6:20,2,'horse'); % intra horse move
% f.Dih3_T2_ref = extract_submodels(absXDih,absXDiv,1:20,2,'inter_hor'); % inter-block hor/ver
% f.Did3_T2_ref = extract_submodels(absXDih,absXDiv,[1:5 7:10 13 14],2,'inter_diag'); % inter-block diagonally
% f.Dis3_T2_ref = extract_submodels(absXDih,absXDiv,[1:5 7:10 13 14],2,'inter_semidiag'); % inter-block semidiagonally
% f.Dix3_T2_ref = extract_submodels(absXDih,absXDiv,1:20,2,'inter_symm'); % inter, neighbouring diag. symmetric
% % Integral co-occurrences from absolute values Ax
% f.Ax_T5_ref = extract_specific_integral_features_abs(absX,5,'MXh');
% f.Ax_T5_ref = [f.Ax_T5_ref;extract_specific_integral_features_abs(absX,5,'MXd')];
% f.Ax_T5_ref = [f.Ax_T5_ref;extract_specific_integral_features_abs(absX,5,'MXs')];
% f.Ax_T5_ref = [f.Ax_T5_ref;extract_specific_integral_features_abs(absX,5,'MXih')];
% f.Ax_T5_ref = [f.Ax_T5_ref;extract_specific_integral_features_abs(absX,5,'MXid')];
% % Integral co-occurrences from differences of abs. values
% % DfH - frequency dependencies, HORIZONTAL difference
% f.DfH_T5_ref = extract_specific_integral_features_diff2(absXDh,5,'MXh');
% f.DfH_T5_ref = [f.DfH_T5_ref;extract_specific_integral_features_diff2(absXDh,5,'MXv')];
% f.DfH_T5_ref = [f.DfH_T5_ref;extract_specific_integral_features_diff2(absXDh,5,'MXd')];
% f.DfH_T5_ref = [f.DfH_T5_ref;extract_specific_integral_features_diff2(absXDh,5,'MXs')];
% % DsH - spatial dependencies, HORIZONTAL differences
% f.DsH_T5_ref = extract_specific_integral_features_diff2(absXDh,5,'MXih');
% f.DsH_T5_ref = [f.DsH_T5_ref;extract_specific_integral_features_diff2(absXDh,5,'MXiv')];
% f.DsH_T5_ref = [f.DsH_T5_ref;extract_specific_integral_features_diff2(absXDh,5,'MXid')];
% f.DsH_T5_ref = [f.DsH_T5_ref;extract_specific_integral_features_diff2(absXDh,5,'MXis')];
% % DfV - frequency dependencies, VERTICAL differences
% f.DfV_T5_ref  = extract_specific_integral_features_diff2(absXDv,5,'MXh');
% f.DfV_T5_ref  = [f.DfV_T5_ref;extract_specific_integral_features_diff2(absXDv,5,'MXv')];
% f.DfV_T5_ref  = [f.DfV_T5_ref;extract_specific_integral_features_diff2(absXDv,5,'MXd')];
% f.DfV_T5_ref  = [f.DfV_T5_ref;extract_specific_integral_features_diff2(absXDv,5,'MXs')];
% % DsV - spatial dependencies, VERTICAL differences
% f.DsV_T5_ref = extract_specific_integral_features_diff2(absXDv,5,'MXih');
% f.DsV_T5_ref = [f.DsV_T5_ref;extract_specific_integral_features_diff2(absXDv,5,'MXiv')];
% f.DsV_T5_ref = [f.DsV_T5_ref;extract_specific_integral_features_diff2(absXDv,5,'MXid')];
% f.DsV_T5_ref = [f.DsV_T5_ref;extract_specific_integral_features_diff2(absXDv,5,'MXis')];
% % DfD - frequency dependencies, DIAGONAL differences
% f.DfD_T5_ref  = extract_specific_integral_features_diff2(absXDd,5,'MXh');
% f.DfD_T5_ref  = [f.DfD_T5_ref;extract_specific_integral_features_diff2(absXDd,5,'MXv')];
% f.DfD_T5_ref  = [f.DfD_T5_ref;extract_specific_integral_features_diff2(absXDd,5,'MXd')];
% f.DfD_T5_ref  = [f.DfD_T5_ref;extract_specific_integral_features_diff2(absXDd,5,'MXs')];
% % DsD - spatial dependencies, DIAGONAL differences
% f.DsD_T5_ref = extract_specific_integral_features_diff2(absXDd,5,'MXih');
% f.DsD_T5_ref = [f.DsD_T5_ref;extract_specific_integral_features_diff2(absXDd,5,'MXiv')];
% f.DsD_T5_ref = [f.DsD_T5_ref;extract_specific_integral_features_diff2(absXDd,5,'MXid')];
% f.DsD_T5_ref = [f.DsD_T5_ref;extract_specific_integral_features_diff2(absXDd,5,'MXis')];
% % DfIH - frequency dependencies, INTER-BLOCK HORIZONTAL differences
% f.DfIH_T5_ref  = extract_specific_integral_features_diff2(absXDih,5,'MXh');
% f.DfIH_T5_ref  = [f.DfIH_T5_ref;extract_specific_integral_features_diff2(absXDih,5,'MXv')];
% f.DfIH_T5_ref  = [f.DfIH_T5_ref;extract_specific_integral_features_diff2(absXDih,5,'MXd')];
% f.DfIH_T5_ref  = [f.DfIH_T5_ref;extract_specific_integral_features_diff2(absXDih,5,'MXs')];
% % DsIH - spatial dependencies, INTER-BLOCK HORIZONTAL differences
% f.DsIH_T5_ref = extract_specific_integral_features_diff2(absXDih,5,'MXih');
% f.DsIH_T5_ref = [f.DsIH_T5_ref;extract_specific_integral_features_diff2(absXDih,5,'MXiv')];
% f.DsIH_T5_ref = [f.DsIH_T5_ref;extract_specific_integral_features_diff2(absXDih,5,'MXid')];
% f.DsIH_T5_ref = [f.DsIH_T5_ref;extract_specific_integral_features_diff2(absXDih,5,'MXis')];
% % DfIV - frequency dependencies, INTER-BLOCK VERTICAL differences
% f.DfIV_T5_ref  = extract_specific_integral_features_diff2(absXDiv,5,'MXh');
% f.DfIV_T5_ref  = [f.DfIV_T5_ref;extract_specific_integral_features_diff2(absXDiv,5,'MXv')];
% f.DfIV_T5_ref  = [f.DfIV_T5_ref;extract_specific_integral_features_diff2(absXDiv,5,'MXd')];
% f.DfIV_T5_ref  = [f.DfIV_T5_ref;extract_specific_integral_features_diff2(absXDiv,5,'MXs')];
% % DsIV - spatial dependencies, INTER-BLOCK VERTICAL differences
% f.DsIV_T5_ref = extract_specific_integral_features_diff2(absXDiv,5,'MXih');
% f.DsIV_T5_ref = [f.DsIV_T5_ref;extract_specific_integral_features_diff2(absXDiv,5,'MXiv')];
% f.DsIV_T5_ref = [f.DsIV_T5_ref;extract_specific_integral_features_diff2(absXDiv,5,'MXid')];
% f.DsIV_T5_ref = [f.DsIV_T5_ref;extract_specific_integral_features_diff2(absXDiv,5,'MXis')];

fn = fieldnames(f);
vector = [];
for k=1:numel(fn)
    vector = [vector f.(fn{k})'];
end






function f = extract_specific_integral_features_diff2(X,T,type)
if strcmp(type,'MXh'),  [dx,dy,chck]=deal(0,1,0); end % horizontal
if strcmp(type,'MXv'),  [dx,dy,chck]=deal(1,0,0); end % vertical
if strcmp(type,'MXd'),  [dx,dy,chck]=deal(1,1,0); end % diagonal from DC
if strcmp(type,'MXs'),  [dx,dy,chck]=deal(-1,1,0);end % semi-diagonal
if strcmp(type,'MXih'), [dx,dy,chck]=deal(0,8,1); end % inter horizontal
if strcmp(type,'MXiv'), [dx,dy,chck]=deal(8,0,1); end % inter vertical
if strcmp(type,'MXid'), [dx,dy,chck]=deal(8,8,1); end % inter diagonal
if strcmp(type,'MXis'), [dx,dy,chck]=deal(-8,8,1);end % inter semidiagonal
if chck
    f = normalize(sign_symmetrize_with_normalization(extractCooccurencesFromColumns(ExtractCoocColumns_integral_inter(X,dx,dy),T)));
else
    f = normalize(sign_symmetrize_with_normalization(extractCooccurencesFromColumns(ExtractCoocColumns_integral(X,dx,dy),T)));
end
function f = extract_specific_integral_features_abs(A,T,type)

if strcmp(type,'MXh')
    % horizontal left->right + vertical top->bottom
    c = ExtractCoocColumns_integral(A,0,1); f = extractCooccurencesFromColumns_abs(c,T);
    c = ExtractCoocColumns_integral(A,1,0); f = f+extractCooccurencesFromColumns_abs(c,T);
    f = normalize(reshape(f,[],1));
    return;
end
if strcmp(type,'MXd')
    % diagonal from DC
    f = normalize(reshape(extractCooccurencesFromColumns_abs(ExtractCoocColumns_integral(A,+1,+1),T),[],1));
    return;
end
if strcmp(type,'MXs')
    % semi-diagonal both directions
    c = ExtractCoocColumns_integral(A,-1,+1); f = extractCooccurencesFromColumns_abs(c,T);
    c = ExtractCoocColumns_integral(A,+1,-1); f = f+extractCooccurencesFromColumns_abs(c,T);
    f = normalize(reshape(f,[],1));
    return;
end
if strcmp(type,'MXih')
    % inter horizontal/vertical
    c = ExtractCoocColumns_integral_inter(A,0,8); f = extractCooccurencesFromColumns_abs(c,T);
    c = ExtractCoocColumns_integral_inter(A,8,0); f = f+extractCooccurencesFromColumns_abs(c,T);
    f = normalize(reshape(f,[],1));
    return;
end
if strcmp(type,'MXid')
    % inter diagonal/semidiagonal
    c = ExtractCoocColumns_integral_inter(A,8,-8); f = extractCooccurencesFromColumns_abs(c,T);
    c = ExtractCoocColumns_integral_inter(A,8,+8); f = f+extractCooccurencesFromColumns_abs(c,T);
    f = normalize(reshape(f,[],1));
    return;
end

function f = extract_submodels_abs(X,IDs,T,type)
f = [];
for ID=IDs
    % interpret cooccurence ID as a list of DCT modes of interest
    [a,b] = ind2sub([6 6],find([0 1 2 3 4 5;6 7 8 9 10 21;11 12 13 14 22 26;15 16 17 23 27 30;18 19 24 28 31 33;20 25 29 32 34 35]==ID));
    if strcmp(type,'sym_8x8'), target = [a,b;b,a]; end
    if strcmp(type,'inter_semidiag'), target = [a+8,b;a,b+8]; end
    if strcmp(type,'inter_symm'), target = [a,b;b,a+8]; end
    if strcmp(type,'inter_hor'), target = [a,b;a,b+8]; end
    if strcmp(type,'inter_diag'), target = [a,b;a+8,b+8]; end
    if strcmp(type,'hor'), target = [a,b;a,b+1]; end
    if strcmp(type,'diag'), target = [a,b;a+1,b+1]; end
    if strcmp(type,'semidiag'), target = [a,b;a-1,b+1]; end
    if strcmp(type,'hor_skip'), target = [a,b;a,b+2]; end
    if strcmp(type,'diag_skip'), target = [a,b;a+2,b+2]; end
    if strcmp(type,'semidiag_skip'), target = [a,b;a-2,b+2]; end
    if strcmp(type,'horse'), target = [a,b;a-1,b+2]; end
    
    % extract the cooccurence features and 8x8 diag. symmetric features
    f1 = extractCooccurencesFromColumns_abs(ExtractCoocColumns(X,target),T);
    f2 = extractCooccurencesFromColumns_abs(ExtractCoocColumns(X,target(:,[2,1])),T);
    [f1,f2] = deal(normalize(f1),normalize(f2));
    f = [f;(f1(:)+f2(:))/2]; %#ok<*AGROW>
end
function F = extract_submodels(X1,X2,IDs,T,type)
F = [];
for ID=IDs
    % interpret cooccurence ID as a list of DCT modes of interest
    [a,b] = ind2sub([6 6],find([0 1 2 3 4 5;6 7 8 9 10 21;11 12 13 14 22 26;15 16 17 23 27 30;18 19 24 28 31 33;20 25 29 32 34 35]==ID));
    if strcmp(type,'sym_8x8'), target = [a,b;b,a]; end
    if strcmp(type,'inter_semidiag'), target = [a+8,b;a,b+8]; end
    if strcmp(type,'inter_symm'), target = [a,b;b,a+8]; end
    if strcmp(type,'inter_hor'), target = [a,b;a,b+8]; end
    if strcmp(type,'inter_diag'), target = [a,b;a+8,b+8]; end
    if strcmp(type,'hor'), target = [a,b;a,b+1]; end
    if strcmp(type,'diag'), target = [a,b;a+1,b+1]; end
    if strcmp(type,'semidiag'), target = [a,b;a-1,b+1]; end
    if strcmp(type,'hor_skip'), target = [a,b;a,b+2]; end
    if strcmp(type,'diag_skip'), target = [a,b;a+2,b+2]; end
    if strcmp(type,'semidiag_skip'), target = [a,b;a-2,b+2]; end
    if strcmp(type,'horse'), target = [a,b;a-1,b+2]; end
    % extract the cooccurence features
    columns = ExtractCoocColumns(X1,target);
    f1 = normalize(extractCooccurencesFromColumns(columns,T));
    % extract 8x8 diagonally symmetric cooccurence features
    columns = ExtractCoocColumns(X2,target(:,[2,1]));
    f2 = normalize(extractCooccurencesFromColumns(columns,T));
    % sign symmetrize
    f = normalize(sign_symmetrize_with_normalization(f1+f2));
    F = [F;f]; %#ok<AGROW>
end
function Plane=DCTPlane(path)
% loads DCT Plane of the given JPEG image + Quantization table
jobj=jpeg_read(path);
Plane=jobj.coef_arrays{1};
function Plane=DCTPlane_reference(path,QF)
I = imread(path);      % decompress into spatial domain
I = I(5:end-4,5:end-4); % crop by 4x4 pixels

TMP = ['img_' num2str(round(rand()*1e7)) num2str(round(rand()*1e7)) '.jpg'];
while exist(TMP,'file'), TMP = ['img_' num2str(round(rand()*1e7)) num2str(round(rand()*1e7)) '.jpg']; end
imwrite(I,TMP,'Quality',QF); % save as temporary jpeg image using imwrite
Plane = DCTPlane(TMP); % load DCT plane of the reference image
delete(TMP); % delete the temporary reference image
function columns = ExtractCoocColumns(A,target)
% Take the target DCT modes and extracts their corresponding n-tuples from
% the DCT plane A. Store them as individual columns.
mask = getMask(target);
v = floor(size(A,1)/8)+1-(size(mask,1)/8); % number of vertical block shifts
h = floor(size(A,2)/8)+1-(size(mask,2)/8); % number of horizontal block shifts

for i=1:size(target,1)
    C = A(target(i,1)+(1:8:8*v)-1,target(i,2)+(1:8:8*h)-1);
    if ~exist('columns','var'),columns = zeros(numel(C),size(target,1)); end
    columns(:,i) = C(:);
end
function F = extractCooccurencesFromColumns(blocks,t)
% blocks = columns of values from which we want to extract the
% cooccurences. marginalize to [-t..t]. no normalization involved

order = size(blocks,2); % cooccurence order
blocks(blocks<-t) = -t; % marginalization
blocks(blocks>t) = t;   % marginalization

switch order
    case 1
        % 1st order cooccurence (histogram)
        F = zeros(2*t+1,1);
        % for loop is faster than hist function
        for i=-t:t
            F(i+t+1) = sum(blocks==i);
        end
    case 2
        % 2nd order cooccurence
        F = zeros(2*t+1,2*t+1);
        for i=-t:t
            fB = blocks(blocks(:,1)==i,2);
            if ~isempty(fB)
                for j=-t:t
                    F(i+t+1,j+t+1) = sum(fB==j);
                end
            end
        end
    case 3
        % 3rd order cooccurence
        F = zeros(2*t+1,2*t+1,2*t+1);
        for i=-t:t
            fB = blocks(blocks(:,1)==i,2:end);
            if ~isempty(fB)
                for j=-t:t
                    fB2 = fB(fB(:,1)==j,2:end);
                    if ~isempty(fB2)
                        for k=-t:t
                            F(i+t+1,j+t+1,k+t+1) = sum(fB2==k);
                        end
                    end
                end
            end
        end
    case 4
        % 4th order cooccurence
        F = zeros(2*t+1,2*t+1,2*t+1,2*t+1);
        for i=-t:t
            fB = blocks(blocks(:,1)==i,2:end);
            if ~isempty(fB)
                for j=-t:t
                    fB2 = fB(fB(:,1)==j,2:end);
                    if ~isempty(fB2)
                        for k=-t:t
                            fB3 = fB2(fB2(:,1)==k,2:end);
                            if ~isempty(fB3)
                                for l=-t:t
                                    F(i+t+1,j+t+1,k+t+1,l+t+1) = sum(fB3==l);
                                end
                            end
                        end
                    end
                end
            end
        end
end % switch
function F = extractCooccurencesFromColumns_abs(blocks,t)
% blocks = columns of values from which we want to extract the
% cooccurences. marginalize to [0..t]. no normalization involved
blocks(blocks>t) = t;   % marginalization
% 2nd order cooccurence
F = zeros(t+1,t+1);
for i=0:t
    fB = blocks(blocks(:,1)==i,2);
    if ~isempty(fB)
        for j=0:t
            F(i+1,j+1) = sum(fB==j);
        end
    end
end
function mask = getMask(target)
% transform list of DCT modes of interest into a mask with all zeros and
% ones at the positions of those DCT modes of interest
x=8;y=8;
if sum(target(:,1)>8)>0 && sum(target(:,1)>16)==0, x=16; end
if sum(target(:,1)>16)>0 && sum(target(:,1)>24)==0, x=24; end
if sum(target(:,1)>24)>0 && sum(target(:,1)>32)==0, x=32; end
if sum(target(:,2)>8)>0 && sum(target(:,2)>16)==0, y=16; end
if sum(target(:,2)>16)>0 && sum(target(:,2)>24)==0, y=24; end
if sum(target(:,2)>24)>0 && sum(target(:,2)>32)==0, y=32; end

mask = zeros(x,y);
for i=1:size(target,1)
    mask(target(i,1),target(i,2)) = 1;
end
function f = sign_symmetrize_with_normalization(f)
% symmetrize 2D cooccurence matrix by sign [x,y,z] = [-x,-y,-z] and output
% as a 1D vector (final form of the feature vector)
sD = size(f,1); T = (sD-1)/2;
if T==2,
    ID1 = [ 1     6    11    16    21     2     7    12    17    22     3     8 ];
    ID2 = [25    20    15    10     5    24    19    14     9     4    23    18 ];
elseif T==5
    ID1 = [1 12 23 34 45 56 67 78 89 100 111 2 13 24 35 46 57 68 79 90 101 112 3 14 25 36 47 58 69 80 91 102 113 4 15 26 37 48 59 70 81 92 103 114 5 16 27 38 49 60 71 82 93 104 115 6 17 28 39 50];
    ID2 = [121 110 99 88 77 66 55 44 33 22 11 120 109 98 87 76 65 54 43 32 21 10 119 108 97 86 75 64 53 42 31 20 9 118 107 96 85 74 63 52 41 30 19 8 117 106 95 84 73 62 51 40 29 18 7 116 105 94 83 72];
else
    error('TODO');
%     REF = reshape(1:numel(f),sD,sD);
%     DONE = zeros(size(REF));
%     [ID1,ID2] = deal(zeros((numel(f)-1)/2,1));
%     id = 0;
%     for i=1:sD
%         for j=1:sD
%             if ~DONE(i,j)
%                 id = id+1;
%                 ID1(id) = REF(i,j);
%                 ID2(id) = REF(sD-i+1,sD-j+1);
%                 DONE(i,j)=1;
%                 DONE(sD-i+1,sD-j+1)=1;
%             end
%         end
%     end
end

% symmetrize
f = f(:);
f(ID1) = (f(ID1)+f(ID2))/2;
f(ID2) = [];

function columns = ExtractCoocColumns_integral_inter(A,dx,dy)
CONST = sqrt(2);
A1=A; A2=A;
A1(1:8:end,1:8:end) = CONST;
A2(1:8:end,1:8:end) = CONST;

if dx>0
    A1(end-dx+1:end,:) = [];
    A2(1:dx,:) = [];
elseif dx<0
    A2(end+dx+1:end,:) = [];
    A1(1:-dx,:) = [];
end

if dy>0
    A1(:,end-dy+1:end) = [];
    A2(:,1:dy) = [];
elseif dy<0
    A2(:,end+dy+1:end) = [];
    A1(:,1:-dy) = [];
end

columns = [A1(:) A2(:)];
columns(columns(:,1)==CONST|columns(:,2)==CONST,:) = [];
function columns = ExtractCoocColumns_integral(A,dx,dy)
CONST = sqrt(2);
A1=A; A2=A;
A1(1:8:end,1:8:end) = CONST;
A2(1:8:end,1:8:end) = CONST;

for i=1:size(A,1)
    x1=i;x2=x1+dx;
    if x1>=1 && x1<=size(A,1) && x2>=1 && x2<=size(A,1) && ceil(x1/8)~=ceil(x2/8)
        A1(x1,:) = CONST;
        A2(x2,:) = CONST;
    end
end
for j=1:size(A,2)
    y1=j;y2=y1+dy;
    if y1>=1 && y1<=size(A,2) && y2>=1 && y2<=size(A,2) && ceil(y1/8)~=ceil(y2/8)
        A1(:,y1) = CONST;
        A2(:,y2) = CONST;
    end
end

if dx>0
    A1(end-dx+1:end,:) = [];
    A2(1:dx,:) = [];
elseif dx<0
    A2(end+dx+1:end,:) = [];
    A1(1:-dx,:) = [];
end

if dy>0
    A1(:,end-dy+1:end) = [];
    A2(:,1:dy) = [];
elseif dy<0
    A2(:,end+dy+1:end) = [];
    A1(:,1:-dy) = [];
end
columns = [A1(:) A2(:)];
columns(columns(:,1)==CONST|columns(:,2)==CONST,:) = [];
function f = normalize(f)
S = sum(f(:));
if S~=0, f=f/S; end