function Q=Qmatrix(quality)
% Q=Qmatrix(quality)
% JPEG quantization matrix for 8x8 dct block as a function of the quality factor
% quality   should be an integer between 1 and 100 (default is 50)

if nargin<1, quality=50; end
Q50=[16 11 10 16  24  40  51  61
     12 12 14 19  26  58  60  55
     14 13 16 24  40  57  69  56
     14 17 22 29  51  87  80  62
     18 22 37 56  68 109 103  77
     24 35 55 64  81 104 113  92
     49 64 78 87 103 121 120 101
     72 92 95 98 112 100 103  99];  % Quantization matrix for 50% quality
  
if quality>=50, 
   Q=max(1,round(2*Q50*(1-quality/100)));
else  
   Q=min(255,round(Q50*50/quality)); 
end