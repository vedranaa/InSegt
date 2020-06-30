function [fIm, vec, mP] = getPCA_feat(im,M,nPatch,nKeep)

[r,c,l] = size(im);

Mh = floor(M/2);
nTot = (r-M+1)*(c-M+1);

nPatch = min(nTot, nPatch);
id = randperm(nTot,nPatch);

s_cid = ceil(id/(r-M+1));
s_rid = id - (s_cid-1)*(r-M+1);

s_cid = s_cid + Mh;
s_rid = s_rid + Mh;

P = zeros(nPatch,l*M^2);

for i = 1:nPatch
    tmp = im((s_rid(i)-Mh):(s_rid(i)+Mh),(s_cid(i)-Mh):(s_cid(i)+Mh),:);
    P(i,:) = tmp(:);
end

mP = mean(P);
P_n = P - ones(nPatch,1)*mP;
cv = cov(P_n);

[vec,val] = eig(cv);

if ( nKeep >= 1 )
    nKeep = min(nKeep, M^2);
else
    v = sqrt(diag(val));
    vp = v/sum(v);
    i = length(v);
    vs = 0;
    while ( i > 0 && vs < nKeep )
        vs = vs + vp(i);
        i = i - 1;
    end
    nKeep = length(v)-i;
end

vec = vec(:,end-nKeep+1:end);

colIm = zeros((r-M+1)*(c-M+1),M*M*l);
for i = 1:l % for color images
    f = (i-1)*M*M+1;
    t = i*M*M;
    colIm(:,f:t) = im2col(im(:,:,i),[M,M])';
end
% Normalize column image
colIm = (colIm - ones((r-M+1)*(c-M+1),1)*mP);
% lColIm = sqrt(sum(colIm.^2));
% id = find(lColIm > 0);
% colIm(:,id) = colIm(:,id)./(ones(size(id,1),1)*lColIm(id));

% Compute feature vector
feat_vec = vec(:,end-nKeep+1:end);
feat = colIm*feat_vec;
fIm_no_border = reshape(feat,[r-M+1,c-M+1,nKeep]);
fIm = zeros(r,c,nKeep);
fIm(Mh+1:end-Mh,Mh+1:end-Mh,:) = fIm_no_border;

