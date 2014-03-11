function main()
    clear;
    VSM();
    LSI();
    [vsm_results,lsi_results] = getData();
    load('QueriesIDs.mat');
    load('trainingDocs.mat');
    w = 0.95;
    fprintf('Calculating combined scores...')
    [scores,docIDs] = calcFinalScores(vsm_results,lsi_results,w);
    fprintf('Done!\n')
    [sorted_scores, indexes] = sort(scores,2,'descend');
    fprintf('Writing combined.txt ...');
    file = fopen(strcat('combined.txt'),'wt');
    for i=1:499
           for j=1:1000
               fprintf(file,'%s\tQ0\t%d\t1\t%f\thw2\n',qIDs{i},docIDs(i,indexes(i,j)),sorted_scores(i,j));
           end
    end
    fprintf('Done!');
end

function LSI()
    tic;
    load('corel5k_text_term_docs.mat');
    fprintf('Calculating SVD...');
    [U,S] = svd(trainTermDocs);
    fprintf('Done!\n');
    k=calcOptimalK(diag(S));
    fprintf('Value of k used : %d (rank-%d approximation)\n',k,k)
    Uk = U(:,1:k);
    qk = Uk'*testTermDocs;
    dk = Uk'*trainTermDocs;
    normDk = normc(dk);
    normQk = normc(qk);
    fprintf('Calculating similarities matrix...');
    sim = normQk'*normDk;
    fprintf('Done!\n');
    fprintf('Sorting similarities matrix...');
    [sorted_scores, indexes] = sort(sim,2,'descend');
    fprintf('Done!\n');
    q_size = size(qk,2);
    load('QueriesIDs.mat');
    load('trainingDocs.mat');
    fprintf('Writing lsi.txt ...');
    file = fopen('lsi.txt','wt');
    for i=1:q_size
       for j=1:500
           fprintf(file,'%s\tQ0\t%s\t1\t%f\thw2_lsi\n',qIDs{i},trainIDs{indexes(i,j)},sorted_scores(i,j));
       end
    end
    fprintf('Done!\n');
    fclose(file);
    toc;
end

%Calculate optimal k using rule of thumb
function [k] = calcOptimalK(diag)
    prec = 0.85; %set precision to 85%
    i = 1;
    vec_sum = sum(diag);
    temp_sum = diag(i);
    temp_prec = temp_sum /vec_sum;
    vec_length = length(diag);
    while(i < vec_length)
       if(temp_prec >= prec)
           break;
       end
       i = i + 1;
       temp_sum = temp_sum + diag(i);
       temp_prec = temp_sum / vec_sum;
    end
    k = i;
end

function [vsm_results,lsi_results] = getData()
    vsm_file = fopen('vsm.txt');
    lsi_file = fopen('lsi.txt');
    vsm_results = textscan(vsm_file, '%d %*s %d %*d %f %*s', 'delimiter','\t');
    lsi_results = textscan(lsi_file, '%d %*s %d %*d %f %*s', 'delimiter','\t');
    fclose(vsm_file);
    fclose(lsi_file);
end

function [scores,docIDs] = calcFinalScores(vsm_results,lsi_results,w)
    scores = zeros(499,1000);
    docIDs = zeros(499,1000);
    size(vsm_results{2});
    for i=1:499
        docs_num = 1;
        qID = vsm_results{1}(i*500);
        for j=1:500
            docID = vsm_results{2}(((i-1)*500)+j);
            [row,column] = find(lsi_results{2}(((i-1)*500 + 1):i*500) == docID);
            if (~isempty(row))
                scores(i,docs_num) = ( w * vsm_results{3}(((i-1)*500)+j) ) + ( (1-w) * lsi_results{3}((((i-1)*500)+row)) );
                docIDs(i,docs_num) = docID;
                docs_num = docs_num + 1;
            else
                scores(i,docs_num) = w * vsm_results{3}(((i-1)*500)+j);
                docIDs(i,docs_num) = docID;
                docs_num = docs_num + 1;
            end
        end
        for j=1:500
            docID = lsi_results{2}(((i-1)*500)+j);
            [row,column] = find(vsm_results{2}(((i-1)*500 + 1):i*500) == docID);
            if (isempty(row))
                scores(i,docs_num) = ( (1-w) * lsi_results{3}(((i-1)*500)+j) );
                docIDs(i,docs_num) = docID;
                docs_num = docs_num + 1;
            end
        end
    end
end