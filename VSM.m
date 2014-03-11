function VSM()
    clear;
    tic
    load('corel5k_text_term_docs.mat');
    fprintf('Calculating doc weights...')
    [IDF,dw] = calcDocWeight(trainTermDocs);
    fprintf('Done!\nCalculating query weights...')
    qw = calcQueryWeight(IDF,testTermDocs);
    fprintf('Done!\nCalculating scores...')
    scores = calcScores(normc(dw),normc(qw));
    fprintf('Done!\nSorting scores...')
    [sorted_scores, indexes] = sort(scores,2,'descend');
    fprintf('Done!\nWriting vsm.txt...')
    load('QueriesIDs.mat');
    load('trainingDocs.mat');
    file = fopen('vsm.txt','wt');
    q_size = size(qw,2);
    for i=1:q_size
       for j=1:500
           fprintf(file,'%s\tQ0\t%s\t1\t%f\thw1_vsm\n',qIDs{i},trainIDs{indexes(i,j)},sorted_scores(i,j));
       end
    end
    fprintf('Done!\n');
    fclose(file);
    toc
end

%This function is used to calculate TF*IDF weights
%@return IDF : vector containing IDF values
%@return res : Tf-IDF matrix (each column represents a weighted document
%vector)
function [IDF,res] = calcDocWeight(A)
    IDF = calcIDF(A);
    res = bsxfun(@times,IDF,A); %perform element wise multiplication between IDF vector's elements and the columns of A matrix
end

%This function is used to calculate IDF
%@return IDF : vector containing IDF values
function [IDF] = calcIDF(A)
    vec_size = size(A,1);
    N = size(A,2);
    IDF = zeros(vec_size,1);
    for i=1:vec_size
        df = sum(A(i,:));
        if(df ~= 0)
            IDF(i) = 1 + (log(N/df));
        end
    end
end

%This function is used to calculate queries weights
%@return res : matrix (each column represents a weighted query
%vector)
function [res] = calcQueryWeight(IDF,A)
    res = bsxfun(@times,IDF,A);
end

%This function is used to 
%check if document d contains at least one term that can
%be found at query q
%@return: value = 0 if at least one matching term was found
%         value = 1 otherwise
function [value] = docContainsQueryTerms(d,q)
    dim = length(d);
    for i=1:dim
        if(d(i) ~= 0 && q(i) ~= 0)
            value = 0;
            return;
        end
    end
    value = 1;
    return;
end

%This function calculates the scores
%@param dw : normalized documents matrix
%@param qw : normalized queries matrix
function [scores] = calcScores(dw,qw)
    q_size = size(qw,2);
    doc_size = size(dw,2);
    scores = zeros(q_size,doc_size);
    for i=1:q_size
        for j=1:doc_size
            val = docContainsQueryTerms(dw(:,j),qw(:,i));
            if(val == 0)
                scores(i,j) = dot(qw(:,i),dw(:,j));
            end
        end
    end
end