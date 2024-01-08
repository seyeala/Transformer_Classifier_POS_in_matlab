function wordvectormap = readvectorsfromcsvf(filepath)
    global wordvectormap;
    % open the file
    fileid = fopen(filepath, 'r');

    %  if the file is not  opened successfully
    if fileid == -1
        error('could not open the file.');
    end

    % prepare  containers. Map to  hold the word-vector pairs
    wordvectormap = containers.Map('keytype', 'char', 'valuetype', 'any');

    % initialize  variables
    buffer = '';
    isvector = false; % track if ins ide a vector
    currentword = '';
    wordcounter = 0; % initialize a  counter for processed words


    while ~feof(fileid)
        line = fgetl(fileid);
        if ischar(line)
            if ~isvector
                % look for the start  of a vector marked by '['
                [currentword, restofline] = strtok(line, '[');
                % remove non-alphanumeric  characters, convert to lowercase
                currentword = lower(regexprep(strtrim(currentword), '[^a-zA-Z0-9_-]', ''));
                buffer = restofline;
                isvector = ~isempty(buffer);
            else
                buffer = [buffer ' ' line]; % c oncatenate
            end

            % check if this line con tains the end by ']'
            if isvector && contains(buffer, ']')
                % extract the vector string
                endidx = find(buffer == ']', 1, 'first');
                vectorstr = buffer(2:endidx-1);
                buffer = buffer(endidx+1:end);
                isvector = false;
                vector = str2num(['[', vectorstr, ']']);
                wordvectormap(currentword) = vector;
                wordcounter = wordcounter + 1;
                if mod(wordcounter, 1000) == 0
                    fprintf('processed %d words...\n', wordcounter);
                end
            end
        end
    end

    fclose(fileid);

    fprintf('total words processed: %d\n', wordcounter);
end
