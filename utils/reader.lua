local stringx = require 'pl.stringx'

local Vocab = torch.class('Vocab')
local DataReader = torch.class('DataReader')

function Vocab:__init()
    self.input = {}
	self.input.vocab = {['<null>']=1, ['<unk>']=2}
	self.input.inv_map = {'<null>', '<unk>'}
	self.input.size = #self.input.inv_map
    self.output = {}
    self.output.vocab = {['<eos>']=1,['<null>']=2}
	self.output.inv_map = {'<eos>','<null>'}
    self.output.size = #self.output.inv_map
    self.null_labels = {'<eos>','<null>'} -- = 'O'
    self.unk = {['<unk>']=0, ['<UNK>']=0,}
end

function Vocab:label_weights()
    local weights = torch.Tensor(self.output.size)
    for i = 1, self.output.vocab['<null>']-1 do
        weights[i] = 1
    end
    for i = self.output.vocab['<null>']+1, self.output.size do
        weights[i] = 1
    end
    weights[self.output.vocab['<null>']] = 0
    return weights
end

function Vocab:add_input(word)
	if self.input.vocab[word] == nil then
		self.input.size = self.input.size + 1
		self.input.vocab[word] = self.input.size
		self.input.inv_map[self.input.size] = word
	end
end

function Vocab:add_output(label)
	if self.output.vocab[label] == nil then
		self.output.size = self.output.size + 1
		self.output.vocab[label] = self.output.size
		self.output.inv_map[self.output.size] = label
	end
end

--function Vocab:is_eos(x)
--	return x == self:get_input('<null>')
--end

function Vocab:is_null(x)
    return x == self:get_input('<null>')
end

function Vocab:get_input(word)
	if self.input.vocab[word] == nil then
		return self.input.vocab['<unk>']
	end
	return self.input.vocab[word]
end

function Vocab:get_output(word)
	if self.output.vocab[word] == nil then
		return self.output.vocab['<null>']
	end
	return self.output.vocab[word]
end

function Vocab:inv_get_input(idx)
	if idx > self.input.size then
		return '<unk>'
	end
	return self.input.inv_map[idx]
end

function Vocab:inv_get_output(idx)
	if idx > self.output.size or idx <= #self.null_labels then
		return 'O'
	end
	return self.output.inv_map[idx]
end

function Vocab:vocab_size()
    return {['input']=self.input.size, ['output']=self.output.size}
end

function Vocab:save(input_file, output_file)
	local file = torch.DiskFile(input_file, 'w')
	for i,v in ipairs(self.input.inv_map) do
		file:writeString(v .. '\n')
	end
	file:close()
	local file = torch.DiskFile(output_file, 'w')
	for i,v in ipairs(self.output.inv_map) do
		file:writeString(v .. '\n')
	end
	file:close()
end

function Vocab:build_vocab(input_file, label_or_not)
    local input, output = {}, {}
	local file = torch.DiskFile(input_file, 'r')
	file:quiet()
	while true do
		local feats, labels = readline(file)
		if file:hasError() then
			break
		end
        --print(table.concat(feats,' '))
        --print(table.concat(labels,' '))
        if #feats ~= #labels then
            os.exit()
        end
        for i = 1, #feats do
            if input[feats[i]] == nil then
                input[feats[i]] = 1
            else
                input[feats[i]] = input[feats[i]] + 1
            end
            if label_or_not ~= false then
                if output[labels[i]] == nil then
                    output[labels[i]] = 1
                else
                    output[labels[i]] = output[labels[i]] + 1
                end
            end
		end
	end
	file:close()
    for word in pairs(input) do
        if input[word] ~= 1 and self.unk[word] == nil then
            self:add_input(word)
        end
    end
    if label_or_not ~= false then
        for label in pairs(output) do
            self:add_output(label)
        end
    end
end

function Vocab:build_vocab_input(input_file)
    local input, output = {}, {}
	local file = torch.DiskFile(input_file, 'r')
	file:quiet()
	while true do
		local data = readline1(file)
		if file:hasError() then
			break
		end
        self:add_input(data[1])
	end
end

function Vocab:build_vocab_output(input_file)
    local input, output = {}, {}
	local file = torch.DiskFile(input_file, 'r')
	file:quiet()
	while true do
		local data = readline1(file)
		if file:hasError() then
			break
		end
        self:add_output(data[1])
	end
end


function get_sentence_length(train, valid, test)
    local mx, my = 0, 0
    local input_filenames = {train, valid, test}
    for i = 1, 3 do
        local input_filename = input_filenames[i]
        if input_filename then
            local input_file = torch.DiskFile(input_filename, 'r')
            input_file:quiet()
            while not input_file:hasError() do
                local feats,labels = readline(input_file)
                if feats == nil then
                    break
                end
                mx = math.max(mx, #feats+1) -- add EOS
                my = math.max(my, #labels+1) -- add EOS
            end
            input_file:close()
        end
    end
    return mx, my
end

function DataReader:__init(input_file, batch_size, vocab, word_win_left, word_win_right)
    assert(word_win_left>=0)
    assert(word_win_right>=0)
    self.word_win_left = word_win_left
    self.word_win_right = word_win_right

	self.input_file = torch.DiskFile(input_file, 'r')
	self.input_file:quiet()

	self.batch_size = batch_size
	self.vocab = vocab
	self.batch_x = {}
	self.batch_y = {}
	self.batch_mask_x = {}
	self.batch_mask_y = {}
	for i = 1, self.batch_size do
		self.batch_x[i] = {}
		self.batch_y[i] = {}
		self.batch_mask_x[i] = {}
		self.batch_mask_y[i] = {}
	end
end

function DataReader:get_batch_4train(is_input_start_from_left, is_there_end_eos)
    if self.input_file:hasError() then
        self.input_file:close()
        return nil, nil, nil
    end
	for i = 1, self.batch_size do
		self.batch_x[i] = {}
		self.batch_y[i] = {}
		self.batch_mask_x[i] = {}
		self.batch_mask_y[i] = {}
	end
    -- <eos> first
    local max_length_in_batch_x, max_length_in_batch_y = 0, 0
    for i = 1, self.batch_size do
        self.batch_y[i][1] = self.vocab:get_output('<eos>')
        self.batch_mask_y[i][1] = 1
        local feats, labels = readline(self.input_file)
        if feats == nil then
            if i == 1 then
                self.input_file:close()
                return nil, nil, nil
            end
            --self.batch_x[i][1] = self.vocab:get_input('<null>')
            self.batch_x[i][1] = {}
            for k = 1, self.word_win_left + 1 + self.word_win_right do
                self.batch_x[i][1][k] = self.vocab:get_input('<null>')
            end
            self.batch_y[i][1+1] = self.vocab:get_output('<null>')
            self.batch_mask_x[i][1] = 0
            self.batch_mask_y[i][1+1] = 0
            max_length_in_batch_x = math.max(max_length_in_batch_x, #self.batch_x[i])
            max_length_in_batch_y = math.max(max_length_in_batch_y, #self.batch_y[i])
        else
            local tmp = {}
            for j = 1, self.word_win_left do
                tmp[j] = self.vocab:get_input('<null>')  -- PADDING
            end
            for j = 1, #feats do
                --self.batch_x[i][j] = self.vocab:get_input(feats[j])
                tmp[self.word_win_left + j] = self.vocab:get_input(feats[j])
                self.batch_mask_x[i][j] = 1
            end
            for j = 1, self.word_win_right do
                tmp[self.word_win_left + #feats + j] = self.vocab:get_input('<null>')  -- PADDING
            end
            for j = 1, #feats do
                local item = {}
                for k = -self.word_win_left, self.word_win_right do
                    item[self.word_win_left + k + 1] = tmp[self.word_win_left + j + k]
                end
                self.batch_x[i][j] = item
            end

            max_length_in_batch_x = math.max(max_length_in_batch_x, #self.batch_x[i])
            for j = 1, #labels do
                self.batch_y[i][j+1] = self.vocab:get_output(labels[j])
                self.batch_mask_y[i][j+1] = 1
            end
            if is_there_end_eos then
                self.batch_y[i][#labels+2] = self.vocab:get_output('<eos>')
                self.batch_mask_y[i][#labels+2] = 0 -- yes, it should be 0
            end
            max_length_in_batch_y = math.max(max_length_in_batch_y, #self.batch_y[i])
        end
    end
    
    for i = 1, self.batch_size do
        if is_input_start_from_left then
            if #self.batch_x[i] < max_length_in_batch_x then
                for j = #self.batch_x[i]+1, max_length_in_batch_x do
                    --self.batch_x[i][j] = self.vocab:get_input('<null>')
                    self.batch_x[i][j] = {}
                    for k = 1, self.word_win_left + 1 + self.word_win_right do
                        self.batch_x[i][j][k] = self.vocab:get_input('<null>')
                    end
                    self.batch_mask_x[i][j] = 0
                end
            end
        else
            local cur_length = #self.batch_x[i]
            if cur_length < max_length_in_batch_x then
                for j = max_length_in_batch_x, max_length_in_batch_x - cur_length + 1, -1 do
                    self.batch_x[i][j] = self.batch_x[i][j - (max_length_in_batch_x - cur_length)]
                    self.batch_mask_x[i][j] = self.batch_mask_x[i][j - (max_length_in_batch_x - cur_length)]
                end
                for j = 1, max_length_in_batch_x - cur_length do
                    --self.batch_x[i][j] = self.vocab:get_input('<null>')
                    self.batch_x[i][j] = {}
                    for k = 1, self.word_win_left + 1 + self.word_win_right do
                        self.batch_x[i][j][k] = self.vocab:get_input('<null>')
                    end
                    self.batch_mask_x[i][j] = 0
                end
            end
        end
        self.batch_x[i][max_length_in_batch_x+1] = nil
        self.batch_mask_x[i][max_length_in_batch_x+1] = nil
        if #self.batch_y[i] < max_length_in_batch_y then
            for j = #self.batch_y[i] + 1, max_length_in_batch_y do
                self.batch_y[i][j] = self.vocab:get_output('<null>')
                self.batch_mask_y[i][j] = 0
            end
        end
        self.batch_y[i][max_length_in_batch_y+1] = nil
        self.batch_mask_y[i][max_length_in_batch_y+1] = nil
    end  
    
    return torch.Tensor(self.batch_x):transpose(1,2), torch.Tensor(self.batch_mask_x):t(), torch.Tensor(self.batch_y):t(), torch.Tensor(self.batch_mask_y):t() -- transposed to length * batch (needed during forward())
end

function DataReader:get_batch_4test(is_there_end_eos)
    if self.input_file:hasError() then
        self.input_file:close()
        return nil, nil, nil
    end
    local batch_x, batch_y = {}, {}
    batch_x[1] = {}
    batch_y[1] = {[1]=self.vocab:get_output('<eos>')}
    local feats, labels = readline(self.input_file)
    if feats == nil then
        self.input_file:close()
        return nil, nil, nil
    end

    local tmp = {}
    for j = 1, self.word_win_left do
        tmp[j] = self.vocab:get_input('<null>')  -- PADDING
    end
    for j = 1, #feats do
        --batch_x[1][j] = self.vocab:get_input(feats[j])
        tmp[self.word_win_left + j] = self.vocab:get_input(feats[j])
    end
    for j = 1, self.word_win_right do
        tmp[self.word_win_left + #feats + j] = self.vocab:get_input('<null>')  -- PADDING
    end
    for j = 1, #feats do
        local item = {}
        for k = -self.word_win_left, self.word_win_right do
            item[self.word_win_left + k + 1] = tmp[self.word_win_left + j + k]
        end
        batch_x[1][j] = item
    end

    for j = 1, #labels do
        batch_y[1][j+1] = self.vocab:get_output(labels[j])
    end
    if is_there_end_eos then
        batch_y[1][#labels+2] = self.vocab:get_output('<eos>')
    end
    
    batch_x[1][#feats+1] = nil
    batch_y[1][#batch_y[1]+1] = nil
    
    return torch.Tensor(batch_x):transpose(1,2), torch.Tensor(batch_y):t() -- transposed to length * batch (needed during forward())
end

