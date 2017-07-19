require 'luarocks.loader'
require 'nngraph'
require 'cunn'
require 'utils/linear3D'

local TRAIN_LOG_WORDS = 100000

local LSTM = torch.class('LSTM')

function LSTM:__init(options)
	self.options = options
    assert(self.options.mx == self.options.my)
    -- padding label doesn't join in BP
    self.label_weights = transfer2gpu(self.options.vocab:label_weights())
    -- hidden size of attention model
    self.options.align_hidden_size = self.options.hidden_size[self.options.layers]
end

-- lstm cell activation function
-- no peephole connection.
function LSTM:lstm(input, prev_c, prev_h, input_size, hidden_size)
	-- bias is automatically included in nn.Linear()
	local function input_hidden_sum()
		local w_i2h = nn.Linear(input_size, hidden_size)
		--local w_h2h = nn.Linear(hidden_size, hidden_size, false) 
		local w_h2h = nn.Linear(hidden_size, hidden_size) 
		return nn.CAddTable()({w_i2h(input), w_h2h(prev_h)}) --w_i2h(input) is a node!
	end
	local function input_hidden_cell_sum(Xcell)
		local w_i2h = nn.Linear(input_size, hidden_size)
        local w_c2h = nn.CMul(hidden_size)
		--local w_h2h = nn.Linear(hidden_size, hidden_size, false) 
		local w_h2h = nn.Linear(hidden_size, hidden_size) 
		return nn.CAddTable()({w_i2h(input), w_c2h(Xcell), w_h2h(prev_h)}) --w_i2h(input) is a node!
	end
	--local input_gate = nn.Sigmoid()(input_hidden_sum())
    local input_gate = nn.Sigmoid()(input_hidden_cell_sum(prev_c))
	--local forget_gate = nn.Sigmoid()(input_hidden_sum())
    local forget_gate = nn.Sigmoid()(input_hidden_cell_sum(prev_c))
	local cell_input = nn.Tanh()(input_hidden_sum())
	local cell = nn.CAddTable()({nn.CMulTable()({input_gate, cell_input}), nn.CMulTable()({forget_gate, prev_c})})
	--local output_gate = nn.Sigmoid()(input_hidden_sum())
    local output_gate = nn.Sigmoid()(input_hidden_cell_sum(cell))
	local hidden = nn.CMulTable()({output_gate, nn.Tanh()(cell)})
	return cell, hidden --lstm returns two nodes!
end

function LSTM:lstm_context(input, prev_c, prev_h, input_size, hidden_size, context_hidden, c_hidden_size)
	-- bias is automatically included in nn.Linear()
    local new_prev_h = nn.JoinTable(2)({prev_h, context_hidden}) 
	local function input_hidden_sum()
		local w_i2h = nn.Linear(input_size, hidden_size)
		----local w_h2h = nn.Linear(hidden_size, hidden_size, false) 
		local w_h2h = nn.Linear(hidden_size + c_hidden_size, hidden_size) 
		return nn.CAddTable()({w_i2h(input), w_h2h(new_prev_h)}) --w_i2h(input) is a node!
	end
	local function input_hidden_cell_sum(Xcell)
		local w_i2h = nn.Linear(input_size, hidden_size)
        local w_c2h = nn.CMul(hidden_size)
		----local w_h2h = nn.Linear(hidden_size, hidden_size, false) 
		local w_h2h = nn.Linear(hidden_size + c_hidden_size, hidden_size) 
		return nn.CAddTable()({w_i2h(input), w_c2h(Xcell), w_h2h(new_prev_h)}) --w_i2h(input) is a node!
	end
	--local input_gate = nn.Sigmoid()(input_hidden_sum())
    local input_gate = nn.Sigmoid()(input_hidden_cell_sum(prev_c))
	--local forget_gate = nn.Sigmoid()(input_hidden_sum())
    local forget_gate = nn.Sigmoid()(input_hidden_cell_sum(prev_c))
	local cell_input = nn.Tanh()(input_hidden_sum())
	local cell = nn.CAddTable()({nn.CMulTable()({input_gate, cell_input}), nn.CMulTable()({forget_gate, prev_c})})
	--local output_gate = nn.Sigmoid()(input_hidden_sum())
    local output_gate = nn.Sigmoid()(input_hidden_cell_sum(cell))
	local hidden = nn.CMulTable()({output_gate, nn.Tanh()(cell)})
	return cell, hidden --lstm returns two nodes!
end

function LSTM:build_word_embedding_layer()
	local input = nn.Identity()() 
	local wvec = LookupTable(self.options.vocab_size['input'], self.options.emb_size)(input)
    local wvec_input = nn.Reshape((self.options.word_win_left + 1 + self.options.word_win_right) * self.options.emb_size)(wvec)
	local model = nn.gModule({input}, {wvec_input})
	--local model = nn.gModule({input}, {wvec})
	model:getParameters():uniform(-self.options.init_weight, self.options.init_weight)
	model = transfer2gpu(model)
	return model
end

function LSTM:build_rnn_layer()
	local input = nn.Identity()() 
	local prev_state = nn.Identity()() -- saves hidden states at all layers, hidden state includes cell activation and hidden state activation
	local next_state = {}
	local net = {[0] = input}
	local prev_split = {prev_state:split(2 * self.options.layers)} -- each hidden layer is split, one for cell, one for hidden, there will be two successors of nn.Identity()() if nn.Identity()() is split.
    local prev_cell = prev_split[1]
    local prev_hidden = prev_split[2]
    local dropped_input = nn.Dropout(self.options.dropout)(net[0])
    local next_cell, next_hidden = self:lstm(dropped_input, prev_cell, prev_hidden, (self.options.word_win_left + 1 + self.options.word_win_right) * self.options.emb_size, self.options.hidden_size[1])
    table.insert(next_state, next_cell)
    table.insert(next_state, next_hidden)
    net[1] = next_hidden
    if self.options.layers > 1 then
        for i = 2, self.options.layers do
            local prev_cell = prev_split[2 * i - 1]
            local prev_hidden = prev_split[2 * i]
            local dropped_input = nn.Dropout(self.options.dropout)(net[i - 1])
            local next_cell, next_hidden = self:lstm(dropped_input, prev_cell, prev_hidden, self.options.hidden_size[i-1], self.options.hidden_size[i])
            table.insert(next_state, next_cell)
            table.insert(next_state, next_hidden)
            net[i] = next_hidden
        end
    end
	
	local model = nn.gModule({input, prev_state}, {nn.Identity()(next_state)})
	model:getParameters():uniform(-self.options.init_weight, self.options.init_weight)
	model = transfer2gpu(model)
	return model
end

function LSTM:build_attention_layer()
    local prev_state = nn.Identity()() -- nstep * batch * hidden_size -- it's repeated
    local encode_states_left = nn.Identity()()  -- table of tensors : nstep * [batch * 1 * hidden_size]
    local encode_states_right = nn.Identity()()  -- table of tensors : nstep * [batch * 1 * hidden_size]
    local mask_c = nn.Identity()() -- batch * nstep

    local dropped_prev_state = nn.Dropout(self.options.dropout)(prev_state)

    local w_a = nn.Linear3D(self.options.hidden_size[self.options.layers], self.options.align_hidden_size, false)
    local u_a = nn.Linear3D(2 * self.options.hidden_size[self.options.layers], self.options.align_hidden_size, false)

    local prev_state_ = nn.Transpose({1, 2})(dropped_prev_state)
    local c_from_dec = w_a(prev_state_) -- batch * nstep * self.options.align_hidden_size
    --local c_from_dec = w_a(prev_state)
	--c_from_dec:annotate({["name"] = "c2"})
    local encode_states_tensor_left = nn.JoinTable(2)(encode_states_left) -- batch * nstep * hidden_size
    local encode_states_tensor_right = nn.JoinTable(2)(encode_states_right) -- batch * nstep * hidden_size
    local encode_states_tensor = nn.JoinTable(3)({encode_states_tensor_left, encode_states_tensor_right})

    local dropped_encode_states_tensor = nn.Dropout(self.options.dropout)(encode_states_tensor)

    --encode_states_tensor:annotate({["name"] = "encode_states_tensor"})
    local c_from_enc = u_a(dropped_encode_states_tensor) -- batch * nstep * self.options.align_hidden_size
    -- --local c_from_dec_uns = nn.Unsqueeze(2)(c_from_dec)
    --local c = nn.CAddTable()({c_from_enc, nn.Replicate(self.options.mx, 2)(c_from_dec)})--_uns)})
    local c = nn.CAddTable()({c_from_enc, c_from_dec})
    local right = nn.Tanh()(c)
    local v_a = nn.Linear3D(self.options.align_hidden_size, 1, false)
    local u = v_a(right) -- batch * nstep * 1
    u:annotate({["name"] = "u"})
    --local u_ = nn.Transpose({2, 3})(u) -- batch * 1 * nstep
    local u_t = nn.Squeeze(3)(u) -- batch * nstep

    --u:annotate({["name"] = "u_t"})
    local energy = nn.Exp()(u_t)
    --energy:annotate({["name"] = "energy"})
    local masked_energy = nn.CMulTable()({mask_c, energy})
    --masked_energy:annotate({["name"] = "masked_energy"})
    local probs = nn.Normalize(1)(masked_energy) -- L_1 norm
    --local probs = nn.SoftMax()(u_t)
    probs:annotate({["name"] = "probs"})
    
    local probs_3d = nn.Unsqueeze(2)(probs)
    local context_state = nn.MM()({probs_3d, encode_states_tensor}) -- no dropout
    --local context_state = nn.MixtureTable(2)({probs, encode_states}) -- batch * 1 * hidden_size
    local output = nn.Squeeze(2)(context_state)
    local model = nn.gModule({prev_state, encode_states_left, encode_states_right, mask_c}, {nn.Identity()(output)})
	model:getParameters():uniform(-self.options.init_weight, self.options.init_weight)
	model = transfer2gpu(model)
    return model
end

function LSTM:build_rnn_layer_context()
	local input = nn.Identity()() 
	local prev_state = nn.Identity()() -- saves hidden states at all layers, hidden state includes cell activation and hidden state activation
    local context_hidden = nn.Identity()()
	local next_state = {}
	local wvec = LookupTable(self.options.vocab_size['output'], self.options.emb_size)(input)
	local net = {[0] = wvec} --local net = {[0] = wvec_input}
	local prev_split = {prev_state:split(2 * self.options.layers)} -- each hidden layer is split, one for cell, one for hidden, there will be two successors of nn.Identity()() if nn.Identity()() is split.
    local prev_cell = prev_split[1]
    local prev_hidden = prev_split[2]
    local dropped_input = nn.Dropout(self.options.dropout)(net[0])
    local dropped_context_hidden = self.options.context_dropout and nn.Dropout(self.options.dropout)(context_hidden) or context_hidden
    local next_cell, next_hidden = self:lstm_context(dropped_input, prev_cell, prev_hidden, self.options.emb_size, self.options.hidden_size[1], dropped_context_hidden, 2 * self.options.hidden_size[self.options.layers])
    table.insert(next_state, next_cell)
    table.insert(next_state, next_hidden)
    net[1] = next_hidden
    if self.options.layers > 1 then
        for i = 2, self.options.layers do
            local prev_cell = prev_split[2 * i - 1]
            local prev_hidden = prev_split[2 * i]
            local dropped_input = nn.Dropout(self.options.dropout)(net[i - 1])
            local next_cell, next_hidden = self:lstm(dropped_input, prev_cell, prev_hidden, self.options.hidden_size[i-1], self.options.hidden_size[i])
            table.insert(next_state, next_cell)
            table.insert(next_state, next_hidden)
            net[i] = next_hidden
        end
    end
	
	local model = nn.gModule({input, prev_state, context_hidden}, {nn.Identity()(next_state)})
	model:getParameters():uniform(-self.options.init_weight, self.options.init_weight)
	model = transfer2gpu(model)
	return model
end

function LSTM:build_output_layer()
	local input = nn.Identity()() 
	local target = nn.Identity()()
	
	local dropped_input = nn.Dropout(self.options.dropout)(input)
	local output = nn.Linear(self.options.hidden_size[self.options.layers], self.options.vocab_size['output'])(dropped_input)
	local log_prob = nn.LogSoftMax()(output)
	log_prob:annotate({["name"] = "log_prob"})
	local classifier = nn.ClassNLLCriterion(self.label_weights)
	classifier.sizeAverage = false
	local err = classifier({log_prob, target})
	local model = nn.gModule({input, target}, {err})
	model:getParameters():uniform(-self.options.init_weight, self.options.init_weight)
	model = transfer2gpu(model)
	return model
end

function LSTM:build_net()
    local network = {} 
    network[1] = self:build_rnn_layer() -- encoder : from left to right
    network[2] = self:build_rnn_layer() -- encoder : from right to left
    network[3] = self:build_rnn_layer_context() -- decoder  
    network[4] = self:build_output_layer() -- output layer of decoder
    network[5] = self:build_attention_layer() -- attention
    network[6] = self:build_word_embedding_layer()
    return network
end

function LSTM:init(input_model)
	if input_model == '' then
		self.core_model = self:build_net()
	else
		if self.options.trace_level > 0 then
			print('Loading model from ' .. input_model)
		end
		self:load_model(input_model)
	end
	self.params, self.grads = {}, {}
    self.models = {}
    for i = 1, 6 do
        self.params[i], self.grads[i] = self.core_model[i]:getParameters()   -- it should be called only once
        --self.params[i]:uniform(-self.options.init_weight, self.options.init_weight) -- parameter initialization
    end
    self.models[1] = make_recurrent(self.core_model[1], self.options.mx)
    self.models[2] = make_recurrent(self.core_model[2], self.options.mx)
    self.models[3] = make_recurrent(self.core_model[3], self.options.my)
    self.models[4] = make_recurrent(self.core_model[4], self.options.my)
    self.models[5] = make_recurrent(self.core_model[5], self.options.my)
    self.models[6] = make_recurrent(self.core_model[6], self.options.mx)

    self.embeddings = {} --reference
    self.grad_em = {}
	self.history_encoder_left, self.history_encoder_right = {}, {}  -- reference
	self.history_decoder = {}  -- reference
	self.tmp_hist_encoder_left, self.tmp_hist_encoder_right = {}, {}
	self.tmp_hist_decoder = {}
    self.context = {}
	self.grad_h_encoder_left, self.grad_h_encoder_right = {}, {}
	self.grad_h_decoder = {}
	self.grad_decoder_from_output = {}
    self.grad_of_context = {}
    self.grad_encoder_from_context_left, self.grad_encoder_from_context_right = {}, {} --
    self.grad_decoder_from_context = {} -- 
    self.history_encoder2decoder = {}
    self.hidden_encoder_left, self.hidden_encoder_right = {}, {}
    self.hidden_encoder_cache_left, self.hidden_encoder_cache_right = {}, {}
	self.hidden_decoder_tmp = {} -- 
	self.test_history_encoder_left, self.test_history_encoder_right = {}, {}
    self.test_hidden_encoder_left, self.test_hidden_encoder_right = {}, {}
    self.test_hidden_encoder_cache_left, self.test_hidden_encoder_cache_right = {}, {}
	self.test_history_decoder = {}
    self.test_history_encoder2decoder = {}
	self.err = transfer2gpu(torch.zeros(1))
	for i = 1, self.options.mx do
        self.hidden_encoder_cache_left[i], self.hidden_encoder_cache_right[i] = transfer2gpu(torch.zeros(self.options.batch_size, 1, self.options.hidden_size[self.options.layers])), transfer2gpu(torch.zeros(self.options.batch_size, 1, self.options.hidden_size[self.options.layers]))
        self.test_hidden_encoder_cache_left[i], self.test_hidden_encoder_cache_right[i] = transfer2gpu(torch.zeros(self.options.test_batch_size, 1, self.options.hidden_size[self.options.layers])), transfer2gpu(torch.zeros(self.options.test_batch_size, 1, self.options.hidden_size[self.options.layers]))
        self.grad_encoder_from_context_left[i], self.grad_encoder_from_context_right[i] = transfer2gpu(torch.zeros(self.options.batch_size, self.options.hidden_size[self.options.layers])), transfer2gpu(torch.zeros(self.options.batch_size, 1, self.options.hidden_size[self.options.layers]))
        self.grad_em[i] = transfer2gpu(torch.zeros(self.options.batch_size, (self.options.word_win_left + 1 + self.options.word_win_right) * self.options.emb_size))
    end
	for i = 0, self.options.mx do
		self.history_encoder_left[i], self.history_encoder_right[i] = {}, {}
		self.tmp_hist_encoder_left[i], self.tmp_hist_encoder_right[i] = {}, {}
		self.test_history_encoder_left[i], self.test_history_encoder_right[i] = {}, {}
		for j = 1, self.options.layers do
			self.history_encoder_left[i][2*j-1], self.history_encoder_right[i][2*j-1] = transfer2gpu(torch.zeros(self.options.batch_size, self.options.hidden_size[j])), transfer2gpu(torch.zeros(self.options.batch_size, self.options.hidden_size[j]))
			self.history_encoder_left[i][2*j], self.history_encoder_right[i][2*j] = transfer2gpu(torch.zeros(self.options.batch_size, self.options.hidden_size[j])), transfer2gpu(torch.zeros(self.options.batch_size, self.options.hidden_size[j]))
			self.tmp_hist_encoder_left[i][2*j-1], self.tmp_hist_encoder_right[i][2*j-1] = transfer2gpu(torch.zeros(self.options.batch_size, self.options.hidden_size[j])), transfer2gpu(torch.zeros(self.options.batch_size, self.options.hidden_size[j]))
			self.tmp_hist_encoder_left[i][2*j], self.tmp_hist_encoder_right[i][2*j] = transfer2gpu(torch.zeros(self.options.batch_size, self.options.hidden_size[j])), transfer2gpu(torch.zeros(self.options.batch_size, self.options.hidden_size[j]))
			self.test_history_encoder_left[i][2*j-1], self.test_history_encoder_right[i][2*j-1] = transfer2gpu(torch.zeros(self.options.test_batch_size, self.options.hidden_size[j])), transfer2gpu(torch.zeros(self.options.test_batch_size, self.options.hidden_size[j]))
			self.test_history_encoder_left[i][2*j], self.test_history_encoder_right[i][2*j] = transfer2gpu(torch.zeros(self.options.test_batch_size, self.options.hidden_size[j])), transfer2gpu(torch.zeros(self.options.test_batch_size, self.options.hidden_size[j]))
		end
	end
    for i = 1, self.options.my do
        self.grad_decoder_from_output[i] = transfer2gpu(torch.zeros(self.options.batch_size, self.options.hidden_size[self.options.layers]))
        self.context[i] = transfer2gpu(torch.zeros(self.options.batch_size, 2 * self.options.hidden_size[self.options.layers]))
        self.grad_of_context[i] = transfer2gpu(torch.zeros(self.options.batch_size, 2 * self.options.hidden_size[self.options.layers]))
        self.grad_decoder_from_context[i] = transfer2gpu(torch.zeros(self.options.batch_size, self.options.hidden_size[self.options.layers]))
	end
	for i = 0, self.options.my do
		self.history_decoder[i] = {}
		self.tmp_hist_decoder[i] = {}
		self.test_history_decoder[i] = {}
		for j = 1, self.options.layers do
			self.history_decoder[i][2*j-1] = transfer2gpu(torch.zeros(self.options.batch_size, self.options.hidden_size[j]))
			self.history_decoder[i][2*j] = transfer2gpu(torch.zeros(self.options.batch_size, self.options.hidden_size[j]))
			self.tmp_hist_decoder[i][2*j-1] = transfer2gpu(torch.zeros(self.options.batch_size, self.options.hidden_size[j]))
			self.tmp_hist_decoder[i][2*j] = transfer2gpu(torch.zeros(self.options.batch_size, self.options.hidden_size[j]))
			self.test_history_decoder[i][2*j-1] = transfer2gpu(torch.zeros(self.options.test_batch_size, self.options.hidden_size[j]))
			self.test_history_decoder[i][2*j] = transfer2gpu(torch.zeros(self.options.test_batch_size, self.options.hidden_size[j]))
		end
    end
	for i = 1, self.options.layers do
		self.grad_h_encoder_left[2*i-1], self.grad_h_encoder_right[2*i-1] = transfer2gpu(torch.zeros(self.options.batch_size, self.options.hidden_size[i])), transfer2gpu(torch.zeros(self.options.batch_size, self.options.hidden_size[i]))
		self.grad_h_encoder_left[2*i], self.grad_h_encoder_right[2*i] = transfer2gpu(torch.zeros(self.options.batch_size, self.options.hidden_size[i])), transfer2gpu(torch.zeros(self.options.batch_size, self.options.hidden_size[i]))
		self.grad_h_decoder[2*i-1] = transfer2gpu(torch.zeros(self.options.batch_size, self.options.hidden_size[i]))
		self.grad_h_decoder[2*i] = transfer2gpu(torch.zeros(self.options.batch_size, self.options.hidden_size[i]))
		self.history_encoder2decoder[2*i-1] = transfer2gpu(torch.zeros(self.options.batch_size, self.options.hidden_size[i]))
		self.history_encoder2decoder[2*i] = transfer2gpu(torch.zeros(self.options.batch_size, self.options.hidden_size[i]))
		self.test_history_encoder2decoder[2*i-1] = transfer2gpu(torch.zeros(self.options.test_batch_size, self.options.hidden_size[i]))
		self.test_history_encoder2decoder[2*i] = transfer2gpu(torch.zeros(self.options.test_batch_size, self.options.hidden_size[i]))
	end
	self.test_cur_hist_decoder = {}
	self.test_next_hist_decoder = {}
    for i = 1, self.options.beam_size do
        self.test_cur_hist_decoder[i] = {}
        self.test_next_hist_decoder[i] = {}
        for j = 1, self.options.layers do
			self.test_cur_hist_decoder[i][2*j-1] = transfer2gpu(torch.zeros(self.options.test_batch_size, self.options.hidden_size[j]))
			self.test_cur_hist_decoder[i][2*j] = transfer2gpu(torch.zeros(self.options.test_batch_size, self.options.hidden_size[j]))
			self.test_next_hist_decoder[i][2*j-1] = transfer2gpu(torch.zeros(self.options.test_batch_size, self.options.hidden_size[j]))
			self.test_next_hist_decoder[i][2*j] = transfer2gpu(torch.zeros(self.options.test_batch_size, self.options.hidden_size[j]))
        end
    end
end

function LSTM:forward_training()
	local len = 0
    local err
    local n_step = self.cur_batch_x:size()[1]
    --reset(self.history_encoder[0])
    reset(self.tmp_hist_encoder_left[0])
    reset(self.tmp_hist_encoder_right[0])
    self.hidden_encoder_left, self.hidden_encoder_right = {}, {} -- init
    --reset(self.history_encoder2decoder)
    for i = 1, n_step do
        local input = self.cur_batch_x[i]
        self.embeddings[i] = self.models[6][i]:forward(input) -- in case of only one input of gModule, not {input}
    end
	for i = 1, n_step do
        local r = n_step-i+1
		local input_left = self.embeddings[i]  -- from left to right
        local input_right = self.embeddings[r] -- from right to left
        for j = 1, self.options.batch_size do
            if self.cur_batch_mask_x[i][j] == 0 then  -- current input is a padding
                for k = 1, 2 * self.options.layers do
                    self.tmp_hist_encoder_left[i-1][k][j]:zero()
                end
            end
            if r < n_step and self.cur_batch_mask_x[r+1][j] == 0 then -- last input is a padding
                for k = 1, 2 * self.options.layers do
                    self.tmp_hist_encoder_right[i-1][k][j]:zero()
                end
            end
        end
		self.history_encoder_left[i] = self.models[1][i]:forward({input_left, self.tmp_hist_encoder_left[i - 1]})
		self.history_encoder_right[i] = self.models[2][i]:forward({input_right, self.tmp_hist_encoder_right[i - 1]})
        replace(self.tmp_hist_encoder_left[i], self.history_encoder_left[i])
        replace(self.tmp_hist_encoder_right[i], self.history_encoder_right[i])
        replace(self.hidden_encoder_cache_left[i], self.history_encoder_left[i][2 * self.options.layers])
        replace(self.hidden_encoder_cache_right[i], self.history_encoder_right[i][2 * self.options.layers])
        for j = 1, self.options.batch_size do
            --[[if self.cur_batch_mask_x[i][j] == 1 then
                if (i == n_step) or (i < n_step and self.cur_batch_mask_x[i+1][j] == 0) then
                    for k = 1, 2 * self.options.layers do
                        replace(self.history_encoder2decoder[k][j], self.history_encoder_left[i][k][j])
                    end
                end
            end--]]
            if self.cur_batch_mask_x[i][j] == 0 then
                self.hidden_encoder_cache_left[i][j]:zero()
            end
            if self.cur_batch_mask_x[r][j] == 0 then
                self.hidden_encoder_cache_right[i][j]:zero()
            end
        end
        self.hidden_encoder_left[i] = self.hidden_encoder_cache_left[i]:view(self.options.batch_size, 1, self.options.hidden_size[self.options.layers]) -- view
        self.hidden_encoder_right[r] = self.hidden_encoder_cache_right[i]:view(self.options.batch_size, 1, self.options.hidden_size[self.options.layers]) -- view
	end
    --replace(self.tmp_hist_decoder[0], self.history_encoder2decoder)
    replace(self.tmp_hist_decoder[0], self.history_encoder_right[n_step])
    for i = 1, n_step do  -- self.my
        local input_label = self.cur_batch_y[i]
        local output_label = self.cur_batch_y[i+1]
        
        self.hidden_decoder_tmp[i] = self.tmp_hist_decoder[i - 1][2 * self.options.layers]:repeatTensor(n_step, 1, 1) -- decoder's hidden repeating
        self.context[i] = self.models[5][i]:forward({self.hidden_decoder_tmp[i], self.hidden_encoder_left, self.hidden_encoder_right, self.mask_x}) -- attention model
        --[[for j = 1, self.options.batch_size do
            if self.cur_batch_mask_y[i+1][j] == 0 then  -- <null> or the end <eos>
                self.context[i][j]:zero()
            end
        end
        --]]
        self.history_decoder[i] = self.models[3][i]:forward({input_label, self.tmp_hist_decoder[i - 1], self.context[i]})
        replace(self.tmp_hist_decoder[i], self.history_decoder[i])
        --[[for j = 1, self.options.batch_size do
            if self.cur_batch_mask_y[i+1][j] == 0 then  -- <null> or the end <eos>
                for k = 1, 2 * self.options.layers do
                    self.tmp_hist_decoder[i][k][j]:zero()
                end
            end
        end
        --]]
        err = self.models[4][i]:forward({self.history_decoder[i][2 * self.options.layers], output_label})
        self.err = self.err:add(err)
    end
end

function LSTM:backward()
    local n_step = self.cur_batch_x:size()[1]
    for i = 1, 6 do
	    self.grads[i]:mul(self.options.momentum / (-self.options.alpha))
    end
	reset(self.grad_h_decoder)
    reset(self.grad_decoder_from_context[n_step])
    reset(self.grad_encoder_from_context_left)
    reset(self.grad_encoder_from_context_right)
    reset(self.grad_em)
    
	local derr = transfer2gpu(torch.ones(1))
	for i = n_step, 1, -1 do
        local input_label = self.cur_batch_y[i]
        local output_label = self.cur_batch_y[i+1]
        local grad_from_output = self.models[4][i]:backward({self.history_decoder[i][2 * self.options.layers], output_label}, derr)[1]
        self.grad_h_decoder[2 * self.options.layers]:add(grad_from_output):add(self.grad_decoder_from_context[i])
        local grads = self.models[3][i]:backward({input_label, self.tmp_hist_decoder[i - 1], self.context[i]}, self.grad_h_decoder)
		replace(self.grad_h_decoder, grads[2])
        replace(self.grad_of_context[i], grads[3])
        for j = 1, self.options.batch_size do
			if self.cur_batch_mask_y[i+1][j] == 0 then
                for k = 1, 2 * self.options.layers do
                    self.grad_h_decoder[k][j]:zero() -- clear the gradient at the end -- unnecessary
                end
                self.grad_of_context[i][j]:zero() -- clear context the gradient at the end (padding)  -- unnecessary
			end
		end
        local attention_grads = self.models[5][i]:backward({self.hidden_decoder_tmp[i], self.hidden_encoder_left, self.hidden_encoder_right, self.mask_x}, self.grad_of_context[i])
        self.grad_decoder_from_context[i-1] = attention_grads[1]:sum(1)[1]
        for x_i = 1, n_step do
            local right_model_index = n_step - x_i + 1
            self.grad_encoder_from_context_left[x_i]:add(attention_grads[2][x_i])   -- batch * hidden   +  batch * 1 * hidden
            self.grad_encoder_from_context_right[right_model_index]:add(attention_grads[3][x_i])   -- batch * hidden   +  batch * 1 * hidden
        end
    end
    self.grad_h_decoder[2 * self.options.layers]:add(self.grad_decoder_from_context[0])
    replace(self.grad_h_encoder_right, self.grad_h_decoder)
    reset(self.grad_h_encoder_left)
	for i = n_step, 1, -1 do
        local input_pos_right = n_step - i + 1
		local input_left = self.embeddings[i]
        local input_right = self.embeddings[input_pos_right]
        for j = 1, self.options.batch_size do
            --[[if self.cur_batch_mask_x[i][j] == 1 then
                if (i == n_step) or (i < n_step and self.cur_batch_mask_x[i+1][j] == 0) then
                    for k = 1, 2 * self.options.layers do
                        self.grad_h_encoder_left[k][j]:add(self.grad_h_decoder[k][j])
                    end
                end
            end--]]
            if self.cur_batch_mask_x[i][j] == 0 then
                self.grad_encoder_from_context_left[i][j]:zero() -- clear the gradient at the end (padding)
            end
            if self.cur_batch_mask_x[input_pos_right][j] == 0 then
                self.grad_encoder_from_context_right[i][j]:zero() -- clear the gradient at the end (padding)
            end
        end
        self.grad_h_encoder_left[2 * self.options.layers]:add(self.grad_encoder_from_context_left[i])
        self.grad_h_encoder_right[2 * self.options.layers]:add(self.grad_encoder_from_context_right[i])
		local grads_encoder_left = self.models[1][i]:backward({input_left, self.tmp_hist_encoder_left[i - 1]}, self.grad_h_encoder_left)
		local grads_encoder_right = self.models[2][i]:backward({input_right, self.tmp_hist_encoder_right[i - 1]}, self.grad_h_encoder_right)
		local grad_h_encoder_left = grads_encoder_left[2]
		local grad_h_encoder_right = grads_encoder_right[2]
        self.grad_em[i]:add(grads_encoder_left[1]) -- grad of embedding
        self.grad_em[input_pos_right]:add(grads_encoder_right[1]) -- grad of embedding
		replace(self.grad_h_encoder_left, grad_h_encoder_left)
		replace(self.grad_h_encoder_right, grad_h_encoder_right)
        for j = 1, self.options.batch_size do
            if self.cur_batch_mask_x[i][j] == 0 then
                for k = 1, 2 * self.options.layers do
                    self.grad_h_encoder_left[k][j]:zero() -- clear the gradient
                end
            end
            if input_pos_right < n_step and self.cur_batch_mask_x[input_pos_right+1][j] == 0 then
                for k = 1, 2 * self.options.layers do
                    self.grad_h_encoder_right[k][j]:zero()
                end
            end
        end
	end

    for i = 1, n_step do
        local input = self.cur_batch_x[i]
        self.models[6][i]:backward({input}, self.grad_em[i])
    end

    for i = 1, 6 do
        self.grads[i]:mul(-self.options.alpha)
        --[[local grad_norm = self.grads[i]:norm()
        if grad_norm > self.options.max_norm then
            self.grads[i]:mul(self.options.max_norm / grad_norm) --clip gradient
        end--]]
        if self.options.beta > 0 then
            self.params[i]:mul(1 - self.options.beta)
        end
        self.params[i]:add(self.grads[i])
    end
end

function LSTM:forward_testing(probs, attention_probs)
    local n_step = self.cur_batch_x:size()[1]
    local err, xx_encoder_left, xx_encoder_right, xx_decoder, xx_decoder2context, context
    reset(self.test_history_encoder_left[0])
    reset(self.test_history_encoder_right[0])
    self.test_hidden_encoder_left = {}
    self.test_hidden_encoder_right = {}
    --reset(self.test_history_decoder[0])
    local wcount = 0
    for i = 1, n_step do
        local input = self.cur_batch_x[i]
        self.embeddings[i] = self.models[6][i]:forward(input)
    end
	for i = 1, n_step do
        local r = n_step - i + 1
		local input_left = self.embeddings[i]
        local input_right = self.embeddings[r]
        for j = 1, self.options.test_batch_size do
            if self.cur_batch_mask_x[i][j] == 0 then
                for k = 1, 2 * self.options.layers do
                    self.test_history_encoder_left[i-1][k][j]:zero()
                end
            end
            if r < n_step and  self.cur_batch_mask_x[r+1][j] == 0 then
                for k = 1, 2 * self.options.layers do
                    self.test_history_encoder_right[i-1][k][j]:zero()
                end
            end
        end

        xx_encoder_left = self.models[1][1]:forward({input_left, self.test_history_encoder_left[i - 1]})
        xx_encoder_right = self.models[2][1]:forward({input_right, self.test_history_encoder_right[i - 1]})
		replace(self.test_history_encoder_left[i], xx_encoder_left)
		replace(self.test_history_encoder_right[i], xx_encoder_right)
        replace(self.test_hidden_encoder_cache_left[i], xx_encoder_left[2 * self.options.layers])
        replace(self.test_hidden_encoder_cache_right[i], xx_encoder_right[2 * self.options.layers])
        for j = 1, self.options.test_batch_size do
            --[[if self.cur_batch_mask_x[i][j] == 1 then
                if (i == n_step) or (i < n_step and self.cur_batch_mask_x[i+1][j] == 0) then
                    for k = 1, 2 * self.options.layers do
                        replace(self.test_history_encoder2decoder[k][j], xx_encoder_left[k][j])
                    end
                end
            end--]]
            if self.cur_batch_mask_x[i][j] == 0 then
                self.test_hidden_encoder_cache_left[i][j]:zero()
            end
            if self.cur_batch_mask_x[r][j] == 0 then
                self.test_hidden_encoder_cache_right[i][j]:zero()
            end
        end
        self.test_hidden_encoder_left[i] = self.test_hidden_encoder_cache_left[i]:view(self.options.test_batch_size, 1, self.options.hidden_size[self.options.layers]) -- view
        self.test_hidden_encoder_right[r] = self.test_hidden_encoder_cache_right[i]:view(self.options.test_batch_size, 1, self.options.hidden_size[self.options.layers]) -- view
	end
    for i = 1, self.options.beam_size do
        --replace(self.test_cur_hist_decoder[i], self.test_history_encoder2decoder)
        replace(self.test_cur_hist_decoder[i], self.test_history_encoder_right[n_step])
    end

    local beam_size = self.options.beam_size
    local beams, scores, errs = {[1]={}}, {[1]={}}, {[1]={}} -- index in beam_size
    for j = 1, self.options.test_batch_size do
        beams[1][j] = {[0]=self.cur_batch_y[1][j]} -- <eos>
        scores[1][j] = 0
        errs[1][j] = 0 --transfer2gpu(torch.zeros(1))
    end
    local results = {}
    local step = 1 -- self.test_tmp_hist_decoder
    local input_template = torch.zeros(self.options.test_batch_size)
    while step <= n_step do -- force output length = input length --beam_size ~= 0 do
        local output_label = self.cur_batch_y[step+1]
        local new_beams, new_scores, new_errs, last_beams = {}, {}, {}, {}
        local idx = {}
        for j = 1, self.options.test_batch_size do
            new_beams[j], new_scores[j], new_errs[j], last_beams[j] = {}, {}, {}, {}
            idx[j] = 1
        end
        for i = 1, #beams do
            local input = input_template
            for j = 1, self.options.test_batch_size do
                input[j] = beams[i][j][step-1]
            end
            input = transfer2gpu(input)
            xx_decoder2context = self.test_cur_hist_decoder[i][2 * self.options.layers]:repeatTensor(n_step, 1, 1)
            context = self.models[5][1]:forward({xx_decoder2context, self.test_hidden_encoder_left, self.test_hidden_encoder_right, self.mask_x}) -- attention
            --print(attention_probs[1].output:float())
            xx_decoder = self.models[3][1]:forward({input, self.test_cur_hist_decoder[i], context})
            local err = self.models[4][1]:forward({xx_decoder[2 * self.options.layers], output_label})
            replace(self.test_next_hist_decoder[i], xx_decoder)
            for j = 1, self.options.test_batch_size do
                if self.cur_batch_mask_y[step+1][j] == 0 then  -- <null> or the end <eos>
                    for k = 1, 2 * self.options.layers do
                        self.test_next_hist_decoder[i][k][j]:zero()
                    end
                end
            end
            local prob = probs[1].output:float()
            if step <= n_step then  -- avoid the case that input sequence length doesn't match output seuqence length
                for j = 1, self.options.test_batch_size do
                    prob[j][self.options.vocab:get_output('<eos>')] = -1000000
                end
            end
            local top_local_beams_id_logprob_batch, top_local_beams_id_batch = prob:topk(beam_size, true, true)
            --top_local_beams_id_logprob_batch = top_local_beams_id_logprob_batch:float()
            --top_local_beams_id_batch = top_local_beams_id_batch:float()
            for j = 1, self.options.test_batch_size do
                if self.cur_batch_mask_y[step+1][j] == 0 then  -- <null> or the end <eos>
                    new_errs[j][i] = errs[i][j]
                    local tmp_size = 1
                    if #scores == 1 then
                        tmp_size = beam_size
                    end
                    for b = 1, tmp_size do
                        new_scores[j][idx[j]] = scores[i][j]
                        new_beams[j][idx[j]] = shallowcopy(beams[i][j])
                        new_beams[j][idx[j]][step] = self.options.vocab:get_output('<null>') 
                        last_beams[j][idx[j]] = i
                        idx[j] = idx[j] + 1
                    end
                    for b = tmp_size+1, beam_size do
                        new_scores[j][idx[j]] = -10000
                        idx[j] = idx[j] + 1
                    end
                else
                    new_errs[j][i] = errs[i][j] - prob[j][output_label[j]]
                    local top_local_beams_id, top_local_beams_id_logprob = top_local_beams_id_batch[j], top_local_beams_id_logprob_batch[j]
                    --print(table.concat(top_local_beams_id,' '))
                    --print(table.concat(top_local_beams_id_logprob,' '))
                    for b = 1, beam_size do
                        local sample_output = top_local_beams_id[b]
                        local sample_score = top_local_beams_id_logprob[b]
                        new_scores[j][idx[j]] = scores[i][j] + sample_score 
                        new_beams[j][idx[j]] = shallowcopy(beams[i][j])
                        new_beams[j][idx[j]][step] = sample_output
                        last_beams[j][idx[j]] = i
                        idx[j] = idx[j] + 1
                    end
                end
            end
        end
        local new_beams_top_beams_logprob_batch, new_beams_top_beams_id_batch = torch.Tensor(new_scores):topk(beam_size, true, true)
        for j = 1, self.options.test_batch_size do
            local new_beams_top_beams_id, new_beams_top_beams_logprob = new_beams_top_beams_id_batch[j], new_beams_top_beams_logprob_batch[j]
            for i = 1, beam_size do
                local new_beams_idx = new_beams_top_beams_id[i]
                if beams[i] == nil then
                    beams[i], errs[i], scores[i] = {}, {}, {}
                end
                beams[i][j] = new_beams[j][new_beams_idx]
                for k = 1, 2 * self.options.layers do
                    replace(self.test_cur_hist_decoder[i][k][j], self.test_next_hist_decoder[last_beams[j][new_beams_idx]][k][j])
                end
                errs[i][j] = new_errs[j][last_beams[j][new_beams_idx]]--:clone()
                scores[i][j] = new_beams_top_beams_logprob[i]
            end
        end
        step = step + 1
    end
    -- get results
    local top_1best = torch.Tensor(beams[1]):t()
    for j = 1, self.options.test_batch_size do
        --self.err:add(errs[1][j]:sum())
        self.test_err = self.test_err + errs[1][j]
    end
    for i = 1, n_step do 
        local cur_output_label = top_1best[i]
        local input = self.cur_batch_x[i] 
        local output_label = self.cur_batch_y[i+1]
        local eos = {}

        for j = 1, self.options.test_batch_size do
            if self.cur_batch_mask_y[i+1][j] == 0 then
                eos[j] = -1
            else
                wcount = wcount + 1
                if i == n_step or ( i < n_step and self.cur_batch_mask_y[i+2][j] == 0)then
                    eos[j] = 1
                else
                    eos[j] = 0
                end
            end
        end
        self.F1:get_batch(cur_output_label, input, output_label, 1, eos)
    end
    return wcount
end

function LSTM:train_one_epoch(train)
	self.reader = DataReader(train, self.options.batch_size, self.options.vocab, self.options.word_win_left, self.options.word_win_right)
	--reset(self.history[self.options.mx]) -- copied to self.history[0] in forward()
    for i = 1, 6 do
	    self.grads[i]:zero()
    end
    self.err:zero()
	local len = 0
    local ce = 0
	local begin_time = torch.tic()
    local read_time, trf_time, forward_time, bp_time = 0, 0, 0, 0
    local is_there_end_eos = false
    local is_input_start_from_left = true -- true : the padding is in the right side of input;       false : the padding is in the left side of input
	while true do
        local beg_time = torch.tic()
		self.cur_batch_x, self.cur_batch_mask_x, self.cur_batch_y, self.cur_batch_mask_y = self.reader:get_batch_4train(is_input_start_from_left, is_there_end_eos)
        --print(self.cur_batch_x)
        --print(self.cur_batch_mask_x)
        --print(self.cur_batch_y)
        --print(self.cur_batch_mask_y)
        --os.exit()
        read_time = read_time + torch.toc(beg_time)
		if self.cur_batch_x == nil then
			break
		end
		len = len + torch.sum(self.cur_batch_mask_x) --count_words(self.options.vocab, self.cur_batch)
        beg_time = torch.tic()
		self.cur_batch_x = transfer2gpu(self.cur_batch_x)
		self.cur_batch_y = transfer2gpu(self.cur_batch_y)
		self.mask_x = transfer2gpu(self.cur_batch_mask_x:t())
		--self.cur_batch_mask_y = transfer2gpu(self.cur_batch_mask_y)
        trf_time = trf_time + torch.toc(beg_time)
        beg_time = torch.tic()
		self:forward_training()
        forward_time = forward_time + torch.toc(beg_time)
        beg_time = torch.tic()
		self:backward()
        bp_time = bp_time + torch.toc(beg_time)
	end
    ce = self.err[1]
    --print(read_time..' '..trf_time..' '..forward_time..' '..bp_time)
    local elapsed_time = torch.toc(begin_time) / 60
    print('trained words = ' .. len .. ', CE = ' .. string.format('%.3f', ce / len) .. ', elapsed time = ' .. string.format('%.1f', elapsed_time) .. ' mins.')
    len = 0
    ce = 0
    io.stdout:flush()
    collectgarbage()
end

function LSTM:evaluate(data, outputfile)
    if outputfile then
        outputfile = io.open(outputfile, 'w')
    end
    self.F1 = F1(self.options.vocab, self.options.test_batch_size, outputfile, self.options.word_win_left) 
    local ce = 0
	local len = 0
	local probs = {}
    local attention_probs = {}
	for i = 1, self.options.my do
		probs[i] = find_module(self.models[4][i], "log_prob")
		attention_probs[i] = find_module(self.models[5][i], "probs")
	end

    local begin_time = torch.tic()
    local read_time, trf_time, forward_time = 0, 0, 0
	self.reader = DataReader(data, self.options.test_batch_size, self.options.vocab, self.options.word_win_left, self.options.word_win_right)
    self.test_err = 0
    --self.batch_output_vocab = transfer2gpu(self.reader:get_output_vocab_4test()) -- for beam search in test
    for i = 1, 6 do
    	disable_dropout(self.models[i])
    end
    local is_there_end_eos = false
    local is_input_start_from_left = true -- true : the padding is in the right side of input;       false : the padding is in the left side of input
	while true do
        local beg_time = torch.tic()
		self.cur_batch_x, self.cur_batch_mask_x, self.cur_batch_y, self.cur_batch_mask_y = self.reader:get_batch_4train(is_input_start_from_left, is_there_end_eos)
        --print(self.cur_batch_x)
        --print(self.cur_batch_y)
		if self.cur_batch_x == nil then
			break
		end
        read_time = read_time + torch.toc(beg_time)
        beg_time = torch.tic()
		self.cur_batch_x = transfer2gpu(self.cur_batch_x)
		self.cur_batch_y = transfer2gpu(self.cur_batch_y)
        --self.mask_x = transfer2gpu(torch.ones(1, self.cur_batch_x:size()[1]))
		self.mask_x = transfer2gpu(self.cur_batch_mask_x:t())
        trf_time = trf_time + torch.toc(beg_time)
        beg_time = torch.tic()
		len = len + self:forward_testing(probs, attention_probs)
        forward_time = forward_time + torch.toc(beg_time)
	end
    ce = self.test_err
    --print(read_time..' '..trf_time..' '..forward_time)
    for i = 1, 6 do
	    enable_dropout(self.models[i])
    end
    return len, ce/len, self.F1:get_metric()
end

function LSTM:load_model(input_file)
    self.core_model = {}
    for i = 1, 6 do
        self.core_model[i] = torch.load(input_file .. '_' .. i)
        self.core_model[i] = transfer2gpu(self.core_model[i])
    end
end

function LSTM:restore(model)
	self:load_model(model)
	self.params, self.grads = {}, {}
    self.models = {}
    for i = 1, 6 do
        self.params[i], self.grads[i] = self.core_model[i]:getParameters()
    end
    self.models[1] = make_recurrent(self.core_model[1], self.options.mx)
    self.models[2] = make_recurrent(self.core_model[2], self.options.mx)
    self.models[3] = make_recurrent(self.core_model[3], self.options.my)
    self.models[4] = make_recurrent(self.core_model[4], self.options.my)
    self.models[5] = make_recurrent(self.core_model[5], self.options.my)
    self.models[6] = make_recurrent(self.core_model[6], self.options.mx)
	collectgarbage()
end

function LSTM:save_model(output_file)
    for i = 1, 6 do
	    torch.save(output_file .. '_' .. i, self.core_model[i])
    end
end
