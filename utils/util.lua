local stringx = require 'pl.stringx'

function get_gpu_compute_rest(deviceCount)
    local device_compute_rest = {}
    for i = 1, deviceCount do
        device_compute_rest[i] = 0.0
        local result = io.popen('nvidia-smi -q -d UTILIZATION -i ' .. i-1, 'r')
        local last_line = ""
        for line in result:lines() do
            local tmp = line:match("^%s*(.-)%s*$")
            if last_line == "Utilization" and string.sub(tmp, 1, string.len('Gpu')) == 'Gpu' then
                local gpu = tmp:match('%d+')
                device_compute_rest[i] = 1 - tonumber(gpu)/100
                break
            end
            last_line = tmp
        end
        result:close()
    end
    return device_compute_rest
end
function chooseGPU()
	local cnt = cutorch.getDeviceCount()
    local device_compute_rest = get_gpu_compute_rest(cnt)
    --print(table.concat(device_compute_rest,' '))
    local device_first_level = {}
    local device_second_level = {}
    local device_third_level = {}
	for i = 1, cnt do
		freeMem, totalMem = cutorch.getMemoryUsage(i)
        memRestRate = freeMem / totalMem
        --print(totalMem/1048576 .. ' ' .. freeMem/1048576 .. ' ' .. freeMem / totalMem)
		if memRestRate > 0.3 and device_compute_rest[i] > 0.5 then
			device_first_level[i] = device_compute_rest[i]
        elseif memRestRate > 0.1 and memRestRate <= 0.3 and device_compute_rest[i] > 0.5 then
            device_second_level[i] = memRestRate
        else
            device_third_level[i] = memRestRate + device_compute_rest[i]
		end
	end
    local best = max_table(device_first_level)
    if best > 0 then
	    print("INFO: Using the first level GPU card")
        return best
    else
        local best = max_table(device_second_level)
        if best > 0 then
            print("WARNING: Using the second level GPU card")
            return best
        else
            print("WARNING: Using the third level GPU card")
            return max_table(device_third_level)
        end
    end
	print("WARNING: No Free GPU is find. Using the primary GPU card")
	return 1
end

function transfer2gpu(module)
	return module:cuda()
end

function readline(file, NE_flag)
	assert(file:isQuiet())
    local input, NE, output, tmp = {}, {}, {}, {}
	local line = file:readString('*l')
    if file:hasError() then
        return nil
    end
    while line ~= '' do
        if line ~= 'EOS O' and line ~= 'BOS O' then
            tmp = stringx.split(line)
            local word,label = tmp[1],tmp[#tmp]
            --print(line..' '..word..' '..label)
            input[#input+1] = word
            output[#output+1] = label
            if NE_flag then
                NE[#NE+1] = tmp[2]
            end
        end
        line = file:readString('*l')
        if file:hasError() then
            return nil
        end
    end
    if NE_flag then
	    return input, NE, output 
    else
        return input, output
    end
end

function readline1(file)
	assert(file:isQuiet())
	local line = file:readString('*l') -- end-of-line character is omitted!
	if file:hasError() then
		return nil
	end
	local data = stringx.split(line)
	return data 
end

function replace(to, from)
	if type(to) == 'table' then
		assert (#from == #to)
		for i = 1, #from do
			to[i]:copy(from[i])
		end
	else
		to:copy(from)
	end
end

function reset(x)
	if type(x) == 'table' then
		for i = 1, #x do
			x[i]:zero()
		end
	else
		x:zero()
	end
end

function clone(x)
	local buffer = torch.MemoryFile('rw'):binary()
	buffer:writeObject(x)
	buffer:seek(1)
	local y = buffer:readObject() -- clone via memory file
	buffer:close()
	return y
end

function count_words(vocab, input)
	local cnt = 0
	assert(type(input) == 'userdata')
	if input:dim() == 1 then
		for i = 1, input:size(1) do
			if not vocab:is_null(input[i]) then
				cnt = cnt + 1
			end
		end
	elseif input:dim() == 2 then
		for i = 2, input:size(1) do
			for j = 1, input:size(2) do
				if not vocab:is_null(input[i][j]) then
					cnt = cnt + 1
				end
			end
		end
	end
	return cnt
end
					

function make_recurrent(net, times)
	local clones = {}
	local params, grads = net:parameters() -- here use parameters() instead of getParameters() because getParameters() returns flattened tables
	if params == nil then
		params = {}
	end
	local buffer = torch.MemoryFile('w'):binary()
	buffer:writeObject(net)
	for t = 1, times do
		local reader = torch.MemoryFile(buffer:storage(), 'r'):binary()
		local clone_net = reader:readObject()
		reader:close()
		local clone_params, clone_grads = clone_net:parameters()
		for i = 1, #params do
			clone_params[i]:set(params[i])
			clone_grads[i]:set(grads[i])
		end
		clones[t] = clone_net
		collectgarbage()
	end
	buffer:close()
	return clones
end

-- Related to inner design of nngraph, confused.
function disable_dropout(node)
	if type(node) == 'table' and node.__typename == nil then
		for i = 1, #node do
			node[i]:apply(disable_dropout)
		end
		return
	end
	if string.match(node.__typename, "Dropout") then
		node.train = false
	end
end

function enable_dropout(node)
	if type(node) == 'table' and node.__typename == nil then
		for i = 1, #node do
			node[i]:apply(enable_dropout)
		end
		return
	end
	if string.match(node.__typename, "Dropout") then
		node.train = true
	end
end

function random_seed(seed, strategy)
	torch.manualSeed(seed)
	cutorch.manualSeed(seed)
	torch.zeros(1, 1):cuda():uniform()
    if strategy ~= nil and strategy ~= 'false' then
        --local rand_file = torch.DiskFile('.randfile', 'w'):binary()
        for i = 1, 100000 do
            local arr = torch.rand(100):float()
            --rand_file:writeFloat(arr:storage())
        end
        --rand_file:close()
    end
end

function find_module(model, pred)
	for _, node in ipairs(model.forwardnodes) do
		if stringx.startswith(node:graphNodeName(), pred) then
			return node.data.module
		end
	end
end

function shallowcopy(orig)
    local orig_type = type(orig)
    local copy
    if orig_type == 'table' then
        copy = {}
        for orig_key, orig_value in pairs(orig) do
            copy[orig_key] = orig_value
        end
    else -- number, string, boolean, etc
        copy = orig
    end
    return copy
end

function combine_all_parameters(...)
    --[[ like module:getParameters, but operates on many modules ]]--

    -- get parameters
    local networks = {...}
    local parameters = {}
    local gradParameters = {}
    for i = 1, #networks do
        local net_params, net_grads = networks[i]:parameters()

        if net_params then
            for _, p in pairs(net_params) do
                parameters[#parameters + 1] = p
            end
            for _, g in pairs(net_grads) do
                gradParameters[#gradParameters + 1] = g
            end
        end
    end

    local function storageInSet(set, storage)
        local storageAndOffset = set[torch.pointer(storage)]
        if storageAndOffset == nil then
            return nil
        end
        local _, offset = unpack(storageAndOffset)
        return offset
    end

    -- this function flattens arbitrary lists of parameters,
    -- even complex shared ones
    local function flatten(parameters)
        if not parameters or #parameters == 0 then
            return torch.Tensor()
        end
        local Tensor = parameters[1].new

        local storages = {}
        local nParameters = 0
        for k = 1,#parameters do
            local storage = parameters[k]:storage()
            if not storageInSet(storages, storage) then
                storages[torch.pointer(storage)] = {storage, nParameters}
                nParameters = nParameters + storage:size()
            end
        end

        local flatParameters = Tensor(nParameters):fill(1)
        local flatStorage = flatParameters:storage()

        for k = 1,#parameters do
            local storageOffset = storageInSet(storages, parameters[k]:storage())
            parameters[k]:set(flatStorage,
                storageOffset + parameters[k]:storageOffset(),
                parameters[k]:size(),
                parameters[k]:stride())
            parameters[k]:zero()
        end

        local maskParameters=  flatParameters:float():clone()
        local cumSumOfHoles = flatParameters:float():cumsum(1)
        local nUsedParameters = nParameters - cumSumOfHoles[#cumSumOfHoles]
        local flatUsedParameters = Tensor(nUsedParameters)
        local flatUsedStorage = flatUsedParameters:storage()

        for k = 1,#parameters do
            local offset = cumSumOfHoles[parameters[k]:storageOffset()]
            parameters[k]:set(flatUsedStorage,
                parameters[k]:storageOffset() - offset,
                parameters[k]:size(),
                parameters[k]:stride())
        end

        for _, storageAndOffset in pairs(storages) do
            local k, v = unpack(storageAndOffset)
            flatParameters[{{v+1,v+k:size()}}]:copy(Tensor():set(k))
        end

        if cumSumOfHoles:sum() == 0 then
            flatUsedParameters:copy(flatParameters)
        else
            local counter = 0
            for k = 1,flatParameters:nElement() do
                if maskParameters[k] == 0 then
                    counter = counter + 1
                    flatUsedParameters[counter] = flatParameters[counter+cumSumOfHoles[k]]
                end
            end
            assert (counter == nUsedParameters)
        end
        return flatUsedParameters
    end

    -- flatten parameters and gradients
    local flatParameters = flatten(parameters)
    local flatGradParameters = flatten(gradParameters)

    -- return new flat vector that contains all discrete parameters
    return flatParameters, flatGradParameters
end

