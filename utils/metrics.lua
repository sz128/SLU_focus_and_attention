local stringx = require 'pl.stringx'

local F1 = torch.class('F1')

function F1:__init(vocab, batch_size, outputfile, word_win_left, nbest)
    self.TP = 0
    self.FP = 0
    self.FN = 0
    self.TN = 0
    self.nbest = nbest or 1
    self.vocab = vocab
    self.vocab_size = vocab:vocab_size()
    self.label_cache, self.pred_cache, self.input_cache = {}, {}, {}
    self.nbest_pred_cache = {}
    self.nbest_pred_score = {}
    for i = 1, batch_size do
        self.label_cache[i] = {'O'}
        self.pred_cache[i] = {'O'}
        self.input_cache[i] = {'BOS'}
        self.nbest_pred_cache[i] = {}
        self.nbest_pred_score[i] = {}
        for j = 1, self.nbest do
            self.nbest_pred_cache[i][j] = {'O'}
            self.nbest_pred_score[i][j] = 0
        end
    end
    self.label_chunks, self.pred_chunks = {}, {}
    self.outputfile = outputfile
    assert(word_win_left>=0)
    self.word_pos = word_win_left+1
    self.trf_time, self.cal_time, self.write_time = 0, 0, 0
end

function F1:get_metric()
    --print(self.trf_time..'#'..self.cal_time..'#'..self.write_time)
    --print(self.TP .. ' ' .. self.FP .. ' ' .. self.FN)
    if self.TP == 0 then
        return {['precision']=0, ['recall']=0, ['F1']=0}
    end
    return {['precision']=self.TP/(self.TP+self.FP), ['recall']=self.TP/(self.TP+self.FN), ['F1']=2*self.TP/(2*self.TP+self.FP+self.FN)}
end

function F1:get_batch(cur_output_labels, input, labels, pos, eos, top_scores)
    for i = 1, labels:size()[1] do --batch size
        --if pos ~= -1 then
        if eos[i] ~= -1 then
            self.label_cache[i][#self.label_cache[i]+1] = self.vocab:inv_get_output(labels[i])
            local beg_time = torch.tic()
            if self.nbest == 1 then
               self.pred_cache[i][#self.pred_cache[i]+1] = self.vocab:inv_get_output(cur_output_labels[i])
            else --if self.nbest > 1 then
               self.pred_cache[i][#self.pred_cache[i]+1] = self.vocab:inv_get_output(cur_output_labels[i][1])
               for j = 1, self.nbest do
                   self.nbest_pred_cache[i][j][#self.nbest_pred_cache[i][j]+1] = self.vocab:inv_get_output(cur_output_labels[i][j])
                   self.nbest_pred_score[i][j] = top_scores[i][j]
               end
            end
            self.input_cache[i][#self.input_cache[i]+1] = self.vocab:inv_get_input(input[i][self.word_pos])
            self.cal_time = self.cal_time + torch.toc(beg_time)
        end
        --if pos == 1 then
        if eos[i] == 1 then
            if #self.label_cache[i] ~= 1 then
                local beg_time = torch.tic()
                self.label_cache[i][#self.label_cache[i]+1] = 'O'
                self.pred_cache[i][#self.pred_cache[i]+1] = 'O'
                self.input_cache[i][#self.input_cache[i]+1] = 'EOS'
                if self.nbest > 1 then
                    for j = 1, self.nbest do
                        self.nbest_pred_cache[i][j][#self.nbest_pred_cache[i][j] + 1] = 'O'
                    end
                end
                if self.outputfile ~= nil then
                    if self.nbest == 1 then
                        for k = 1, #self.input_cache[i] do
                            self.outputfile:write(self.input_cache[i][k] .. ' ' .. self.label_cache[i][k] .. ' ' .. self.pred_cache[i][k] .. '\n')
                            --print(self.input_cache[i][k] .. ' ' .. self.label_cache[i][k] .. ' ' .. self.pred_cache[i][k] .. '\n')
                        end
                        self.outputfile:write('\n')
                    else
                        for j = 1, self.nbest do
                            self.outputfile:write('# ' .. (j-1) .. ' ' .. self.nbest_pred_score[i][j] .. '\n')
                            for k = 1, #self.input_cache[i] do
                                self.outputfile:write(self.input_cache[i][k] .. ' ' .. self.label_cache[i][k] .. ' ' .. self.nbest_pred_cache[i][j][k] .. '\n')
                            end
                            self.outputfile:write('\n')
                        end
                    end
                end
                self.label_chunks = self:get_chunks(self.label_cache[i])
                self.pred_chunks = self:get_chunks(self.pred_cache[i])
                self.label_cache[i] = {'O'}
                self.pred_cache[i] = {'O'}
                self.input_cache[i] = {'BOS'}
                self.nbest_pred_cache[i] = {}
                self.nbest_pred_score[i] = {}
                for j = 1, self.nbest do
                    self.nbest_pred_cache[i][j] = {'O'}
                    self.nbest_pred_score[i][j] = 0
                end
                for key,value in pairs(self.pred_chunks) do
                    if self.label_chunks[key] then
                        self.TP = self.TP + 1
                    else
                        self.FP = self.FP + 1
                    end
                end
                for key,value in pairs(self.label_chunks) do
                    if self.pred_chunks[key] == nil then
                        self.FN = self.FN + 1
                    end
                end
                self.write_time = self.write_time + torch.toc(beg_time)
            end
        end
    end
end

function F1:get_batch_prob(probs, input, labels, pos, eos, rankingloss)
    probs = probs:float()
    for i = 1, probs:size()[1] do --batch size
        --if pos ~= -1 then
        if eos[i] ~= -1 then
            self.label_cache[i][#self.label_cache[i]+1] = self.vocab:inv_get_output(labels[i])
            local beg_time = torch.tic()
            --local probs_table = {}
            --for k = 1, self.vocab_size['output'] do
            --    probs_table[k] = -probs[i][k]
            --end
            self.trf_time = self.trf_time + torch.toc(beg_time)
            beg_time = torch.tic()
            top_idx, top_prob = self:get_top(probs[i],1)
            --print(top_idx[1]..' '..top_prob[1]..';'..probs[i][1])
            if rankingloss and top_prob[1] < 0 then
                self.pred_cache[i][#self.pred_cache[i]+1] = 'O'
            else
                self.pred_cache[i][#self.pred_cache[i]+1] = self.vocab:inv_get_output(top_idx[1])
            end
            self.input_cache[i][#self.input_cache[i]+1] = self.vocab:inv_get_input(input[i][self.word_pos])
            self.cal_time = self.cal_time + torch.toc(beg_time)
        end
        --if pos == 1 then
        if eos[i] == 1 then
            if #self.label_cache[i] ~= 1 then
                local beg_time = torch.tic()
                self.label_cache[i][#self.label_cache[i]+1] = 'O'
                self.pred_cache[i][#self.pred_cache[i]+1] = 'O'
                self.input_cache[i][#self.input_cache[i]+1] = 'EOS'
                if self.outputfile ~= nil then
                    for k = 1, #self.input_cache[i] do
                        self.outputfile:write(self.input_cache[i][k] .. ' ' .. self.label_cache[i][k] .. ' ' .. self.pred_cache[i][k] .. '\n')
                        --print(self.input_cache[i][k] .. ' ' .. self.label_cache[i][k] .. ' ' .. self.pred_cache[i][k] .. '\n')
                    end
                    self.outputfile:write('\n')
                end
                self.label_chunks = self:get_chunks(self.label_cache[i])
                self.pred_chunks = self:get_chunks(self.pred_cache[i])
                self.label_cache[i] = {'O'}
                self.pred_cache[i] = {'O'}
                self.input_cache[i] = {'BOS'}
                for key,value in pairs(self.pred_chunks) do
                    if self.label_chunks[key] then
                        self.TP = self.TP + 1
                    else
                        self.FP = self.FP + 1
                    end
                end
                for key,value in pairs(self.label_chunks) do
                    if self.pred_chunks[key] == nil then
                        self.FN = self.FN + 1
                    end
                end
                self.write_time = self.write_time + torch.toc(beg_time)
            end
        end
    end
end

function F1:get_top(probs,nbest)
    local top_idx, top_prob = {[1]=0}, {[1]=-100}
    for i = 1, self.vocab_size['output'] do
        if probs[i] > top_prob[1] then
            top_idx[1] = i
            top_prob[1] = probs[i]
        end
    end
    return top_idx, top_prob
end

function F1:get_chunks(labels)
    local chunks = {}
    local start_idx, end_idx = 0, 0
    local chunkStart, chunkEnd = false, false
    local prevTag, prevType = 'O','O'
    local Tag, Type = 'O','O'
    local nextTag, nextType = 'O','O'
    for idx = 2, #labels-1 do
        chunkStart, chunkEnd = false, false
        if labels[idx-1] ~= 'O' then
            prevTag, prevType = unpack(stringx.split(labels[idx-1],'-'))
        else
            prevTag, prevType = 'O', 'O'
        end
        if labels[idx] ~= 'O' then
            Tag, Type = unpack(stringx.split(labels[idx],'-'))
        else
            Tag, Type = 'O', 'O'
        end
        if labels[idx+1] ~= 'O' then
            nextTag, nextType = unpack(stringx.split(labels[idx+1],'-'))
        else
            nextTag, nextType = 'O', 'O'
        end

        if Tag == 'B' or (prevTag == 'O' and Tag == 'I') then
            chunkStart = true
        end
        if Tag ~= 'O' and prevType ~= Type then
            chunkStart = true
        end

        if (Tag == 'B' or Tag == 'I') and (nextTag == 'B' or nextTag == 'O') then
            chunkEnd = true
        end
        if Tag ~= 'O' and Type ~= nextType then
            chunkEnd = true
        end

        if chunkStart then
            start_idx = idx
        end
        if chunkEnd then
            end_idx = idx
            chunks[start_idx..'-'..end_idx..'-'..Type] = true
            start_idx,end_idx = 0,0
        end
    end
    return chunks
end

