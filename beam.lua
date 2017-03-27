require 'nn'
require 'string'
require 'hdf5'
require 'nngraph'

require 'data.lua'
require 'models.lua'
require 'util.lua'
require 'logging'

ENTROPY = 0
GOLD_SIZE = 0
MUTE = true

stringx = require('pl.stringx')

cmd = torch.CmdLine()
cmd:option('-log_path', '', [[Logging path]])

-- check attn options
cmd:option('-num_argmax', 0, [[Change multisampling to do different number of argmax]])
cmd:option('-view_attn', 0, [[View attention weights at each time step]])
cmd:option('-print_attn', 0, [[Print attention weights]])
cmd:option('-print_sent_attn', 1, [[Print sentence attention instead of all attn]])

cmd:option('-no_pad', 1, [[No pad format for data]])
cmd:option('-no_pad_sent_l', 10, [[Number of `sentences` we have for a doc]])
cmd:option('-repeat_words', 0, [[Repeat words format for data]])

-- file location
cmd:option('-model', 'seq2seq_lstm_attn.t7.', [[Path to model .t7 file]])
cmd:option('-src_file', '',[[Source sequence to decode (one line per sequence)]])
cmd:option('-targ_file', '', [[True target sequence (optional)]])
cmd:option('-src_hdf5', '', [[Instead of src_file and targ_file, can provide a src_hdf5 from preprocessing]])

cmd:option('-output_file', 'pred.txt', [[Path to output the predictions (each line will be the
                                       decoded sequence]])
cmd:option('-src_dict', 'data/demo.src.dict', [[Path to source vocabulary (*.src.dict file)]])
cmd:option('-targ_dict', 'data/demo.targ.dict', [[Path to target vocabulary (*.targ.dict file)]])
cmd:option('-char_dict', 'data/demo.char.dict', [[If using chars, path to character 
                                                vocabulary (*.char.dict file)]])

-- beam search options
cmd:option('-beam', 5,[[Beam size]])
cmd:option('-max_sent_l', 250, [[Maximum sentence length. If any sequences in srcfile are longer
                               than this then it will error out]])
cmd:option('-simple', 0, [[If = 1, output prediction is simply the first time the top of the beam
                         ends with an end-of-sentence token. If = 0, the model considers all 
                         hypotheses that have been generated so far that ends with end-of-sentence 
                         token and takes the highest scoring of all of them.]])
cmd:option('-replace_unk', 0, [[Replace the generated UNK tokens with the source token that 
                              had the highest attention weight. If srctarg_dict is provided, 
                              it will lookup the identified source token and give the corresponding 
                              target token. If it is not provided (or the identified source token
                              does not exist in the table) then it will copy the source token]])
cmd:option('-srctarg_dict', 'data/en-de.dict', [[Path to source-target dictionary to replace UNK 
                             tokens. See README.md for the format this file should be in]])
cmd:option('-score_gold', 1, [[If = 1, score the log likelihood of the gold as well]])
cmd:option('-n_best', 1, [[If > 1, it will also output an n_best list of decoded sentences]])
cmd:option('-gpuid',  -1, [[ID of the GPU to use (-1 = use CPU)]])
--cmd:option('-gpuid2', -1, [[Second GPU ID]])
cmd:option('-cudnn', 0, [[If using character model, this should be = 1 if the character model
                          was trained using cudnn]])
opt = cmd:parse(arg)

function reset_state(state, batch_l)
   local u = {}
   for i = 1, #state do
      state[i]:zero()
      table.insert(u, state[i][{{1, batch_l}}])
   end
   return u
end


function copy(orig)
   local orig_type = type(orig)
   local copy
   if orig_type == 'table' then
      copy = {}
      for orig_key, orig_value in pairs(orig) do
         copy[orig_key] = orig_value
      end
   else
      copy = orig
   end
   return copy
end

local StateAll = torch.class("StateAll")

function StateAll.initial(start)
   return {start}
end

function StateAll.advance(state, token)
   local new_state = copy(state)
   table.insert(new_state, token)
   return new_state
end

function StateAll.disallow(out)
   local bad = {1, 3} -- 1 is PAD, 3 is BOS
   for j = 1, #bad do
      out[bad[j]] = -1e9
   end
end

function StateAll.same(state1, state2)
   for i = 2, #state1 do
      if state1[i] ~= state2[i] then
         return false
      end
   end
   return true
end

function StateAll.next(state)
   return state[#state]
end

function StateAll.heuristic(state)
   return 0
end

function StateAll.print(state)
   local result = ""
   for i = 1, #state do
      result = result .. state[i] .. " "
   end
   logging:info(result, MUTE)
end

function pretty_print(t)
  for i,x in ipairs(t) do
    local result = ""
    if i > 1 then
      for j = 1, x:size(1) do
        result = result .. string.format("%.4f ", x[j])
      end
      logging:info(result, MUTE)
    end
  end
end

-- Convert a flat index to a row-column tuple.
function flat_to_rc(v, flat_index)
   local row = math.floor((flat_index - 1) / v:size(2)) + 1
   return row, (flat_index - 1) % v:size(2) + 1
end

function get_nonzeros(source)
  local nonzeros = {}
  for i = 1, source:size(1) do
    table.insert(nonzeros, source[i]:ne(1):sum())
  end
  return nonzeros
end

function generate_beam(model, initial, K, max_sent_l, source, gold)
   --reset decoder initial states
   local n = max_sent_l
  -- Backpointer table.
   local prev_ks = torch.LongTensor(n, K):fill(1)
   -- Current States.
   local next_ys = torch.LongTensor(n, K):fill(1)
   -- Current Scores.
   local scores = torch.FloatTensor(n, K)
   scores:zero()

   local nonzeros = get_nonzeros(source)
   source = source:t():contiguous() -- get words to dim 1 for LSTM
   --local source_l = math.min(source:size(1), opt.max_sent_l)
   local source_char_l = math.min(source:size(1), opt.max_sent_l)
   local source_sent_l = source:size(2)
   local attn_argmax = {}   -- store attn weights
   attn_argmax[1] = {}
   local attn_list = {}
   attn_list[1] = {}
   local deficit_list = {}
   deficit_list[1] = {}
   local sentence_attn_list = {}
   sentence_attn_list[1] = {}

   local attn_argmax_words
   if model_opt.hierarchical == 1 then
     attn_argmax_words = {}
     attn_argmax_words[1] = {}
   end

   local states = {} -- store predicted word idx
   states[1] = {}
   for k = 1, 1 do
      table.insert(states[1], initial)
      table.insert(attn_argmax[1], initial)
      table.insert(attn_list[1], initial)
      table.insert(sentence_attn_list[1], initial)
      table.insert(deficit_list[1], initial)
      next_ys[1][k] = State.next(initial)
   end

   local source_input
   -- for batch
   if model_opt.use_chars_enc == 1 then
      source_input = source:view(source_char_l, 1, source:size(2)):contiguous():cuda()
   else
      source_input = source:view(source_char_l, 1)
   end
   local pad_mask = source_input:eq(1)
   local source_lens = source_input:ne(1):sum(1):cuda():squeeze(1)
   --source_lens[source_lens:eq(0)]:fill(1) -- prevent indexing 0
   local rnn_state_mask
   if model_opt.no_bow == 1 then
     rnn_state_mask = torch.zeros(1, source_sent_l, source_char_l, model_opt.rnn_size):byte():cuda()
     for t = 1, source_sent_l do
       local idx = source_lens[1][t]
       rnn_state_mask[1][t][idx]:fill(1)
     end
   end

   --local rnn_state_enc = {}
   --for i = 1, #init_fwd_enc do
      --table.insert(rnn_state_enc, init_fwd_enc[i]:zero())
   --end   
   local rnn_state_enc = reset_state(init_fwd_enc, 1*source_sent_l)
   local context = context_proto[{{}, {1, source_sent_l}, {1,source_char_l}}]:clone() -- 1 x source_l x source_char_l x rnn_size
   local context_bow = context_bow_proto[{{}, {1, source_sent_l}}]:clone() -- 1 x source_l x word_vec_size
   local rnn_state_bow_enc
   if model_opt.bow_encoder_lstm == 1 then
     rnn_state_bow_enc = reset_state(init_fwd_bow_enc, 1)
   end

    -- pos embeds
    if model_opt.pos_embeds == 1 and model_opt.hierarchical == 1 then
      local pos = pos_proto[{{}, {1, source_sent_l}}]:reshape(1*source_sent_l)
      local pos_states = model[layers_idx['pos_embeds']]:forward(pos) -- 1*source_sent_l x num_layers*rnn_size*2

      for l = 1, model_opt.num_layers do
        rnn_state_enc[l*2-1]:copy(pos_states[{{},{(l*2-2)*model_opt.rnn_size+1, (l*2-1)*model_opt.rnn_size}}])
        rnn_state_enc[l*2]:copy(pos_states[{{},{(l*2-1)*model_opt.rnn_size+1, (l*2)*model_opt.rnn_size}}])
      end
    end

   
   for t = 1, source_char_l do
      local encoder_input = {source_input[t], table.unpack(rnn_state_enc)}
      local out = model[1]:forward(encoder_input)
      if model_opt.mask_padding == 1 then
        local cur_mask = pad_mask[t]:view(1*source_sent_l, 1):expand(1*source_sent_l, model_opt.rnn_size)
        for L = 1, model_opt.num_layers do
          out[L*2-1]:maskedFill(cur_mask, 0)
          out[L*2]:maskedFill(cur_mask, 0)
        end
      end

      rnn_state_enc = out
      context[{{},{},t}]:copy(out[#out]:view(1, source_sent_l, model_opt.rnn_size))
   end

   local masked_selecter, rnn_states
   if model_opt.no_bow == 1 then
     masked_selecter = make_last_state_selecter(model_opt.rnn_size, 1, source_sent_l) -- TODO: seems wrong
     rnn_states = masked_selecter:forward({context, rnn_state_mask})
   end
   if model_opt.hierarchical == 1 then
     local bow_out
     if model_opt.pos_embeds_sent == 1 then
       local pos = pos_proto[{{}, {1, source_sent_l}}]
       bow_out = model[layers_idx['bow_encoder']]:forward({source_input:permute(2,3,1):contiguous(), pos})
     else
       bow_out = model[layers_idx['bow_encoder']]:forward(source_input:permute(2,3,1):contiguous())
     end

     if model_opt.bow_encoder_lstm == 1 then
       -- pass bag of words through LSTM over sentences for context
       for t = 1, source_sent_l do
         local bow_encoder_input
         if model_opt.no_bow == 1 then
           bow_encoder_input = {rnn_states[t], table.unpack(rnn_state_bow_enc)}
         else
           bow_encoder_input = {bow_out[t], table.unpack(rnn_state_bow_enc)}
         end
         local out = model[layers_idx['bow_encoder_lstm']]:forward(bow_encoder_input)
         rnn_state_bow_enc = out
         context_bow[{{}, t}]:copy(out[#out])
       end
     else
       context_bow:copy(bow_out)
     end
   end
   rnn_state_dec = {}
   for i = 1, #init_fwd_dec do
      table.insert(rnn_state_dec, init_fwd_dec[i]:zero())
   end

   if model_opt.init_dec == 1 then
      init_dec_module = make_init_dec_module(model_opt, 1, source_sent_l)
      for L = 1, model_opt.num_layers do
         rnn_state_dec[L*2-1+model_opt.input_feed]:copy(
            init_dec_module:forward(rnn_state_enc[L*2-1]):expand(K, model_opt.rnn_size))
         rnn_state_dec[L*2+model_opt.input_feed]:copy(
            init_dec_module:forward(rnn_state_enc[L*2]):expand(K, model_opt.rnn_size))
      end
   end
   
   if model_opt.brnn == 1 then
      local rnn_state_enc = reset_state(init_fwd_enc, 1*source_sent_l)
      --for i = 1, #rnn_state_enc do
         --rnn_state_enc[i]:zero()
      --end      
      for t = source_char_l, 1, -1 do
         local encoder_input = {source_input[t], table.unpack(rnn_state_enc)}
         local out = model[layers_idx['encoder_bwd']]:forward(encoder_input)
         rnn_state_enc = out
         context[{{},{},t}]:add(out[#out]:view(1, source_sent_l, model_opt.rnn_size))
      end
      if model_opt.init_dec == 1 then
         for L = 1, model_opt.num_layers do
            rnn_state_dec[L*2-1+model_opt.input_feed]:add(
               init_dec_module:forward(rnn_state_enc[L*2-1]:expand(K, model_opt.rnn_size)))
            rnn_state_dec[L*2+model_opt.input_feed]:add(
               init_dec_module:forward(rnn_state_enc[L*2]:expand(K, model_opt.rnn_size)))
         end         
      end      
   end   
   context = context:expand(K, source_sent_l, source_char_l, model_opt.rnn_size)
   if model_opt.hierarchical == 1 then
     context_bow = context_bow:expand(K, source_sent_l, model_opt.bow_size)
   end
   
   out_float = torch.FloatTensor()
   
   local i = 1
   local done = false
   local max_score = -1e9
   local found_eos = false
   while (not done) and (i < n) do
      i = i+1
      states[i] = {}
      attn_argmax[i] = {}
      attn_list[i] = {}
      sentence_attn_list[i] = {}
      deficit_list[i] = {}
      if model_opt.hierarchical == 1 then
        attn_argmax_words[i] = {}
      end
      local decoder_input1
      if model_opt.use_chars_dec == 1 then
         decoder_input1 = word2charidx_targ:index(1, next_ys:narrow(1,i-1,1):squeeze())
      else
         decoder_input1 = next_ys:narrow(1,i-1,1):squeeze()
         if opt.beam == 1 then
            decoder_input1 = torch.LongTensor({decoder_input1})
         end        
      end
      local decoder_input = {decoder_input1, table.unpack(rnn_state_dec)}
      local out_decoder = model[2]:forward(decoder_input)
      local decoder_attn_input
      if model_opt.hierarchical == 1 then
        decoder_attn_input = {out_decoder[#out_decoder], context, context_bow}
      else
        decoder_attn_input = {out_decoder[#out_decoder], context}
      end
      local attn_out = model[layers_idx['decoder_attn']]:forward(decoder_attn_input)
      local out = model[3]:forward(attn_out) -- K x vocab_size
      
      rnn_state_dec = {} -- to be modified later
      if model_opt.input_feed == 1 then
         table.insert(rnn_state_dec, attn_out)
      end      
      for j = 1, #out_decoder do
         table.insert(rnn_state_dec, out_decoder[j])
      end
      out_float:resize(out:size()):copy(out)
      for k = 1, K do
         State.disallow(out_float:select(1, k))
         out_float[k]:add(scores[i-1][k])
      end
      -- All the scores available.

       local flat_out = out_float:view(-1)
       if i == 2 then
          flat_out = out_float[1] -- all outputs same for first batch
       end

       if model_opt.start_symbol == 1 then
          decoder_softmax.output[{{},1}]:zero()
          decoder_softmax.output[{{},source_l}]:zero()
       end
       
       for k = 1, K do
          while true do
             local score, index = flat_out:max(1)
             local score = score[1]
             local prev_k, y_i = flat_to_rc(out_float, index[1])
             states[i][k] = State.advance(states[i-1][prev_k], y_i)
             local diff = true
             for k2 = 1, k-1 do
                if State.same(states[i][k2], states[i][k]) then
                   diff = false
                end
             end
             
             if i < 2 or diff then
                 if opt.view_attn == 1 then
                   -- outdated...
                   print('decoder attention at time ' .. i)
                   if model_opt.hierarchical == 1 then
                     print('row:', decoder_softmax.output) -- K x source_sent_l
                     print('words:', decoder_softmax_words.output:view(K, source_sent_l, source_char_l))
                   else
                     print('all words:', decoder_softmax.output:view(K, source_sent_l, source_char_l))
                   end
                   io.read()
                 end
                 if model_opt.hierarchical == 1 then
                    local row_attn = decoder_softmax.output[prev_k]:clone()
                    local word_attn = decoder_softmax_words.output:reshape(K, source_sent_l, source_char_l)[prev_k]:clone()
                    local result
                    for r = 1, row_attn:size(1) do
                      if nonzeros[r] > 0 then -- ignore blank sentences
                        word_attn[r]:mul(row_attn[r])
                        local cur = word_attn[r][{{1, nonzeros[r]}}]
                        if result == nil then
                          result = cur
                        else
                          result = torch.cat(result, cur, 1)
                        end
                      end
                    end
                    attn_list[i][k] = State.advance(attn_list[i-1][prev_k], result)
                    sentence_attn_list[i][k] = State.advance(sentence_attn_list[i-1][prev_k], row_attn)
                    deficit_list[i][k] = State.advance(deficit_list[i-1][prev_k], 1-result:sum())
                    max_attn, max_index = result:max(1)
                    attn_argmax[i][k] = State.advance(attn_argmax[i-1][prev_k],max_index[1])         
                 else
                    local pre_attn = decoder_softmax.output:reshape(K, source_sent_l, source_char_l)[prev_k]:clone()
                    local row_attn = pre_attn:sum(2):squeeze(2)
                    local result
                    for r = 1, source_sent_l do
                      local cur
                      if nonzeros[r] > 0 then -- ignore blank sentences
                        cur = pre_attn[r][{{1, nonzeros[r]}}]
                        if result == nil then
                          result = cur
                        else
                          result = torch.cat(result, cur, 1)
                        end
                      end
                    end
                    attn_list[i][k] = State.advance(attn_list[i-1][prev_k], result)
                    sentence_attn_list[i][k] = State.advance(sentence_attn_list[i-1][prev_k], row_attn)
                    deficit_list[i][k] = State.advance(deficit_list[i-1][prev_k], 1-result:sum())
                    max_attn, max_index = result:max(1)
                    attn_argmax[i][k] = State.advance(attn_argmax[i-1][prev_k],max_index[1])         
                 end

                prev_ks[i][k] = prev_k
                next_ys[i][k] = y_i
                scores[i][k] = score
                flat_out[index[1]] = -1e9
                break -- move on to next k 
             end
             flat_out[index[1]] = -1e9
          end
       end
       for j = 1, #rnn_state_dec do
          rnn_state_dec[j]:copy(rnn_state_dec[j]:index(1, prev_ks[i]))
       end
       end_hyp = states[i][1]
       end_score = scores[i][1]
        end_attn_argmax = attn_argmax[i][1]
       end_attn_list = attn_list[i][1]
       end_sentence_attn_list = sentence_attn_list[i][1]
       end_deficit_list = deficit_list[i][1]
       if end_hyp[#end_hyp] == END then
          done = true
          found_eos = true
       else
          for k = 1, K do
             local possible_hyp = states[i][k]
             if possible_hyp[#possible_hyp] == END then
                found_eos = true
                if scores[i][k] > max_score then
                   max_hyp = possible_hyp
                   max_score = scores[i][k]
                    max_attn_argmax = attn_argmax[i][k]
                    max_attn_list = attn_list[i][k]
                    max_sentence_attn_list = sentence_attn_list[i][k]
                    max_deficit_list = deficit_list[i][k]
                end
             end             
          end          
       end       
   end
   local gold_score = 0
   if opt.score_gold == 1 then
      local gold_sentence_attn_list = {}
      gold_sentence_attn_list[1] = initial
      local gold_attn_list = {}
      gold_attn_list[1] = initial

      rnn_state_dec = {}
      for i = 1, #init_fwd_dec do
         table.insert(rnn_state_dec, init_fwd_dec[i][{{1}}]:zero())
      end
      if model_opt.init_dec == 1 then
         for L = 1, model_opt.num_layers do
            --rnn_state_dec[L*2]:copy(init_dec_module:forward(rnn_state_enc[L*2-1][{{1}}]))
            --rnn_state_dec[L*2+1]:copy(init_dec_module:forward(rnn_state_enc[L*2][{{1}}]))
            rnn_state_dec[L*2]:copy(init_dec_module:forward(rnn_state_enc[L*2-1]))
            rnn_state_dec[L*2+1]:copy(init_dec_module:forward(rnn_state_enc[L*2]))
         end
      end
      local target_l = gold:size(1) 
      for t = 2, target_l do
         local decoder_input1
         if model_opt.use_chars_dec == 1 then
            decoder_input1 = word2charidx_targ:index(1, gold[{{t-1}}])
         else
            decoder_input1 = gold[{{t-1}}]
         end
         local decoder_input = {decoder_input1, table.unpack(rnn_state_dec)}
         local out_decoder = model[2]:forward(decoder_input)
         local decoder_attn_input
         if model_opt.hierarchical == 1 then
           decoder_attn_input = {out_decoder[#out_decoder], context[{{1}}], context_bow[{{1}}]}
         else
           decoder_attn_input = {out_decoder[#out_decoder], context[{{1}}]}
         end
         local attn_out = model[layers_idx['decoder_attn']]:forward(decoder_attn_input)
         if opt.print_attn == 1 or opt.print_sent_attn == 1 then
             gold_sentence_attn_list[t] = {}
             gold_attn_list[t] = {}
             if model_opt.hierarchical == 1 then
                local row_attn = decoder_softmax.output[1]:clone()
                local word_attn = decoder_softmax_words.output:clone()
                local result
                for r = 1, row_attn:size(1) do
                  if nonzeros[r] > 0 then -- ignore blank sentences
                    word_attn[r]:mul(row_attn[r])
                    local cur = word_attn[r][{{1, nonzeros[r]}}]
                    if result == nil then
                      result = cur
                    else
                      result = torch.cat(result, cur, 1)
                    end
                  end
                end
                gold_sentence_attn_list[t] = row_attn
                gold_attn_list[t] = result
             else
                local pre_attn = decoder_softmax.output:clone()
                if opt.no_pad == 1 then
                  pre_attn = pre_attn:view(opt.no_pad_sent_l, model_opt.max_word_l)
                end
                local row_attn = pre_attn:sum(2):squeeze(2)
                local result
                for r = 1, source_sent_l do
                  local cur
                  if nonzeros[r] > 0 then -- ignore blank sentences
                    cur = pre_attn[r][{{1, nonzeros[r]}}]
                    if result == nil then
                      result = cur
                    else
                      result = torch.cat(result, cur, 1)
                    end
                  end
                end
                gold_sentence_attn_list[t] = row_attn
                gold_attn_list[t] = result
             end
         end

         local out = model[3]:forward(attn_out) -- K x vocab_size
         rnn_state_dec = {} -- to be modified later
         if model_opt.input_feed == 1 then
           table.insert(rnn_state_dec, attn_out)
         end
         for j = 1, #out_decoder do
            table.insert(rnn_state_dec, out_decoder[j])
         end
         gold_score = gold_score + out[1][gold[t]]

      end      
      if opt.print_attn == 1 then
        logging:info('ATTN GOLD', MUTE)
        pretty_print(gold_attn_list)
      end
      if opt.print_sent_attn == 1 then
        -- sentence attn
        logging:info('ATTN LEVEL GOLD', MUTE)
        pretty_print(gold_sentence_attn_list)

        for j = 2, #gold_sentence_attn_list do
          local p = gold_sentence_attn_list[j]
          ENTROPY = ENTROPY + p:cmul(torch.log(p + 1e-8)):sum()
          GOLD_SIZE = GOLD_SIZE + p:size(1)
        end
      end
   end
   if opt.simple == 1 or end_score > max_score or not found_eos then
      max_hyp = end_hyp
      max_score = end_score
      max_attn_argmax = end_attn_argmax
      max_attn_list = end_attn_list
      max_sentence_attn_list = end_sentence_attn_list
      max_deficit_list = end_deficit_list
   end

   return max_hyp, max_score, max_attn_argmax, gold_score, states[i], scores[i], attn_argmax[i], max_attn_list, max_deficit_list, max_sentence_attn_list
end

function idx2key(file)   
   local f = io.open(file,'r')
   local t = {}
   for line in f:lines() do
      local c = {}
      for w in line:gmatch'([^%s]+)' do
         table.insert(c, w)
      end
      t[tonumber(c[2])] = c[1]
   end   
   return t
end

function flip_table(u)
   local t = {}
   for key, value in pairs(u) do
      t[value] = key
   end
   return t   
end


function get_layer(layer)
   if layer.name ~= nil then
      if layer.name == 'decoder_attn' then
         decoder_attn = layer
      elseif layer.name:sub(1,3) == 'hop' then
         hop_attn = layer
      elseif layer.name:sub(1,7) == 'softmax' then
         table.insert(softmax_layers, layer)
      elseif layer.name == 'word_vecs_enc' then
         word_vecs_enc = layer
      elseif layer.name == 'word_vecs_dec' then
         word_vecs_dec = layer
      elseif layer.name == 'sampler' then
        sampler_layer = layer
      end       
   end
   if layer.__typename == 'nn.SoftPlus' then
     probe_layer = layer
   end
end

function sent2wordidx(sent, word2idx, start_symbol)
   local t = {}
   local u = {}
   if start_symbol == 1 then
      table.insert(t, START)
      table.insert(u, START_WORD)
   end
   
   for word in sent:gmatch'([^%s]+)' do
      local idx = word2idx[word] or UNK 
      table.insert(t, idx)
      table.insert(u, word)
   end
   if start_symbol == 1 then
      table.insert(t, END)
      table.insert(u, END_WORD)
   end   
   return torch.LongTensor(t), u
end

function doc2charidx(doc, char2idx, max_word_l, start_symbol)
   local words = {}
   local st = 1
   for idx in doc:gmatch("()(</s>)") do
     local sent = doc:sub(st, idx-2)
     st = idx + 5
     table.insert(words, {})
     if start_symbol == 1 then
        table.insert(words[#words], START_WORD)
     end   
     for word in sent:gmatch'([^%s]+)' do
        table.insert(words[#words], word)
     end
     if start_symbol == 1 then
        table.insert(words[#words], END_WORD)
     elseif opt.no_pad == 1 then
        table.insert(words[#words], "</s>")
     end   
   end
   --local chars = torch.ones(#words, max_word_l)
   --for i = 1, #words do
      --chars[i] = word2charidx(words[i], char2idx, max_word_l, chars[i])
   --end
   local chars
   if opt.no_pad == 1 then
     local rep_words = opt.repeat_words
     chars = torch.ones(opt.no_pad_sent_l, max_word_l)
     local i = 1
     local j = 1
     local done = false
     for _,word in ipairs(words) do
       for _, char in ipairs(word) do
         local char_idx = char2idx[char] or UNK
         chars[i][j] = char_idx
         j = j+1
         if j == max_word_l + 1 then
           i = i+1
           if i == opt.no_pad_sent_l + 1 then
             done = true
             break
           end

           if rep_words > 0 then
             -- copy last row over
             local tail = chars[{{i-1}, {max_word_l-rep_words+1, max_word_l}}]
             chars[{{i},{1,rep_words}}]:copy(tail)
             j = rep_words + 1
           else
             j = 1
           end
         end
       end
       if done then break end
     end
   else
     chars = torch.ones(#words, max_word_l)
     for i = 1, #words do
        chars[i] = word2charidx(words[i], char2idx, max_word_l, chars[i], start_symbol)
     end
   end
   return chars, words
end

function word2charidx(word, char2idx, max_word_l, t, start_symbol)
   local i = 1
   if start_symbol == 1 then
     t[1] = START
     i = 2
   end
   --for _, char in utf8.next, word do
      --char = utf8.char(char)
   for _,char in ipairs(word) do
      local char_idx = char2idx[char] or UNK
      t[i] = char_idx
      i = i+1
      if i > max_word_l then
         if start_symbol == 1 then
           t[i] = END
         end
         break
      end
   end
   if (i < max_word_l) and (start_symbol == 1) then
      t[i] = END
   end
   return t
end

function wordidx2sent(sent, idx2word, source_str, attn, skip_end)
   local t = {}
   if skip_end == nil then skip_end = true end
   local start_i = 1
   local end_i
   if torch.isTensor(sent) then
     end_i = sent:size(1)
   else
     end_i = #sent
   end
   if skip_end then
      start_i = start_i + 1
      end_i = end_i - 1
   end   
   for i = start_i, end_i do -- skip START and END
      if sent[i] == UNK then
         if opt.replace_unk == 1 then
            local s = source_str[attn[i]]
            if phrase_table[s] ~= nil then
               logging:info(s .. ':' ..phrase_table[s])
            end            
            local r = phrase_table[s] or s
            table.insert(t, r)            
         else
            table.insert(t, idx2word[sent[i]])
         end         
      else
         table.insert(t, idx2word[sent[i]])         
      end           
   end
   return table.concat(t, ' ')
end

function clean_sent(sent)
   local s = stringx.replace(sent, UNK_WORD, '')
   s = stringx.replace(s, START_WORD, '')
   s = stringx.replace(s, END_WORD, '')
   --s = stringx.replace(s, START_CHAR, '')
   --s = stringx.replace(s, END_CHAR, '')
   return s
end

function strip(s)
   return s:gsub("^%s+",""):gsub("%s+$","")
end

function iterate_tensor(t)
  assert(torch.isTensor(t), 'non-tensor provided')
  local i = 0
  local function f(t, _)
    if i < t:size(1) then
      i = i+1
      return t[i]
    else
      return nil
    end
  end
  return f, t, nil
end

function main()
   -- some globals
   PAD = 1; UNK = 2; START = 3; END = 4
   PAD_WORD = '<blank>'; UNK_WORD = '<unk>'; START_WORD = '<d>'; END_WORD = '</d>'
   START_CHAR = '{'; END_CHAR = '}'
   MAX_SENT_L = opt.max_sent_l

   -- parse input params
   opt = cmd:parse(arg)
   assert(opt.log_path ~= '', 'need to set logging')
   logging = logger(opt.log_path)
   logging:info("Command line args:")
   logging:info(arg)
   logging:info("End command line args")

   logging:info('max_sent_l: ' .. MAX_SENT_L)
   if path.exists(opt.src_hdf5) then
     logging:info('using hdf5 file ' .. opt.src_hdf5)
   else
     assert(path.exists(opt.src_file), 'src_file does not exist')
     assert(path.exists(opt.model), 'model does not exist')
   end
   
   if opt.gpuid >= 0 then
      require 'cutorch'
      require 'cunn'
      if opt.cudnn == 1 then
         require 'cudnn'
      end      
   end      
   logging:info('loading ' .. opt.model .. '...')
   checkpoint = torch.load(opt.model)
   logging:info('done!')

   if opt.replace_unk == 1 then
      phrase_table = {}
      if path.exists(opt.srctarg_dict) then
         local f = io.open(opt.srctarg_dict,'r')
         for line in f:lines() do
            local c = line:split("|||")
            phrase_table[strip(c[1])] = c[2]
         end
      end      
   end


   -- load model and word2idx/idx2word dictionaries
   model, model_opt = checkpoint[1], checkpoint[2]
   for i = 1, #model do
      model[i]:evaluate()
   end
   layers_idx = model_opt.save_idx
   
   assert(opt.src_dict ~= '', 'need dictionary')
   opt.targ_dict = opt.src_dict
   opt.char_dict = opt.src_dict
   
   idx2word_src = idx2key(opt.src_dict)
   word2idx_src = flip_table(idx2word_src)
   idx2word_targ = idx2key(opt.targ_dict)
   word2idx_targ = flip_table(idx2word_targ)
   
   -- load character dictionaries if needed
   if model_opt.use_chars_enc == 1 or model_opt.use_chars_dec == 1 then
      --utf8 = require 'lua-utf8'      
      char2idx = flip_table(idx2key(opt.char_dict))
      model[1]:apply(get_layer)
   end
   if model_opt.use_chars_dec == 1 then
      word2charidx_targ = torch.LongTensor(#idx2word_targ, model_opt.max_word_l):fill(PAD)
      for i = 1, #idx2word_targ do
         word2charidx_targ[i] = word2charidx(idx2word_targ[i], char2idx,
                                             model_opt.max_word_l, word2charidx_targ[i])
      end      
   end  

   -- load gold labels if it exists
   if path.exists(opt.targ_file) then
      print('loading GOLD labels at ' .. opt.targ_file)
      gold = {}
      local file = io.open(opt.targ_file, 'r')
      for line in file:lines() do
         table.insert(gold, line)
      end
   else
      if opt.src_hdf5 == '' then
        -- no gold data
        opt.score_gold = 0
      end
   end

   local file
   local src_sents = {}
   local num_sents = 0
   if opt.src_hdf5 ~= '' then
     file = hdf5.open(opt.src_hdf5, 'r')
     local source_char = file:read('source_char'):all()
     num_sents = source_char:size(1)
     for i = 1, num_sents do
       table.insert(src_sents, source_char[i])
     end

     -- reinit gold
     gold = {}
     local targets = file:read('target'):all()
     for i = 1, num_sents do
       table.insert(gold, targets[i])
     end
   else
     file = io.open(opt.src_file, "r")
     for line in file:lines() do
       table.insert(src_sents, line)
       num_sents = num_sents + 1
     end
   end

   if opt.gpuid >= 0 then
      cutorch.setDevice(opt.gpuid)
      for i = 1, #model do
         model[i]:double():cuda()
         model[i]:evaluate()
      end
   end

   softmax_layers = {}
   model[2]:apply(get_layer)
    decoder_attn = model[layers_idx['decoder_attn']]
    decoder_attn:apply(get_layer)
    decoder_softmax = softmax_layers[1]
    decoder_softmax_words = softmax_layers[2]
    if model_opt.hierarchical == 0 then
      assert(decoder_softmax_words == nil)
    end
    if model_opt.hierarchical == 1 and model_opt.attn_type == 'hard' then
      if opt.num_argmax > 0 then
        sampler_layer.multisampling = opt.num_argmax -- do this number of argmax
      end
    end
    attn_layer = torch.zeros(opt.beam, MAX_SENT_L)      
   
   
   MAX_WORD_L = model_opt.max_word_l
   context_proto = torch.zeros(1, MAX_SENT_L, MAX_WORD_L, model_opt.rnn_size)
   context_bow_proto = torch.zeros(1, MAX_SENT_L, model_opt.bow_size)
   if model_opt.pos_embeds == 1 or model_opt.pos_embeds_sent == 1 then
     pos_proto = torch.LongTensor(1, MAX_SENT_L):zero():cuda()
     for t = 1, MAX_SENT_L do
       pos_proto[{{}, {t}}]:fill(t)
     end
   end

   local h_init_dec = torch.zeros(opt.beam, model_opt.rnn_size)
   --local h_init_enc = torch.zeros(1, model_opt.rnn_size) 
   local h_init_enc = torch.zeros(MAX_SENT_L, model_opt.rnn_size) 
   if opt.gpuid >= 0 then
      h_init_enc = h_init_enc:cuda()      
      h_init_dec = h_init_dec:cuda()
      cutorch.setDevice(opt.gpuid)
      context_proto = context_proto:cuda()
      context_bow_proto = context_bow_proto:cuda()
       attn_layer = attn_layer:cuda()
   end
   init_fwd_enc = {}
   init_fwd_dec = {} -- initial context
   init_fwd_bow_enc = {}
   if model_opt.input_feed == 1 then
      table.insert(init_fwd_dec, h_init_dec:clone())
   end
   
   for L = 1, model_opt.num_layers do
      table.insert(init_fwd_enc, h_init_enc:clone())
      table.insert(init_fwd_enc, h_init_enc:clone())
      table.insert(init_fwd_dec, h_init_dec:clone()) -- memory cell
      table.insert(init_fwd_dec, h_init_dec:clone()) -- hidden state      
      if model_opt.bow_encoder_lstm == 1 then
        table.insert(init_fwd_bow_enc, h_init_enc:clone())
        table.insert(init_fwd_bow_enc, h_init_enc:clone())
      end
   end      
     
   pred_score_total = 0
   gold_score_total = 0
   pred_words_total = 0
   gold_words_total = 0
   total_deficit = 0
   
   State = StateAll
   local sent_id = 0
   pred_sents = {}
   local out_file = io.open(opt.output_file,'w')   

   for _,line in ipairs(src_sents) do
      sent_id = sent_id + 1

      local source, source_str
      local target, target_str
      if opt.src_hdf5 == '' then 
        line = clean_sent(line)      
        logging:info('SENT ' .. sent_id .. ': ' ..line, MUTE)
        if model_opt.use_chars_enc == 0 then
           source, source_str = sent2wordidx(line, word2idx_src, model_opt.start_symbol)
        else
           source, source_str = doc2charidx(line, char2idx, model_opt.max_word_l, model_opt.start_symbol)
        end
        if opt.score_gold == 1 then
           target, target_str = sent2wordidx(gold[sent_id], word2idx_targ, 1)
        end
      else
        -- line is a tensor
        source_str = wordidx2sent(line:view(line:nElement()), idx2word_src, nil, nil, false)
        logging:info('SENT ' .. sent_id .. ': ' .. source_str, MUTE)
        source = line
        if opt.score_gold == 1 then
          target = gold[sent_id]
          local nonzero = target:ne(1):sum()
          target = target[{{1, nonzero}}] -- remove padding
          gold[sent_id] = target
          target_str = wordidx2sent(gold[sent_id], idx2word_targ, nil, nil, false)
        end
      end

      state = State.initial(START)
      pred, pred_score, attn, gold_score, all_sents, all_scores, all_attn, attn_list, deficit_list, sentence_attn_list  = generate_beam(model, state, opt.beam, MAX_SENT_L, source, target) -- use attn_list to print attn
      pred_score_total = pred_score_total + pred_score
      pred_words_total = pred_words_total + #pred - 1
      pred_sent = wordidx2sent(pred, idx2word_targ, source_str, attn, true)
      out_file:write(pred_sent .. '\n')      
      logging:info('PRED ' .. sent_id .. ': ' .. pred_sent, MUTE)
      if gold ~= nil then
         if opt.src_hdf5 == '' then
           logging:info('GOLD ' .. sent_id .. ': ' .. gold[sent_id], MUTE)
         else
           logging:info('GOLD ' .. sent_id .. ': ' .. target_str, MUTE)
         end
         if opt.score_gold == 1 then
            logging:info(string.format("PRED SCORE: %.4f, GOLD SCORE: %.4f", pred_score, gold_score), MUTE)
            gold_score_total = gold_score_total + gold_score
            gold_words_total = gold_words_total + target:size(1) - 1                     
         end
      end
      if opt.n_best > 1 then
         for n = 1, opt.n_best do
            pred_sent_n = wordidx2sent(all_sents[n], idx2word_targ, source_str, all_attn[n], false)
            local out_n = string.format("%d ||| %s ||| %.4f", n, pred_sent_n, all_scores[n])
            logging:info(out_n, MUTE)
            out_file:write(out_n .. '\n')
         end         
      end
      if opt.print_attn == 1 then
        logging:info('ATTN PRED', MUTE)
        pretty_print(attn_list)
      end
      if opt.print_sent_attn == 1 then
        logging:info('ATTN LEVEL PRED', MUTE)
        pretty_print(sentence_attn_list)
      end
        --deficit_list[1] = 0
        --total_deficit = total_deficit + torch.Tensor(deficit_list):sum()
        ----print(deficit_list)
        ----io.read()
      --end

      logging:info('', MUTE)
   end
   logging:info(string.format("PRED AVG SCORE: %.4f, PRED PPL: %.4f", pred_score_total / pred_words_total,
                       math.exp(-pred_score_total/pred_words_total)))
   if opt.score_gold == 1 then      
      logging:info(string.format("GOLD AVG SCORE: %.4f, GOLD PPL: %.4f",
                          gold_score_total / gold_words_total,
                          math.exp(-gold_score_total/gold_words_total)))
   end
   logging:info(string.format("attn deficit: %.4f", total_deficit/pred_words_total))
   logging:info(string.format("gold entropy: %.4f", ENTROPY / GOLD_SIZE))
   out_file:close()
end
main()

