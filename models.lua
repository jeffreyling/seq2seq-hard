require 'hard_attn'

function nn.Module:reuseMem()
   self.reuse = true
   return self
end

function nn.Module:setReuse()
   if self.reuse then
      self.gradInput = self.output
   end
end

function make_lstm(data, opt, model, use_chars)
   assert(model == 'enc' or model == 'dec')
   local name = '_' .. model
   local dropout = opt.dropout or 0
   local n = opt.num_layers
   local rnn_size = opt.rnn_size
   local input_size = opt.word_vec_size
   local offset = 0
  -- there will be 2*n+3 inputs
   local inputs = {}
   table.insert(inputs, nn.Identity()()) -- x (batch_size x source_l)
   if model == 'dec' then
      if opt.input_feed == 1 then
        table.insert(inputs, nn.Identity()()) -- prev context_attn (batch_size x rnn_size)
        offset = offset + 1
      end
   end
   for L = 1,n do
      table.insert(inputs, nn.Identity()()) -- prev_c[L]
      table.insert(inputs, nn.Identity()()) -- prev_h[L]
   end

   local x, input_size_L
   local outputs = {}
  for L = 1,n do
     -- c,h from previous timesteps
    local prev_c = inputs[L*2+offset]    
    local prev_h = inputs[L*2+1+offset]
    -- the input to this layer
    if L == 1 then
       if use_chars == 0 then
	  local word_vecs
	  if model == 'enc' then
	     word_vecs = nn.LookupTable(data.source_size, input_size)
	  else
	     word_vecs = nn.LookupTable(data.target_size, input_size)
	  end	  
	  word_vecs.name = 'word_vecs' .. name
	  x = word_vecs(inputs[1]) -- batch_size x word_vec_size
       else
	  local char_vecs = nn.LookupTable(data.char_size, opt.word_vec_size)
	  char_vecs.name = 'word_vecs' .. name
	  x = nn.Reshape(-1, opt.word_vec_size, false)(char_vecs(inputs[1])) -- (batch_size*char_length) x word_vec_size
	  --local charcnn = make_cnn(opt.char_vec_size,  opt.kernel_width, opt.num_kernels)
	  --charcnn.name = 'charcnn' .. name
	  --x = charcnn(char_vecs(inputs[1]))
	  --if opt.num_highway_layers > 0 then
	     --local mlp = make_highway(input_size, opt.num_highway_layers)
	     --mlp.name = 'mlp' .. name
	     --x = mlp(x)
	  --end	  
       end
       input_size_L = input_size
       if model == 'dec' then
          if opt.input_feed == 1 then
             x = nn.JoinTable(2)({x, inputs[1+offset]}) -- batch_size x (word_vec_size + rnn_size)
             input_size_L = input_size + rnn_size
          end	  
       end
    else
       x = outputs[(L-1)*2]
       if opt.res_net == 1 and L > 2 then
	  x = nn.CAddTable()({x, outputs[(L-2)*2]})       
       end       
       input_size_L = rnn_size
       --if opt.multi_attn == L and model == 'dec' then
		--local multi_attn = make_decoder_attn(data, opt, 1)
		--multi_attn.name = 'multi_attn' .. L
		--x = multi_attn({x, inputs[2]})
       --end
       if dropout > 0 then
	  x = nn.Dropout(dropout, nil, false)(x)
       end       
    end
    -- evaluate the input sums at once for efficiency
    local i2h = nn.Linear(input_size_L, 4 * rnn_size):reuseMem()(x)
    local h2h = nn.LinearNoBias(rnn_size, 4 * rnn_size):reuseMem()(prev_h)
    local all_input_sums = nn.CAddTable()({i2h, h2h})

    local reshaped = nn.Reshape(4, rnn_size, true)(all_input_sums)
    local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
    -- decode the gates
    local in_gate = nn.Sigmoid():reuseMem()(n1)
    local forget_gate = nn.Sigmoid():reuseMem()(n2)
    local out_gate = nn.Sigmoid():reuseMem()(n3)
    -- decode the write inputs
    local in_transform = nn.Tanh():reuseMem()(n4)
    -- perform the LSTM update
    local next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
      })
    -- gated cells form the output
    local next_h = nn.CMulTable()({out_gate, nn.Tanh():reuseMem()(next_c)})
    
    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end
  return nn.gModule(inputs, outputs)
end

function make_lstm_bow(data, opt)
   local dropout = opt.dropout or 0
   local n = opt.num_layers
   local rnn_size = opt.rnn_size
   local input_size
   if opt.no_bow == 1 then
     input_size = opt.rnn_size
   else
     input_size = opt.word_vec_size
   end
   local offset = 0
  -- there will be 2*n+3 inputs
   local inputs = {}
   table.insert(inputs, nn.Identity()()) -- x (batch_size x word_vec_size)
   for L = 1,n do
      table.insert(inputs, nn.Identity()()) -- prev_c[L]
      table.insert(inputs, nn.Identity()()) -- prev_h[L]
   end

   local x, input_size_L
   local outputs = {}
  for L = 1,n do
     -- c,h from previous timesteps
    local prev_c = inputs[L*2+offset]    
    local prev_h = inputs[L*2+1+offset]
    -- the input to this layer
    if L == 1 then
       x = inputs[1] -- batch_size x word_vec_size
       input_size_L = input_size
    else
       x = outputs[(L-1)*2]
       if opt.res_net == 1 and L > 2 then
	  x = nn.CAddTable()({x, outputs[(L-2)*2]})       
       end       
       input_size_L = rnn_size
       --if opt.multi_attn == L and model == 'dec' then
		--local multi_attn = make_decoder_attn(data, opt, 1)
		--multi_attn.name = 'multi_attn' .. L
		--x = multi_attn({x, inputs[2]})
       --end
       if dropout > 0 then
	  x = nn.Dropout(dropout, nil, false)(x)
       end       
    end
    -- evaluate the input sums at once for efficiency
    local i2h = nn.Linear(input_size_L, 4 * rnn_size):reuseMem()(x)
    local h2h = nn.LinearNoBias(rnn_size, 4 * rnn_size):reuseMem()(prev_h)
    local all_input_sums = nn.CAddTable()({i2h, h2h})

    local reshaped = nn.Reshape(4, rnn_size, true)(all_input_sums)
    local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
    -- decode the gates
    local in_gate = nn.Sigmoid():reuseMem()(n1)
    local forget_gate = nn.Sigmoid():reuseMem()(n2)
    local out_gate = nn.Sigmoid():reuseMem()(n3)
    -- decode the write inputs
    local in_transform = nn.Tanh():reuseMem()(n4)
    -- perform the LSTM update
    local next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
      })
    -- gated cells form the output
    local next_h = nn.CMulTable()({out_gate, nn.Tanh():reuseMem()(next_c)})
    
    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end
  return nn.gModule(inputs, outputs)
end

function make_bow_encoder(data, opt)
  -- takes 3D tensor input: batch_l x source_l x source_char_l
  
  local input = nn.Identity()()
  local word_vecs = nn.LookupTable(data.char_size, opt.word_vec_size)
  word_vecs.name = 'word_vecs_bow'
  local embeds = word_vecs(nn.Reshape(-1, true)(input)) -- batch_l x (source_l*source_char_l) x word_vec_size
  embeds = nn.ViewAs()({embeds, nn.Replicate(opt.word_vec_size, 4)(input)}) -- batch_l x source_l x source_char_l x word_vec_size

  local output
  if opt.conv_bow == 1 then
    local template = nn.View(-1):setNumInputDims(1)(input) -- (batch_l*source_l) x source_char_l
    template = nn.Replicate(opt.word_vec_size, 3)(template) -- (batch_l*source_l) x source_char_l x word_vec_size
    local reshaped_embeds = nn.ViewAs()({embeds, template})
    local conv = make_cnn(opt.word_vec_size, opt.kernel_width, opt.num_kernels)

    local template2 = nn.Sum(3)(input) -- batch_l x source_l
    template2 = nn.Replicate(opt.num_kernels, 3)(template2) -- batch_l x source_l x num_kernels
    local conv_output = conv(reshaped_embeds)
    if opt.num_highway_layers > 0 then
      local mlp = make_highway(opt.num_kernels, opt.num_highway_layers)
      conv_output = mlp(conv_output)
    end	  
    output = nn.ViewAs()({conv_output, template2})
  else
    -- bag of words
    output = nn.Sum(3)(embeds) -- batch_l x source_l x word_vec_size
  end
  if opt.concat_doc_bow == 1 then
    local doc_bow = nn.Sum(2)(nn.Sum(2)(embeds)) -- batch_l x word_vec_size
    doc_bow = nn.ReplicateAs(2,2)({doc_bow, input}) -- batch_l x source_l x word_vec_size
    output = nn.JoinTable(3)({output, doc_bow}) -- batch_l x source_l x {word_vec_size,num_kernels}+word_vec_size
  end
  if opt.bow_encoder_lstm == 1 then
    -- make output suitable for LSTM
    output = nn.Transpose({1,2})(output) -- source_l x batch_l x {word_vec_size,num_kernels}
  end

  return nn.gModule({input}, {output})
end

function make_last_state_selecter(opt, batch_l, source_l)
  local selecter = nn.Sequential()
  selecter:add(nn.MaskedSelect())
  selecter:add(nn.Reshape(batch_l, source_l, opt.rnn_size, false))
  selecter:add(nn.Transpose({1,2})) -- source_l x batch_l x rnn_size
  return selecter:cuda()
end

function make_decoder_attn(data, opt, simple)
   -- 2D tensor target_t (batch_l x rnn_size) and
   -- 4D tensor for context (batch_l x source_l x source_char_l x rnn_size)

   local inputs = {}
   local outputs = {}
   table.insert(inputs, nn.Identity()())
   table.insert(inputs, nn.Identity()())
   local target_t = nn.LinearNoBias(opt.rnn_size, opt.rnn_size)(inputs[1])
   local context = nn.Reshape(-1, opt.rnn_size, true)(inputs[2]) -- batch_l x (source_l*source_char_l) x rnn_size
   simple = simple or 0
   local dropout = opt.dropout or 0
   -- get attention

   local attn = nn.MM()({context, nn.Replicate(1,3)(target_t)}) -- batch_l x (source_l*source_char_l) x 1
   attn = nn.Sum(3)(attn)
   if opt.attn_type == 'hard' then
     local mul_constant = nn.MulConstant(opt.temperature) -- multiply for temperature
     mul_constant.name = 'mul_constant'
     attn = mul_constant(attn)
   end
   local softmax_attn = nn.SoftMax()
   softmax_attn.name = 'softmax_attn'
   attn = softmax_attn(attn)

   -- sample (hard attention)
   if opt.attn_type == 'hard' then
     local sampler = nn.ReinforceCategorical(opt.semi_sampling_p, opt.entropy_scale)
     sampler.name = 'sampler'
     attn = sampler(attn) -- one hot
   end
   attn = nn.Replicate(1,2)(attn) -- batch_l x  1 x (source_l*source_char_l)

   -- apply attention to context
   local context_combined = nn.MM()({attn, context}) -- batch_l x 1 x rnn_size
   context_combined = nn.Sum(2)(context_combined) -- batch_l x rnn_size
   local context_output
   if simple == 0 then
      context_combined = nn.JoinTable(2)({context_combined, inputs[1]}) -- batch_l x rnn_size*2
      context_output = nn.Tanh()(nn.LinearNoBias(opt.rnn_size*2,
						 opt.rnn_size)(context_combined))
   else
      context_output = nn.CAddTable()({context_combined,inputs[1]})
   end

   if dropout > 0 then
      context_output = nn.Dropout(dropout, nil, false)(context_output)
   end     
   table.insert(outputs, context_output)
   --return nn.gModule(inputs, {context_output})   
   return nn.gModule(inputs, outputs)   
end

function make_hierarchical_decoder_attn(data, opt, simple)
   -- inputs:
   -- 2D tensor target_t (batch_l x rnn_size)
   -- 4D tensor for context (batch_l x source_l x source_char_l x rnn_size)
   -- 3D tensor for BOW context (batch_l x source_l x word_vec_size)
   --     or if using LSTM, (batch_l x source_l x rnn_size)

   local inputs = {}
   local outputs = {}
   table.insert(inputs, nn.Identity()())
   table.insert(inputs, nn.Identity()())
   table.insert(inputs, nn.Identity()())
   local target_t = inputs[1]
   local context = inputs[2]
   local bow_context = inputs[3]
   simple = simple or 0
   local dropout = opt.dropout or 0
   -- get attention

   -- attention over sentences
   local bow_size
   if opt.bow_encoder_lstm == 1 then
     bow_size = opt.rnn_size
   else
     if opt.conv_bow == 1 then
       bow_size = opt.num_kernels
     else
       bow_size = opt.word_vec_size
     end
   end
   if opt.concat_doc_bow == 1 then
     bow_size = bow_size + opt.word_vec_size
   end
   local attn1 = nn.MM()({bow_context,
        nn.Replicate(1,3)(nn.LinearNoBias(opt.rnn_size, bow_size)(target_t))}) -- batch_l x source_l x 1
   attn1 = nn.Sum(3)(attn1)
   local mul_constant1 = nn.MulConstant(opt.temperature) -- multiply for temperature
   mul_constant1.name = 'mul_constant'
   attn1 = mul_constant1(attn1)
   local softmax_attn1 = nn.SoftMax()
   softmax_attn1.name = 'softmax_attn1'
   attn1 = softmax_attn1(attn1) -- batch_l x source_l
   if opt.attn_type == 'hard' then
     -- sample (hard attention)
     local sampler = nn.ReinforceCategorical(opt.semi_sampling_p, opt.entropy_scale)
     sampler.name = 'sampler'
     attn1 = sampler(attn1) -- one hot
   end

   -- attention over words of each sentence
   local reshape_context = nn.Reshape(-1, opt.rnn_size, true)(context) -- batch_l x (source_l*source_char_l) x opt.rnn_size
   local attn2 = nn.MM()({reshape_context,
        nn.Replicate(1,3)(nn.LinearNoBias(opt.rnn_size, opt.rnn_size)(target_t))}) -- batch_l x (source_l*source_char_l) x 1
   attn2 = nn.Sum(3)(attn2)
   attn2 = nn.View(-1):setNumInputDims(1)(nn.ViewAs(3)({attn2, context})) -- (batch_l*source_l) x source_char_l
   local mul_constant2 = nn.MulConstant(opt.temperature) -- multiply for temperature
   mul_constant2.name = 'mul_constant'
   attn2 = mul_constant2(attn2)
   local softmax_attn2 = nn.SoftMax()
   softmax_attn2.name = 'softmax_attn2'
   attn2 = softmax_attn2(attn2)
   if opt.attn_word_type == 'hard' then
     -- word level sampling
     local sampler_word = nn.ReinforceCategorical(opt.semi_sampling_p, opt.entropy_scale)
     sampler_word.name = 'sampler_word'
     attn2 = sampler_word(attn2) -- one hot
   end
   attn2 = nn.ViewAs(3)({attn2, context}) -- batch_l x source_l x source_char_l
   -- multiply attentions together
   local mul_attn = nn.CMulTable()({nn.ReplicateAs(3,3)({attn1, attn2}), attn2}) -- batch_l x source_l x source_char_l
   mul_attn = nn.Replicate(1,2)(nn.View(-1):setNumInputDims(2)(mul_attn)) -- batch_l x 1 x (source_l*source_char_l)

   -- apply attention to context
   local context_combined = nn.MM()({mul_attn, reshape_context}) -- batch_l x 1 x rnn_size
   context_combined = nn.Sum(2)(context_combined) -- batch_l x rnn_size
   local context_output
   if simple == 0 then
      context_combined = nn.JoinTable(2)({context_combined, inputs[1]}) -- batch_l x rnn_size*2
      context_output = nn.Tanh()(nn.LinearNoBias(opt.rnn_size*2,
						 opt.rnn_size)(context_combined))
   else
      context_output = nn.CAddTable()({context_combined,inputs[1]})
   end

   if dropout > 0 then
      context_output = nn.Dropout(dropout, nil, false)(context_output)
   end     
   table.insert(outputs, context_output)
   --return nn.gModule(inputs, {context_output})   
   return nn.gModule(inputs, outputs)   
end

function make_init_dec_module(opt, batch_l, source_l)
  local init_dec_module = nn.Sequential()
  init_dec_module:add(nn.View(batch_l, source_l, opt.rnn_size))
  init_dec_module:add(nn.Sum(2))
  if opt.gpuid >= 0 then
    init_dec_module:cuda()
  end

  return init_dec_module
end

function make_generator(data, opt)
   local model = nn.Sequential()
   model:add(nn.Linear(opt.rnn_size, data.target_size))
   model:add(nn.LogSoftMax())

   local w = torch.ones(data.target_size)
   w[1] = 0
   criterion = nn.ClassNLLCriterion(w)
   criterion.sizeAverage = false
   return model, criterion
end

function make_reinforce(data, opt)
  local baseline_m = nn.Linear(opt.rnn_size, 1)

  local reward_criterion = nn.ReinforceNLLCriterion()
  reward_criterion.zero_one = opt.zero_one

  return baseline_m, reward_criterion
end

-- cnn Unit
function make_cnn(input_size, kernel_width, num_kernels)
   local output
   local input = nn.Identity()() 
   if opt.cudnn == 1 then
      local conv = cudnn.SpatialConvolution(1, num_kernels, input_size,
					    kernel_width, 1, 1, 0)
      local conv_layer = conv(nn.View(1, -1, input_size):setNumInputDims(2)(input))
      output = nn.Sum(3)(nn.Max(3)(nn.Tanh()(conv_layer)))
   else
      local conv = nn.TemporalConvolution(input_size, num_kernels, kernel_width)
      local conv_layer = conv(input)
      output = nn.Max(2)(nn.Tanh()(conv_layer))
   end
   return nn.gModule({input}, {output})
end

function make_highway(input_size, num_layers, output_size, bias, f)
    -- size = dimensionality of inputs
    -- num_layers = number of hidden layers (default = 1)
    -- bias = bias for transform gate (default = -2)
    -- f = non-linearity (default = ReLU)
    
    local num_layers = num_layers or 1
    local input_size = input_size
    local output_size = output_size or input_size
    local bias = bias or -2
    local f = f or nn.ReLU()
    local start = nn.Identity()()
    local transform_gate, carry_gate, input, output
    for i = 1, num_layers do
       if i > 1 then
	  input_size = output_size
       else
	  input = start
       end       
       output = f(nn.Linear(input_size, output_size)(input))
       transform_gate = nn.Sigmoid()(nn.AddConstant(bias, true)(
					nn.Linear(input_size, output_size)(input)))
       carry_gate = nn.AddConstant(1, true)(nn.MulConstant(-1)(transform_gate))
       local proj
       if input_size==output_size then
	  proj = nn.Identity()
       else
	  proj = nn.LinearNoBias(input_size, output_size)
       end
       input = nn.CAddTable()({
	                     nn.CMulTable()({transform_gate, output}),
                             nn.CMulTable()({carry_gate, proj(input)})})
    end
    return nn.gModule({start},{input})
end

