require 'nn'
require 'nngraph'
require 'hdf5'

require 'data.lua'
require 'util.lua'
require 'models.lua'
require 'model_utils.lua'
require 'logging'

--nngraph.setDebug(true)

cmd = torch.CmdLine()

-- debugging
cmd:option('-log_path', '', [[Logging path]])
cmd:option('-try_worst', 0, [[For debugging memory constraints]])
cmd:option('-debug', 0, [[Debug]])

-- sentence embeddings
-- TODO
cmd:option('-sentence_a', 1e-4, [[ See Arora et al (2017) ]])

-- synthetic dataset
cmd:option('-synth_data', 0, [[ Using synthetic dataset ]])

-- useful
cmd:option('-sent_ent', 0.0, [[Sentence entropy regularizer for soft]])
cmd:option('-sparsemax', 0, [[Use sparsemax for sentence attn]])
cmd:option('-sent_learning_rate', 0, [[Learning rate for bow encoder.]])
cmd:option('-fix_bow_vecs', 0, [[No gradients for bow encoder vecs]])
cmd:option('-all_lstm', 0, [[Run LSTM encoder without factoring]])
cmd:option('-use_sigmoid_sent', 0, [[Use sigmoid instead of softmax for sent attn (SOFT ONLY!)]])
cmd:option('-use_sigmoid_word', 0, [[Use sigmoid instead of softmax for word attn (SOFT ONLY!)]])
cmd:option('-hop_attn', 1, [[Do this many hops for attention]])

-- not useful
cmd:option('-linear_bow', 0, [[Add linear layer above bow encoder]])
cmd:option('-linear_bow_size', 100, [[linear layer above bow encoder size]])
cmd:option('-bow_dropout', 0.0, [[Use dropout after bow]])
cmd:option('-separate_vecs', 0, [[Separate sentence and enc vecs]])
cmd:option('-train_temp', 0, [[Trained temperature for sent attn]])
cmd:option('-coarse_attn_only', 0, [[Only use coarse features for context]])
cmd:option('-bahdanau_attn', 0, [[Use Bahdanau attention (concat + MLP)]])
cmd:option('-hid_attn_size', 500, [[hidden size of Bahdanau MLP]])
cmd:option('-inf_mask', 0, [[Mask out attention for padding]])
cmd:option('-add_sent_context', 0, [[Add previous sentence context to each row]])
cmd:option('-add_sent_context_init', 0, [[Init encoder with previous sent context]])
cmd:option('-pos_embeds', 0, [[Use positional embeddings as encoder initialization]])
cmd:option('-pos_embeds_sent', 0, [[Use positional embeddings for each sentence]])
cmd:option('-pos_dim', 25, [[Embedding size for bow context pos]])
--cmd:option('-different_sampling', 1, [[Make this many distributions for sampling, then sum]])
--cmd:option('-diff_method', 'oneplus', [[softplus or oneplus]])

cmd:option('-no_pad', 0, [[Single block of document as image]])
cmd:option('-conv_bow', 1, [[Use convolution instead of summing bag of words]])
cmd:option('-no_bow', 0, [[Use LSTM instead of BOW encoder]])
cmd:option('-bow_encoder_lstm', 0, [[LSTM over sentence BOW (should use with no_bow)]])
cmd:option('-mask_padding', 1, [[Mask LSTM states for padding words]])
cmd:option('-hierarchical', 1, [[Do hierarchical attention]])
cmd:option('-attn_type', 'soft', [[`soft`, `hard` for first attention on decoder side]])

-- model size
cmd:option('-num_layers', 2, [[Number of layers in the LSTM encoder/decoder]])
cmd:option('-rnn_size', 500, [[Size of LSTM hidden states]])
cmd:option('-word_vec_size', 300, [[Word embedding sizes]])

-- FIX THESE FOR TRANSLATION
cmd:option('-init_dec', 0, [[Initialize the hidden/cell state of the decoder at time 
                           0 to be the last hidden/cell state of the encoder. If 0, 
                           the initial states of the decoder are set to zero vectors]])
cmd:option('-input_feed', 1, [[If = 1, feed the context vector at each time step as additional
                             input (vica concatenation with the word embeddings) to the decoder]])

-- necessary for summary
cmd:option('-use_chars_enc', 1, [[If = 1, use character on the encoder 
                                side (instead of word embeddings]])

-- crazy hacks
cmd:option('-denoise', 0, [[Denoising autoencoder p]])
cmd:option('-share_embed', 0, [[ Autoencoder mode: share enc/dec embeddings ]])
cmd:option('-fix_encoder', 0, [[Fix encoder]])
cmd:option('-fix_decoder', 0, [[Fix decoder]])
cmd:option('-reinit_encoder', 0, [[Reinit encoder weights]])
cmd:option('-reinit_decoder', 0, [[Reinit decoder weights]])
cmd:option('-zero_one', 0, [[Use zero-one loss instead of log-prob]])
cmd:option('-moving_variance', 0, [[Use moving variance to normalize rewards (thus ignoring reward_scale)]])
cmd:option('-soft_curriculum', 0, [[Anneal semi_sampling_p as 1/sqrt(epoch) if set to 1]])

-- hard attention specs (attn_type == 'hard')
cmd:option('-multisampling', 0, [[If > 0, in ReinforceCategorical do k samples instead of 1]])
cmd:option('-sampling_curric', 0, [[Set 1 to anneal 5,4,3,2,1 samples]])
cmd:option('-start_soft', 0, [[If training from a soft model, but we want to train hard. Here we copy the parameters]])
cmd:option('-reward_scale', 0.3, [[Scale reward by this factor]])
cmd:option('-entropy_scale', 0.0, [[Scale entropy term]])
cmd:option('-semi_sampling_p', 0, [[Probability of passing params through over sampling,
                                    set 0 to always sample]])
cmd:option('-subtract_first', 1, [[Subtract baseline before adding discounted rewards, i.e. sum of gamma^t*(r_t - b_t)]])

cmd:option('-baseline_method', 'average', [[What baseline update to use. Options are `learned`, `average`]])
cmd:option('-baseline_lr', 0.1, [[Learning rate for averaged baseline, b_{k+1} = (1-lr)*b_k + lr*r]])
cmd:option('-baseline_learning_rate', 0.001, [[Learning rate for learned baseline]])
cmd:option('-global_baseline', 0, [[Baseline global instead of time dependent. Time dependent baseline is better]])
cmd:option('-global_variance', 1, [[Variance global instead of time dependent. Global variance is better]])

-- note that without input feed, actions at time t do not affect future rewards
cmd:option('-discount', 0.5, [[Discount factor for rewards, between 0 and 1]])
cmd:option('-soft_anneal', 0, [[Train with soft attention for this many epochs to begin]])
cmd:option('-num_samples', 1, [[Number of times to sample for each data point (most people do 1)]])
--cmd:option('-temperature', 1, [[Temperature for sampling]])

cmd:option('-brnn', 0, [[If = 1, use a bidirectional RNN. Hidden states of the fwd/bwd
                              RNNs are summed.]])


-- data files
cmd:text("")
cmd:text("**Data options**")
cmd:text("")
cmd:option('-data_file','data/demo-train.hdf5',[[Path to the training *.hdf5 file 
                                               from preprocess.py]])
cmd:option('-val_data_file','data/demo-val.hdf5',[[Path to validation *.hdf5 file 
                                                 from preprocess.py]])
cmd:option('-savefile', 'seq2seq_lstm_attn', [[Savefile name (model will be saved as 
                         savefile_epochX_PPL.t7 where X is the X-th epoch and PPL is 
                         the validation perplexity]])
cmd:option('-num_shards', 0, [[If the training data has been broken up into different shards, 
                             then training files are in this many partitions]])
cmd:option('-train_from', '', [[If training from a checkpoint then this is the path to the
                                pretrained model.]])

-- rnn model specs
cmd:text("")
cmd:text("**Model options**")
cmd:text("")

--cmd:option('-attn', 1, [[If = 1, use attention on the decoder side. If = 0, it uses the last
                       --hidden state of the decoder as context at each time step.]])
cmd:option('-use_chars_dec', 0, [[If = 1, use character on the decoder 
                                side (instead of word embeddings]])
cmd:option('-reverse_src', 0, [[If = 1, reverse the source sequence. The original 
                              sequence-to-sequence paper found that this was crucial to 
                              achieving good performance, but with attention models this
                              does not seem necessary. Recommend leaving it to 0]])
cmd:option('-multi_attn', 0, [[If > 0, then use a another attention layer on this layer of 
                           the decoder. For example, if num_layers = 3 and `multi_attn = 2`, 
                           then the model will do an attention over the source sequence
                           on the second layer (and use that as input to the third layer) and 
                           the penultimate layer]])
cmd:option('-res_net', 0, [[Use residual connections between LSTM stacks whereby the input to 
                          the l-th LSTM layer if the hidden state of the l-1-th LSTM layer 
                          added with the l-2th LSTM layer. We didn't find this to help in our 
                          experiments]])

cmd:text("")
cmd:text("Below options only apply if using the character model.")
cmd:text("")

-- char-cnn model specs (if use_chars == 1)
--cmd:option('-char_vec_size', 300, [[Size of the character embeddings]])
cmd:option('-kernel_width', 6, [[Size (i.e. width) of the convolutional filter]])
cmd:option('-num_kernels', 600, [[Number of convolutional filters (feature maps). So the
                                 representation from characters will have this many dimensions]])
cmd:option('-num_highway_layers', 0, [[Number of highway layers in the character model]])

cmd:text("")
cmd:text("**Optimization options**")
cmd:text("")

-- optimization
cmd:option('-epochs', 20, [[Number of training epochs]])
cmd:option('-start_epoch', 1, [[If loading from a checkpoint, the epoch from which to start]])
cmd:option('-param_init', 0.1, [[Parameters are initialized over uniform distribution with support
                               (-param_init, param_init)]])
cmd:option('-learning_rate', 1, [[Starting learning rate. If AdaGrad is used, then this is the
                                  global learning rate.]])
cmd:option('-learning_method', 'sgd', [[sgd, adagrad, adam, adadelta]])
cmd:option('-max_grad_norm', 5, [[If the norm of the gradient vector exceeds this, renormalize it
                                to have the norm equal to max_grad_norm]])
cmd:option('-dropout', 0.3, [[Dropout probability. 
                            Dropout is applied between vertical LSTM stacks.]])
cmd:option('-lr_decay', 0.5, [[Decay learning rate by this much if (i) perplexity does not decrease
                      on the validation set or (ii) epoch has gone past the start_decay_at_limit]])
cmd:option('-start_decay', 0, [[Start decay if 1]])
cmd:option('-curriculum', 0, [[For this many epochs, order the minibatches based on source
                sequence length. Sometimes setting this to 1 will increase convergence speed.]])
cmd:option('-pre_word_vecs_enc', '', [[If a valid path is specified, then this will load 
                                      pretrained word embeddings (hdf5 file) on the encoder side. 
                                      See README for specific formatting instructions.]])
cmd:option('-pre_word_vecs_dec', '', [[If a valid path is specified, then this will load 
                                      pretrained word embeddings (hdf5 file) on the decoder side. 
                                      See README for specific formatting instructions.]])
cmd:option('-fix_word_vecs_enc', 0, [[If = 1, fix word embeddings on the encoder side]])
cmd:option('-fix_word_vecs_dec', 0, [[If = 1, fix word embeddings on the decoder side]])
cmd:option('-max_batch_l', '', [[If blank, then it will infer the max batch size from validation 
                               data. You should only use this if your validation set uses a different
                               batch size in the preprocessing step]])
cmd:text("")
cmd:text("**Other options**")
cmd:text("")


cmd:option('-start_symbol', 0, [[Use special start-of-sentence and end-of-sentence tokens
                       on the source side. We've found this to make minimal difference]])
-- GPU
cmd:option('-gpuid', -1, [[Which gpu to use. -1 = use CPU]])
--cmd:option('-gpuid2', -1, [[If this is >= 0, then the model will use two GPUs whereby the encoder
                           --is on the first GPU and the decoder is on the second GPU. 
                           --This will allow you to train with bigger batches/models.]])
cmd:option('-cudnn', 0, [[Whether to use cudnn or not for convolutions (for the character model).
                         cudnn has much faster convolutions so this is highly recommended 
                         if using the character model]])
-- bookkeeping
cmd:option('-save_every', 1, [[Save every this many epochs]])
cmd:option('-print_every', 50, [[Print stats after this many batches]])
cmd:option('-seed', 3435, [[Seed for random initialization]])
cmd:option('-prealloc', 1, [[Use memory preallocation and sharing between cloned encoder/decoders]])

cmd:text("")
cmd:text("Options for hard attention")
cmd:text("")



opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

function save_checkpoint(savefile)
  opt.save_idx = layers_idx
  if opt.debug == 0 then
    torch.save(savefile, {layers, opt})
  end
end

function zero_table(t)
   for i = 1, #t do
      t[i]:zero()
   end
end

function train(train_data, valid_data)
   local timer = torch.Timer()
   local num_params = 0
   local start_decay = 0
   params, grad_params = {}, {}
   opt.train_perf = {}
   opt.val_perf = {}
   
   for i = 1, #layers do
      local p, gp = layers[i]:getParameters()
      if opt.train_from:len() == 0 then
   p:uniform(-opt.param_init, opt.param_init)
      end
      num_params = num_params + p:size(1)
      params[i] = p
      grad_params[i] = gp
   end

   if opt.pre_word_vecs_enc:len() > 0 then   
      local f = hdf5.open(opt.pre_word_vecs_enc)     
      local pre_word_vecs = f:read('word_vecs'):all()
      for i = 1, pre_word_vecs:size(1) do
	 word_vec_layers[1].weight[i]:copy(pre_word_vecs[i])
      end      
   end
   if opt.pre_word_vecs_dec:len() > 0 then      
      local f = hdf5.open(opt.pre_word_vecs_dec)     
      local pre_word_vecs = f:read('word_vecs'):all()
      for i = 1, pre_word_vecs:size(1) do
	 word_vec_layers[2].weight[i]:copy(pre_word_vecs[i])
      end      
   end
   if opt.hierarchical == 1 and opt.no_bow == 0 then
     -- copy word2vec to bow_encoder
     num_params = num_params - word_vec_layers[1].weight:nElement()
     word_vecs_bow.weight:copy(word_vec_layers[1].weight)
   end

   if opt.share_embed == 1 then
     -- share word vecs
     word_vec_layers[2].weight:copy(word_vec_layers[1].weight)
   end

   if opt.brnn == 1 then --subtract shared params for brnn
      num_params = num_params - word_vec_layers[1].weight:nElement()
      word_vec_layers[3].weight:copy(word_vec_layers[1].weight)
      --if opt.use_chars_enc == 1 then
	 --for i = 1, charcnn_offset do
	    --num_params = num_params - charcnn_layers[i]:nElement()
	    --charcnn_layers[i+charcnn_offset]:copy(charcnn_layers[i])
	 --end	 
      --end            
   end

   logging:info("Number of parameters: " .. num_params)
   
   -- padding
   word_vec_layers[1].weight[1]:zero()            
   word_vec_layers[2].weight[1]:zero()
   if opt.brnn == 1 then
     word_vec_layers[3].weight[1]:zero()
   end      
   if opt.hierarchical == 1 and opt.no_bow == 0 then
     word_vecs_bow.weight[1]:zero()
   end

   -- prototypes for gradients so there is no need to clone
   local max_word_l_sz = opt.max_word_l
   if opt.denoise > 0 then
     max_word_l_sz = max_word_l_sz + 5
   end
   if opt.all_lstm == 1 then
     max_word_l_sz = opt.max_sent_l_src*opt.max_word_l
   end
   local encoder_grad_proto = torch.zeros(opt.max_batch_l, opt.max_sent_l, max_word_l_sz, opt.rnn_size)
   local encoder_bwd_grad_proto = torch.zeros(opt.max_batch_l, opt.max_sent_l, max_word_l_sz, opt.rnn_size)
   context_proto = torch.zeros(opt.max_batch_l, opt.max_sent_l, max_word_l_sz, opt.rnn_size)
   local encoder_bow_lstm_grad_proto
   if opt.bow_encoder_lstm == 1 then
     local grad_sz
     if opt.no_bow == 1 then
       grad_sz = opt.rnn_size
     else
       grad_sz = opt.word_vec_size
       if opt.conv_bow == 1 then
         grad_sz = opt.num_kernels
       end
       if opt.linear_bow == 1 then
         grad_sz = opt.linear_bow_size
       end
     end
     if opt.pos_embeds_sent == 1 then
       grad_sz = grad_sz + opt.pos_dim
     end
     encoder_bow_lstm_grad_proto = torch.zeros(opt.max_batch_l, opt.max_sent_l, grad_sz)
   end
   local encoder_bow_grad_proto = torch.zeros(opt.max_batch_l, opt.max_sent_l, opt.bow_size)
   context_bow_proto = torch.zeros(opt.max_batch_l, opt.max_sent_l, opt.bow_size)
   if opt.pos_embeds == 1 or opt.pos_embeds_sent == 1 then
     pos_proto = torch.zeros(opt.max_batch_l, opt.max_sent_l):cuda()
     for t = 1, opt.max_sent_l do
       pos_proto[{{}, {t}}]:fill(t)
     end
     if opt.pos_embeds == 1 then
       pos_grad_proto = torch.zeros(opt.max_batch_l*opt.max_sent_l, opt.num_layers*opt.rnn_size*2):cuda()
     end
   end
   local add_sent_context_init_grad_proto
   if opt.add_sent_context_init == 1 then
     add_sent_context_init_grad_proto = torch.zeros(opt.max_batch_l*opt.max_sent_l, opt.num_layers*opt.rnn_size*2):cuda()
   end

   -- clone encoder/decoder up to max source/target length   
   decoder_clones = clone_many_times(decoder, opt.max_sent_l_targ)
   decoder_attn_clones = clone_many_times(decoder_attn, opt.max_sent_l_targ)
   encoder_clones = clone_many_times(encoder, max_word_l_sz)
   if opt.hierarchical == 1 and opt.bow_encoder_lstm == 1 then
     bow_encoder_lstm_clones = clone_many_times(bow_encoder_lstm, opt.max_sent_l_src)
   end
   if opt.brnn == 1 then
      encoder_bwd_clones = clone_many_times(encoder_bwd, max_word_l_sz)
   end   
   for i = 1, max_word_l_sz do
      if encoder_clones[i].apply then
         encoder_clones[i]:apply(function(m) m:setReuse() end)
         if opt.prealloc == 1 then encoder_clones[i]:apply(function(m) m:setPrealloc() end) end
      end
      if opt.brnn == 1 then
         encoder_bwd_clones[i]:apply(function(m) m:setReuse() end)
         if opt.prealloc == 1 then encoder_bwd_clones[i]:apply(function(m) m:setPrealloc() end) end
      end      
   end
   if opt.hierarchical == 1 and opt.bow_encoder_lstm == 1 then
      for i = 1, opt.max_sent_l_src do
        bow_encoder_lstm_clones[i]:apply(function(m) m:setReuse() end)
      end
   end
   for i = 1, opt.max_sent_l_targ do
      if decoder_clones[i].apply then
         decoder_clones[i]:apply(function(m) m:setReuse() end)
         if opt.prealloc == 1 then decoder_clones[i]:apply(function(m) m:setPrealloc() end) end
      end
      if decoder_attn_clones[i].apply then
         if opt.prealloc == 1 then decoder_attn_clones[i]:apply(function(m) m:setPrealloc() end) end
      end
   end   

   if opt.denoise > 0 then
     source_proto = torch.zeros(opt.max_word_l + 5, opt.max_batch_l, opt.max_sent_l):cuda()
     source_proto2 = torch.zeros(opt.max_batch_l, opt.max_sent_l):cuda()
   end
   function denoise(source, source_l, source_char_l, batch_l, r)
     r = r or 0.1

     local new_source = source_proto[{{}, {1, batch_l}, {1, source_l}}]:fill(3)
     local new_source_proto = source_proto2[{{1, batch_l}, {1, source_l}}]
     local new_source_l = 0
     for t = 1, source_char_l do
       p = torch.uniform()
       if p < 1-r and new_source_l < opt.max_word_l + 5 then
         new_source_l = new_source_l + 1
         new_source[new_source_l]:copy(source[t])
       end
       p = torch.uniform()
       if p < r and new_source_l < opt.max_word_l + 5 then
         new_source_l = new_source_l + 1
         local pad_mask = source[t]:eq(1)
         new_source[new_source_l]:fill(torch.random(5, valid_data.target_size))
         new_source[new_source_l]:maskedFill(pad_mask, 1) -- replace pad
       end
       p = torch.uniform()
       if p < r and new_source_l > 1 and t < source_char_l then
         local pad_mask = new_source[new_source_l]:eq(1)
         new_source_proto:copy(new_source[new_source_l-1])
         new_source[new_source_l-1]:copy(new_source[new_source_l])
         new_source[new_source_l]:copy(new_source_proto)
         new_source[new_source_l]:maskedFill(pad_mask, 1) -- replace pad
       end
     end
     new_source_l = math.max(new_source_l, 1)
     return new_source[{{1, new_source_l}}], new_source_l
   end

   if opt.reinit_encoder == 1 then
     params[1]:uniform(-opt.param_init, opt.param_init)
   end
   if opt.reinit_decoder == 1 then
      -- don't reinit attention weights though
     local p, _ = decoder_attn_layers[1]:parameters()

     local save_params = {}
     for i = 1, #p do
       table.insert(save_params, p[i]:clone())
     end
     params[2]:uniform(-opt.param_init, opt.param_init)
    for i = 1, #p do
      p[i]:copy(save_params[i])
    end
  end

   if opt.attn_type == 'hard' and opt.baseline_method == 'average' then
       -- reset baselines
       if opt.global_baseline == 0 and type(opt.baseline) == 'number' then
         -- baseline should be time dependent on target
         opt.baseline = torch.zeros(opt.max_sent_l_targ)
       end

       if opt.moving_variance == 1 then
         if opt.global_variance == 0 and type(opt.reward_variance) == 'number' then
           opt.reward_variance = torch.zeros(opt.max_sent_l_targ)
         end
       end
   end

   local h_init = torch.zeros(opt.max_batch_l, opt.rnn_size)
   local h_init_enc
   if opt.all_lstm == 1 then
     h_init_enc = h_init
   else
     h_init_enc = torch.zeros(opt.max_batch_l*opt.max_sent_l, opt.rnn_size) -- 2D encoder
   end
   if opt.gpuid >= 0 then
      h_init = h_init:cuda()      
      h_init_enc = h_init_enc:cuda()
      context_proto = context_proto:cuda()
      context_bow_proto = context_bow_proto:cuda()
      encoder_grad_proto = encoder_grad_proto:cuda()
      if opt.brnn == 1 then
        encoder_bwd_grad_proto = encoder_bwd_grad_proto:cuda()
      end	 
      if opt.hierarchical == 1 then
        encoder_bow_grad_proto = encoder_bow_grad_proto:cuda()
        if opt.bow_encoder_lstm == 1 then
          encoder_bow_lstm_grad_proto = encoder_bow_lstm_grad_proto:cuda()
        end
      end
   end

   init_fwd_enc = {}
   init_bwd_enc = {}
   init_fwd_dec = {}
   init_bwd_dec = {}
   init_fwd_bow_enc = {}
   init_bwd_bow_enc = {}
   if opt.input_feed == 1 then
      table.insert(init_fwd_dec, h_init:clone())
   end
   
   for L = 1, opt.num_layers do
      table.insert(init_fwd_enc, h_init_enc:clone())
      table.insert(init_fwd_enc, h_init_enc:clone())
      table.insert(init_bwd_enc, h_init_enc:clone())
      table.insert(init_bwd_enc, h_init_enc:clone())
      table.insert(init_fwd_dec, h_init:clone()) -- memory cell
      table.insert(init_fwd_dec, h_init:clone()) -- hidden state
      table.insert(init_bwd_dec, h_init:clone())
      table.insert(init_bwd_dec, h_init:clone())      
      if opt.bow_encoder_lstm == 1 then
        table.insert(init_fwd_bow_enc, h_init:clone())
        table.insert(init_fwd_bow_enc, h_init:clone())
        table.insert(init_bwd_bow_enc, h_init:clone())
        table.insert(init_bwd_bow_enc, h_init:clone())
      end
   end      

   dec_offset = 2 -- offset depends on input feeding
   if opt.input_feed == 1 then
      dec_offset = dec_offset + 1
   end
   
   function reset_state(state, batch_l, t)
      if t == nil then
	 local u = {}
	 for i = 1, #state do
	    state[i]:zero()
	    table.insert(u, state[i][{{1, batch_l}}])
	 end
	 return u
      else
	 local u = {[t] = {}}
	 for i = 1, #state do
	    state[i]:zero()
	    table.insert(u[t], state[i][{{1, batch_l}}])
	 end
	 return u
      end      
   end

   -- clean layer before saving to make the model smaller
   function clean_layer(layer)
      if opt.gpuid >= 0 then
	 layer.output = torch.CudaTensor()
	 layer.gradInput = torch.CudaTensor()
      else
	 layer.output = torch.DoubleTensor()
	 layer.gradInput = torch.DoubleTensor()
      end
      if layer.modules then
	 for i, mod in ipairs(layer.modules) do
	    clean_layer(mod)
	 end
      elseif torch.type(self) == "nn.gModule" then
	 layer:apply(clean_layer)
      end      
   end

   -- decay learning rate if val perf does not improve or we hit the opt.start_decay_at limit
   function decay_lr(epoch)
      if opt.start_decay == 1 then
         start_decay = 1
      end
      
      if opt.val_perf[#opt.val_perf] ~= nil and opt.val_perf[#opt.val_perf-1] ~= nil then
	 local curr_ppl = opt.val_perf[#opt.val_perf]
	 local prev_ppl = opt.val_perf[#opt.val_perf-1]
	 if curr_ppl > prev_ppl then
	    start_decay = 1
	 end
      end
      if start_decay == 1 then
         for j = 1, #layer_etas do
           layer_etas[j] = layer_etas[j] * opt.lr_decay
         end
      end
   end   

   -- given an (unnormalized) reward and stochastic layers, broadcasts the reward to those layers
   function compute_VR_reward(unnorm_reward, mask, t, h_state)
     local baseline_lr = opt.baseline_lr
     local batch_l = unnorm_reward:size(1)

     -- variance reduction: baselines and scaling
     local b, b_learned, b_const
     local scale = opt.reward_scale -- default
     if opt.baseline_method == 'learned' then
       assert(h_state ~= nil, 'need to pass in hidden state for learned baseline')
       b_learned = baseline_m:forward(h_state):squeeze(2)
       b = b_learned
     elseif opt.baseline_method == 'average' then
       -- we update the moving averages first
       if opt.global_baseline == 1 then
         opt.baseline = (1-baseline_lr)*opt.baseline + baseline_lr*unnorm_reward:mean()
         b = opt.baseline
       else
         opt.baseline[t] = (1-baseline_lr)*opt.baseline[t] + baseline_lr*unnorm_reward:mean()
         b = opt.baseline[t]
       end

       if opt.moving_variance == 1 then
         local var_update = unnorm_reward:var()
         if var_update == var_update then -- prevent nan
           if opt.global_variance == 1 then
             opt.reward_variance = (1-baseline_lr)*opt.reward_variance + baseline_lr*var_update
             scale = 1/math.sqrt(opt.reward_variance + 1e-8)
           else
             opt.reward_variance[t] = (1-baseline_lr)*opt.reward_variance[t] + baseline_lr*var_update
             scale = 1/math.sqrt(opt.reward_variance[t] + 1e-8)
           end
         end
       end
     end

     -- get the variance reduced reward
     local cur_reward = variance_reduce(unnorm_reward, b, scale, mask)
     cur_reward:div(batch_l)

     return cur_reward, b, b_learned
   end

   -- called once per epoch
   function train_batch(data, epoch)
      local train_nonzeros = 0
      local train_loss = 0	       
      local batch_order = torch.randperm(data.length) -- shuffle mini batch order     
      local start_time = timer:time().real
      local num_words_target = 0
      local num_words_source = 0
      local num_samples = opt.num_samples

      if opt.attn_type == 'hard' then
         if opt.sampling_curric == 1 then
           sampler_layers = {}
           for i = 1, opt.max_sent_l_targ do
             decoder_attn_clones[i]:apply(get_RL_layer)
           end
           for i = 1, opt.max_sent_l_targ do
              local cur_layer = sampler_layers[i]
              if epoch > 5 and cur_layer.multisampling > 1 then
                -- run with multisampling for 5 epochs then decrease
                cur_layer.multisampling = cur_layer.multisampling - 1
              end
           end
         end

        -- soft anneal 
        local cur_soft_anneal = false
        if opt.soft_anneal > 0 then
          if epoch == 1 then
            -- train with soft attention
            for _,module in ipairs(sampler_layers) do
              module.through = true
            end
          elseif epoch == opt.soft_anneal + 1 then
            for _,module in ipairs(sampler_layers) do
              module.through = false
            end
          end
          if epoch <= opt.soft_anneal then
            cur_soft_anneal = true
          end
        end

        -- soft curriculum
        if opt.soft_curriculum == 1 then
          local p = 1/math.sqrt(epoch)
          logging:info(string.format('soft curriculum sampling p = %.2f', p))
        end
      end
      
      for i = 1, data:size() do
        zero_table(grad_params, 'zero')
        local d
        if epoch <= opt.curriculum then
          if torch.uniform() < 0.1 then
            d = data[torch.random(1,data.length)]
          else
            d = data[torch.random(1, max_curric_i)]
          end
        else
          d = data[batch_order[i]]
          if opt.try_worst == 1 then
            print('Trying worst')
            local idx, b,t,s,sc = data:get_worst()
            print(b,t,s,sc)
            d = data[idx]
          end
        end
        local target, target_out, nonzeros, source = d[1], d[2], d[3], d[4]
        local batch_l, target_l, source_l, target_l_all = d[5], d[6], d[7], d[8]
        local source_char_l = d[9]
        if opt.denoise > 0 then
          -- denoising autoencoder
          source, source_char_l = denoise(source, source_l, source_char_l, batch_l, opt.denoise)
        end
        local bow_source
        if opt.all_lstm == 1 then
          bow_source = source:transpose(1,2):reshape(batch_l, source_l, source_char_l)
        else
          bow_source = source:permute(2,3,1):contiguous()
        end

        local loss = 0
        for sample_i = 1, num_samples do
          local encoder_grads = encoder_grad_proto[{{1, batch_l}, {1, source_l}, {1, source_char_l}}]
          local encoder_bwd_grads 
          if opt.brnn == 1 then
            encoder_bwd_grads = encoder_bwd_grad_proto[{{1, batch_l}, {1, source_l}, {1, source_char_l}}]
          end	 
          if opt.hierarchical == 1 and opt.bow_encoder_lstm == 1 then
            encoder_bow_lstm_grads = encoder_bow_lstm_grad_proto[{{1, batch_l}, {1, source_l}}]
          end
          local context = context_proto[{{1, batch_l}, {1, source_l}, {1, source_char_l}}]
          local rnn_state_mask -- selects last LSTM state of each sentence within batch
          if opt.no_bow == 1 then
            local source_sent_l = bow_source:ne(1):sum(3):cuda():squeeze(3) -- batch_l x source_l
            source_sent_l[source_sent_l:eq(0)]:fill(1)
            rnn_state_mask = torch.zeros(batch_l, source_l, source_char_l, opt.rnn_size):cudaByte()
            for j = 1, batch_l do
              for t = 1, source_l do
                local idx = source_sent_l[j][t]
                if idx == 0 then idx = 1 end
                rnn_state_mask[j][t][idx]:fill(1)
              end
            end
          end

          local encoder_bow_grads
          local context_bow
          local rnn_state_bow_enc
          if opt.hierarchical == 1 then
            encoder_bow_grads = encoder_bow_grad_proto[{{1, batch_l}, {1, source_l}}]
            context_bow = context_bow_proto[{{1, batch_l}, {1, source_l}}]
            if opt.bow_encoder_lstm == 1 then
              rnn_state_bow_enc = reset_state(init_fwd_bow_enc, batch_l, 0)
            end
          end
          --if opt.gpuid >= 0 then
            --cutorch.setDevice(opt.gpuid)
          --end	 

          -- forward prop encoder bow
          local rnn_states -- rnn states for no_bow
          local bow_out -- context per sentence
          if opt.hierarchical == 1 then
            if opt.no_bow == 1 then
              masked_selecter = make_last_state_selecter(opt, batch_l, source_l)
              rnn_states = masked_selecter:forward({context, rnn_state_mask})
              if opt.pos_embeds_sent == 1 then
                local pos = pos_proto[{{1,batch_l}, {1, source_l}}]
                bow_out = pos_embeds_sent:forward({rnn_states, pos})
              else
                bow_out = rnn_states
              end
            else
              bow_encoder:training()
              if opt.pos_embeds_sent == 1 then
                local pos = pos_proto[{{1,batch_l}, {1, source_l}}]
                bow_out = bow_encoder:forward({bow_source, pos})
              else
                bow_out = bow_encoder:forward(bow_source)
              end
            end
              
            if opt.bow_encoder_lstm == 1 then
              -- pass bag of words through LSTM over sentences for context
              for t = 1, source_l do
                bow_encoder_lstm_clones[t]:training()
                local bow_encoder_input = {bow_out[t], table.unpack(rnn_state_bow_enc[t-1])}
                local out = bow_encoder_lstm_clones[t]:forward(bow_encoder_input)
                rnn_state_bow_enc[t] = out
                context_bow[{{}, t}]:copy(out[#out])
              end
            else
              context_bow:copy(bow_out)
            end
          end

          local rnn_state_enc
          if opt.all_lstm == 1 then
            rnn_state_enc = reset_state(init_fwd_enc, batch_l, 0)
          else
            rnn_state_enc = reset_state(init_fwd_enc, batch_l*source_l, 0) -- different batch size for summary, batch_l*source_l x 2*num_layers
          end

          -- pos embeddings
          if opt.pos_embeds == 1 and opt.hierarchical == 1 then
            pos_embeds:training()
            local pos = pos_proto[{{1,batch_l}, {1, source_l}}]:reshape(batch_l*source_l)
            local pos_states = pos_embeds:forward(pos) -- batch_l*source_l x num_layers*rnn_size*2

            for l = 1, opt.num_layers do
              rnn_state_enc[0][l*2-1]:copy(pos_states[{{},{(l*2-2)*opt.rnn_size+1, (l*2-1)*opt.rnn_size}}])
              rnn_state_enc[0][l*2]:copy(pos_states[{{},{(l*2-1)*opt.rnn_size+1, (l*2)*opt.rnn_size}}])
            end
          end

          local sent_context
          if opt.add_sent_context_init == 1 then
            sent_context = add_sent_context_init:forward(context_bow:contiguous())
            if source_l >= 2 then
              sent_context[{{}, {2,source_l}}]:copy(sent_context[{{}, {1,source_l-1}}])
            end
            sent_context[{{}, {1}}]:zero()
            sent_context = sent_context:view(batch_l*source_l, opt.num_layers*opt.rnn_size*2)

            for l = 1, opt.num_layers do
              rnn_state_enc[0][l*2-1]:copy(sent_context[{{},{(l*2-2)*opt.rnn_size+1, (l*2-1)*opt.rnn_size}}])
              rnn_state_enc[0][l*2]:copy(sent_context[{{},{(l*2-1)*opt.rnn_size+1, (l*2)*opt.rnn_size}}])
            end
          elseif opt.add_sent_context == 1 then
            sent_context = torch.zeros(batch_l, source_l, opt.bow_size):cuda()
            if source_l >= 2 then
              sent_context[{{}, {2,source_l}}]:copy(context_bow[{{}, {1,source_l-1}}])
            end
          end

          -- forward prop encoder
          if opt.all_lstm == 1 then
            for t = 1, source_char_l*source_l do
              encoder_clones[t]:training()
              local encoder_input = {source[t], table.unpack(rnn_state_enc[t-1])}
              local out = encoder_clones[t]:forward(encoder_input)
              rnn_state_enc[t] = out
              local t1, t2 = get_indices(t, source_char_l)
              context[{{},t1,t2}]:copy(out[#out])
            end
          else
            for t = 1, source_char_l do
              encoder_clones[t]:training()
              local encoder_input
              if opt.add_sent_context == 1 then
                encoder_input = {source[t], sent_context, table.unpack(rnn_state_enc[t-1])}
              else
                encoder_input = {source[t], table.unpack(rnn_state_enc[t-1])}
              end
              local out = encoder_clones[t]:forward(encoder_input)

              -- mask out padding on encoder
              if opt.mask_padding == 1 then
                local cur_mask = source[t]:eq(1)
                cur_mask = cur_mask:view(batch_l*source_l, 1):expand(batch_l*source_l, opt.rnn_size)
                for L = 1, opt.num_layers do
                  out[L*2-1]:maskedFill(cur_mask, 0)
                  out[L*2]:maskedFill(cur_mask, 0)
                end
              end
                
              rnn_state_enc[t] = out
              context[{{},{},t}]:copy(out[#out]:view(batch_l, source_l, opt.rnn_size))
            end
          end

          -- forward prop decoder
          local rnn_state_dec = reset_state(init_fwd_dec, batch_l, 0)
          if opt.init_dec == 1 then
            assert(false, 'broken init dec...')
            init_dec_modules = {}
            for L = 1, 2*opt.num_layers do
              table.insert(init_dec_modules, make_init_dec_module(opt, batch_l, source_l))
            end
            --for L = 1, opt.num_layers do
              --rnn_state_dec[0][L*2-1+opt.input_feed]:copy(
                  --init_dec_modules[L*2-1]:forward({rnn_state_enc, rnn_state_mask})
              --rnn_state_dec[0][L*2+opt.input_feed]:copy(init_dec_modules[L*2]:forward(rnn_state_final[L*2]))
            --end
          end

          local rnn_state_enc_bwd
          if opt.brnn == 1  then
            rnn_state_enc_bwd = reset_state(init_fwd_enc, batch_l*source_l, source_char_l+1)       	   
            for t = source_char_l, 1, -1 do
              encoder_bwd_clones[t]:training()
              local encoder_input = {source[t], table.unpack(rnn_state_enc_bwd[t+1])}
              local out = encoder_bwd_clones[t]:forward(encoder_input)
              rnn_state_enc_bwd[t] = out
              context[{{},{},t}]:add(out[#out]:view(batch_l, source_l, opt.rnn_size))
            end
            if opt.init_dec == 1 then
              init_dec_modules_bwd = {}
              for L = 1, 2*opt.num_layers do
                table.insert(init_dec_modules_bwd, make_init_dec_module(opt, batch_l, source_l))
              end
              for L = 1, opt.num_layers do
                rnn_state_dec[0][L*2-1+opt.input_feed]:add(init_dec_modules_bwd[L*2-1]:forward(rnn_state_enc_bwd[1][L*2-1]))
                rnn_state_dec[0][L*2+opt.input_feed]:add(init_dec_modules_bwd[L*2]:forward(rnn_state_enc_bwd[1][L*2]))
              end
            end
          end

          local preds = {}
          local decoder_out = {}
          local inf_mask
          if opt.hierarchical == 1 and opt.inf_mask == 1 then
            inf_mask = source:permute(2,3,1):contiguous():eq(1):mul(-math.huge):cuda() -- batch_l x source_l x source_char_l
          end
          for t = 1, target_l do
            decoder_clones[t]:training()
            decoder_attn_clones[t]:training()
            local decoder_input = {target[t], table.unpack(rnn_state_dec[t-1])}
            local out = decoder_clones[t]:forward(decoder_input)
            table.insert(decoder_out, out[#out]) -- for backprop
            local decoder_attn_input
            if opt.hierarchical == 1 then
              if opt.coarse_attn_only == 1 then
                decoder_attn_input = {out[#out], context_bow}
              else
                decoder_attn_input = {out[#out], context, context_bow}
                if opt.inf_mask == 1 then
                  table.insert(decoder_attn_input, inf_mask)
                end
              end
            else
              decoder_attn_input = {out[#out], context}
            end
            local pred = decoder_attn_clones[t]:forward(decoder_attn_input)
            table.insert(preds, pred)

            local next_state = {}
            if opt.input_feed == 1 then
              table.insert(next_state, pred)
            end
            for j = 1, #out do
              table.insert(next_state, out[j])
            end
            rnn_state_dec[t] = next_state
          end

          -- zero out grads
          encoder_grads:zero()
          if opt.brnn == 1 then
            encoder_bwd_grads:zero()
          end
          if opt.hierarchical == 1 then
            encoder_bow_grads:zero()
            if opt.bow_encoder_lstm == 1 then
              encoder_bow_lstm_grads:zero()
            end
          end

          -- backward prop decoder
          local drnn_state_dec = reset_state(init_bwd_dec, batch_l)
          local sum_reward -- for hard attn
          local discount = opt.discount
          for t = target_l, 1, -1 do
            local pred = generator:forward(preds[t])
            if opt.attn_type == 'hard' then
              -- TODO: could still use some cleanup here...
              -- compute and broadcast reward to relevant layers
              local mask = target_l_all:lt(t) -- for padding
              local reward_input = {pred, mask}
              reward_criterion:forward(reward_input, target_out[t])
              local unnorm_reward = reward_criterion.reward

              local targ_reward = unnorm_reward -- default
              local norm_reward, b, b_learned
              if opt.subtract_first == 1 then
                  -- get (r-b) before taking discounted sum
                  norm_reward, b, b_learned = compute_VR_reward(targ_reward, mask, t, preds[t])
                  if opt.input_feed == 1 then
                    if t == target_l then
                      -- cumulative reward
                      sum_reward = norm_reward
                    else
                      sum_reward:mul(discount)
                      sum_reward:add(norm_reward)
                    end
                    norm_reward = sum_reward:clone()
                  end
              else
                  -- get discounted sum then take (R-b)
                  if opt.input_feed == 1 then
                    if t == target_l then
                      -- cumulative reward
                      sum_reward = unnorm_reward
                    else
                      sum_reward:mul(discount)
                      sum_reward:add(unnorm_reward)
                    end
                    targ_reward = sum_reward:clone()
                  end
                  norm_reward, b, b_learned = compute_VR_reward(targ_reward, mask, t, preds[t])
              end
              norm_reward:mul(opt.reward_scale) -- helps performance, kind of like learning rate

              -- broadcast
              local cur_samplers = {}
              function get_single_layer(layer)
                if layer.name ~= nil then
                    if layer.name == 'sampler' then
                      table.insert(cur_samplers, layer)
                    end
                end
              end
              if opt.attn_type == 'hard' then
                decoder_attn_clones[t]:apply(get_single_layer)
                for _,layer in ipairs(cur_samplers) do
                  if opt.soft_curriculum == 1 then
                    layer.semi_sampling_p = 1/math.sqrt(epoch)
                  end
                  if opt.sampling_curric == 1 then
                    if epoch > 5 and layer.multisampling > 1 then
                      -- run with multisampling for 5 epochs then decrease
                      layer.multisampling = layer.multisampling - 1
                    end
                  end
                  layer:reinforce(norm_reward)
                end
              end

              -- update learned baselines
              if opt.baseline_method == 'learned' then
                local dl_db = reward_criterion:update_baseline(b_learned, mask, targ_reward)
                -- no need to divide by batch_l since MSECriterion does it
                baseline_m:backward(preds[t], dl_db:view(dl_db:size(1), 1))
              end
            end

            -- standard backprop
            loss = loss + criterion:forward(pred, target_out[t])/batch_l
            local dl_dpred = criterion:backward(pred, target_out[t])
            dl_dpred:div(batch_l)
            local dl_dtarget = generator:backward(preds[t], dl_dpred)
            local decoder_attn_input
            if opt.hierarchical == 1 then
              if opt.coarse_attn_only == 1 then
                decoder_attn_input = {decoder_out[t], context_bow}
              else
                decoder_attn_input = {decoder_out[t], context, context_bow}
              end
              if opt.inf_mask == 1 then
                table.insert(decoder_attn_input, inf_mask)
              end
            else
              decoder_attn_input = {decoder_out[t], context}
            end
            local dl_dattn = decoder_attn_clones[t]:backward(decoder_attn_input, dl_dtarget)

            drnn_state_dec[#drnn_state_dec]:add(dl_dattn[1])
            local decoder_input = {target[t], table.unpack(rnn_state_dec[t-1])}
            local dlst = decoder_clones[t]:backward(decoder_input, drnn_state_dec)
            -- accumulate encoder/decoder grads
            if opt.hierarchical == 1 then
              if opt.coarse_attn_only == 1 then
                encoder_bow_grads:add(dl_dattn[2])
              else
                encoder_grads:add(dl_dattn[2])
                encoder_bow_grads:add(dl_dattn[3])
              end
            else
              encoder_grads:add(dl_dattn[2])
            end
            if opt.brnn == 1 then
              encoder_bwd_grads:add(dl_dattn[2])
            end

            drnn_state_dec[#drnn_state_dec]:zero()
            if opt.input_feed == 1 then
              drnn_state_dec[#drnn_state_dec]:add(dlst[2])
            end
            for j = dec_offset, #dlst do
              drnn_state_dec[j-dec_offset+1]:copy(dlst[j])
            end
          end
          word_vec_layers[2].gradWeight[1]:zero()
          if opt.fix_word_vecs_dec == 1 then
            word_vec_layers[2].gradWeight:zero()
          end

          -- backward prop encoder
          local drnn_state_enc
          if opt.all_lstm == 1 then
            drnn_state_enc = reset_state(init_bwd_enc, batch_l)
          else
            drnn_state_enc = reset_state(init_bwd_enc, batch_l*source_l)
          end

          if opt.init_dec == 1 then
            for L = 1, opt.num_layers do
              drnn_state_enc[L*2-1]:copy(init_dec_modules[L*2-1]:backward(rnn_state_enc[source_char_l][L*2-1], drnn_state_dec[L*2-1]))
              drnn_state_enc[L*2]:copy(init_dec_modules[L*2]:backward(rnn_state_enc[source_char_l][L*2], drnn_state_dec[L*2]))
            end	    
          end

          if opt.all_lstm == 1 then
            for t = source_l*source_char_l, 1, -1 do
              local t1, t2 = get_indices(t, source_char_l)
              local encoder_input = {source[t], table.unpack(rnn_state_enc[t-1])}

              -- attn grads
              drnn_state_enc[#drnn_state_enc]:add(encoder_grads[{{},t1,t2}])

              local dlst = encoder_clones[t]:backward(encoder_input, drnn_state_enc)
              for j = 1, #drnn_state_enc do
                drnn_state_enc[j]:copy(dlst[j+1])
              end	    
            end
          else
            for t = source_char_l, 1, -1 do
              local encoder_input
              if opt.add_sent_context == 1 then
                encoder_input = {source[t], sent_context, table.unpack(rnn_state_enc[t-1])}
              else
                encoder_input = {source[t], table.unpack(rnn_state_enc[t-1])}
              end

              if opt.mask_padding == 1 then
                if t < source_char_l then
                  local cur_mask = source[t+1]:eq(1)
                  cur_mask = cur_mask:view(batch_l*source_l, 1):expand(batch_l*source_l, opt.rnn_size)
                  for L = 1, opt.num_layers do
                    drnn_state_enc[L*2-1]:maskedFill(cur_mask, 0)
                    drnn_state_enc[L*2]:maskedFill(cur_mask, 0)
                  end
                end
              end

              -- attn grads
              drnn_state_enc[#drnn_state_enc]:add(encoder_grads[{{},{},t}])
              if opt.mask_padding == 1 then
                local cur_mask = source[t]:eq(1)
                cur_mask = cur_mask:view(batch_l*source_l, 1):expand(batch_l*source_l, opt.rnn_size)
                drnn_state_enc[#drnn_state_enc]:maskedFill(cur_mask, 0)
              end

              local dlst = encoder_clones[t]:backward(encoder_input, drnn_state_enc)
              for j = 1+opt.add_sent_context, #drnn_state_enc do
                drnn_state_enc[j]:copy(dlst[j+1])
              end	    

              if opt.add_sent_context == 1 then
                if source_l >= 2 then
                  encoder_bow_grads[{{}, {1,source_l-1}}]:add(dlst[2][{{}, {2,source_l}}])
                end
              end
            end
          end

          -- backward add_sent_context_init
          if opt.add_sent_context_init == 1 then
            local add_sent_context_init_grads = add_sent_context_init_grad_proto[{{1,batch_l*source_l}}]
            add_sent_context_init_grads:zero()
            for l = 1, opt.num_layers do
              add_sent_context_init_grads[{{}, {(l*2-2)*opt.rnn_size+1, (l*2-1)*opt.rnn_size}}]:copy(drnn_state_enc[l*2-1])
              add_sent_context_init_grads[{{}, {(l*2-1)*opt.rnn_size+1, (l*2)*opt.rnn_size}}]:copy(drnn_state_enc[l*2])
            end
            local sent_context_grad = add_sent_context_init_grads:view(batch_l, source_l, opt.num_layers*opt.rnn_size*2)
            if source_l >= 2 then
              sent_context_grad[{{}, {1,source_l-1}}]:copy(sent_context_grad[{{}, {2,source_l}}])
            end
            sent_context_grad[{{}, {source_l}}]:zero()
            encoder_bow_grads:add(add_sent_context_init:backward(context_bow:contiguous(), sent_context_grad))
          end

          -- backward pos embeds
          if opt.pos_embeds == 1 and opt.hierarchical == 1 then
            local pos_grads = pos_grad_proto[{{1,batch_l*source_l}}]
            pos_grads:zero()
            for l = 1, opt.num_layers do
              pos_grads[{{}, {(l*2-2)*opt.rnn_size+1, (l*2-1)*opt.rnn_size}}]:copy(drnn_state_enc[l*2-1])
              pos_grads[{{}, {(l*2-1)*opt.rnn_size+1, (l*2)*opt.rnn_size}}]:copy(drnn_state_enc[l*2])
            end
            local pos = pos_proto[{{1,batch_l}, {1, source_l}}]:reshape(batch_l*source_l)
            pos_embeds:backward(pos, pos_grads)
          end

          -- encoder bow
          local drnn_state_bow_enc = reset_state(init_bwd_bow_enc, batch_l)
          if opt.hierarchical == 1 then
            if opt.bow_encoder_lstm == 1 then
              for t = source_l, 1, -1 do
                local bow_encoder_input = {bow_out[t], table.unpack(rnn_state_bow_enc[t-1])}
                drnn_state_bow_enc[#drnn_state_bow_enc]:add(encoder_bow_grads[{{},t}])
                local dlst = bow_encoder_lstm_clones[t]:backward(bow_encoder_input, drnn_state_bow_enc)
                for j = 1, #drnn_state_bow_enc do
                  drnn_state_bow_enc[j]:copy(dlst[j+1])
                end
                encoder_bow_lstm_grads[{{},t}]:copy(dlst[1])
              end

              local dl_dpos_embeds
              if opt.pos_embeds_sent == 1 then
                local pos = pos_proto[{{1,batch_l}, {1, source_l}}]
                dl_dpos_embeds = pos_embeds_sent:backward({rnn_states, pos},  encoder_bow_lstm_grads)[1]
              end

              -- backward through word embeds
              if opt.no_bow == 1 then
                local cur_grads = encoder_bow_lstm_grads
                if opt.pos_embeds_sent == 1 then
                  cur_grads = dl_dpos_embeds
                end
                local ctx_grads = masked_selecter:backward({context, rnn_state_mask}, cur_grads)[1]
                encoder_grads:add(ctx_grads)
              else
                bow_encoder:backward(source:permute(2,3,1):contiguous(), cur_grads:contiguous())
              end
            else
              if opt.no_bow == 1 then
                if opt.pos_embeds_sent == 1 then
                  local pos = pos_proto[{{1,batch_l}, {1, source_l}}]
                  encoder_bow_grads:add(pos_embeds_sent:backward({rnn_states, pos}, encoder_bow_grads)[1])
                end

                -- backward encoder_bow_grads to the LSTM sentence encoder
                local ctx_grads = masked_selecter:backward({context, rnn_state_mask}, encoder_bow_grads)[1]
                encoder_grads:add(ctx_grads)
              else
                if opt.pos_embeds_sent == 1 then
                  local pos = pos_proto[{{1,batch_l}, {1, source_l}}]
                  bow_encoder:backward({source:permute(2,3,1):contiguous(),pos}, encoder_bow_grads:contiguous())
                else
                  bow_encoder:backward(bow_source, encoder_bow_grads:contiguous())
                end
              end
            end

            if opt.no_bow == 0 then
              word_vecs_bow.gradWeight[1]:zero()
              if opt.fix_bow_vecs == 1 then
                 -- no update
                 word_vecs_bow.gradWeight:zero()
              end
            end
          end

          if opt.brnn == 1 then
            assert(false, 'fix pad mask stuff!')
            local drnn_state_enc = reset_state(init_bwd_enc, batch_l*source_l)
            if opt.init_dec == 1 then
              for L = 1, opt.num_layers do
                drnn_state_enc[L*2-1]:copy(init_dec_modules_bwd[L*2-1]:backward(rnn_state_enc_bwd[1][L*2-1], drnn_state_dec[L*2-1]))
                drnn_state_enc[L*2]:copy(init_dec_modules_bwd[L*2]:backward(rnn_state_enc_bwd[1][L*2], drnn_state_dec[L*2]))
              end
            end
            for t = 1, source_char_l do
              local encoder_input = {source[t], table.unpack(rnn_state_enc_bwd[t+1])}
              drnn_state_enc[#drnn_state_enc]:add(encoder_bwd_grads[{{},{},t}])
              local dlst = encoder_bwd_clones[t]:backward(encoder_input, drnn_state_enc)
              for j = 1, #drnn_state_enc do
                drnn_state_enc[j]:copy(dlst[j+1])
              end
            end	      	    
          end

          word_vec_layers[1].gradWeight[1]:zero()
          if opt.fix_word_vecs_enc == 1 then
            word_vec_layers[1].gradWeight:zero()
          end
          if opt.brnn == 1 then
            word_vec_layers[3].gradWeight[1]:zero()
          end

          if cur_soft_anneal then
            -- no need to sample many times with soft
            num_samples = 1
            break
          end
        end -- end sampling

        -- normalize by number of samples
        for j = 1, #grad_params do
          grad_params[j]:div(num_samples)
        end

        local grad_norm = 0
        local grad_stats = ''
        for name, idx in pairs(layers_idx) do
          local cur_norm = grad_params[idx]:norm()^2
          grad_norm = grad_norm + cur_norm
          grad_stats = grad_stats .. string.format('%s: %.2f, ', name, cur_norm)
        end
        if i % 50 == 0 then
          logging:info(grad_stats)
        end
        grad_norm = grad_norm^0.5	 

        if opt.hierarchical == 1 and opt.no_bow == 0 and opt.separate_vecs == 0 then
          word_vec_layers[1].gradWeight:add(word_vecs_bow.gradWeight)
        end
        if opt.brnn == 1 then
          word_vec_layers[1].gradWeight:add(word_vec_layers[3].gradWeight)
          --if opt.use_chars_enc == 1 then
            --for j = 1, charcnn_offset do
              --charcnn_grad_layers[j]:add(charcnn_grad_layers[j+charcnn_offset])
            --end
          --end	    
        end	 
        if opt.share_embed == 1 then
          word_vec_layers[1].gradWeight:add(word_vec_layers[2].gradWeight)
        end

        if opt.fix_encoder == 1 then
          grad_params[1]:zero()
        end
        if opt.fix_decoder == 1 then
          -- don't fix attention weights though
          local _, g = decoder_attn_layers[1]:parameters()
          local save_grads = {}
          for j = 1, #g do
            table.insert(save_grads, g[i]:clone())
          end
          grad_params[2]:zero()
          for j = 1, #g do
            g[i]:copy(save_grads[i])
          end
        end

        -- Shrink norm and update params
        local param_norm = 0
        local shrinkage = opt.max_grad_norm / grad_norm
        for j = 1, #grad_params do
          if opt.attn_type == 'hard' and opt.baseline_method == 'learned' and j == layers_idx['baseline_m'] then
            -- special case
            local n = grad_params[j]:norm()
            local s = opt.max_grad_norm / n
            if s < 1 then
              grad_params[j]:mul(s)
            end
          else
            if shrinkage < 1 then
              grad_params[j]:mul(shrinkage)
            end
          end

          if opt.learning_method == 'adagrad' then
            adagradStep(params[j], grad_params[j], layer_etas[j], optStates[j])
          elseif opt.learning_method == 'adam' then
            adamStep(params[j], grad_params[j], layer_etas[j], optStates[j])
          elseif opt.learning_method == 'sgd' then
            params[j]:add(grad_params[j]:mul(-layer_etas[j]))
          end	    
          param_norm = param_norm + params[j]:norm()^2
        end	 
        param_norm = param_norm^0.5
        if opt.hierarchical == 1 and opt.no_bow == 0 and opt.separate_vecs == 0 then
          word_vecs_bow.weight:copy(word_vec_layers[1].weight)
        end
        if opt.brnn == 1 then
          word_vec_layers[3].weight:copy(word_vec_layers[1].weight)
          --if opt.use_chars_enc == 1 then
            --for j = 1, charcnn_offset do
              --charcnn_layers[j+charcnn_offset]:copy(charcnn_layers[j])
            --end
          --end	    
        end
          if opt.share_embed == 1 then
            word_vec_layers[2].weight:copy(word_vec_layers[1].weight)
          end

        -- Bookkeeping
        num_words_target = num_words_target + batch_l*target_l
        num_words_source = num_words_source + batch_l*source_l
        train_nonzeros = train_nonzeros + nonzeros
        loss = loss / num_samples -- normalize
        train_loss = train_loss + loss*batch_l
        local time_taken = timer:time().real - start_time
        if i % opt.print_every == 0 then
          local stats = string.format('Epoch: %d, Batch: %d/%d, Batch size: %d, LR: %.4f, ',
          epoch, i, data:size(), batch_l, layer_etas[1])
          stats = stats .. string.format('PPL: %.2f, |Param|: %.2f, |GParam|: %.2f, ',
          math.exp(train_loss/train_nonzeros), param_norm, grad_norm)
          stats = stats .. string.format('Training: %d/%d/%d total/(source sentences)/target tokens/sec',
          (num_words_target+num_words_source) / time_taken,
          num_words_source / time_taken,
          num_words_target / time_taken)			   
          logging:info(stats)

            --logging:info('baseline:')
            --logging:info(opt.baseline)
            --if opt.moving_variance == 1 then
              --logging:info('variance:')
              --logging:info(opt.reward_variance)
            --end

        end
        if i % 200 == 0 then
          collectgarbage()
        end
      end
      return train_loss, train_nonzeros
    end

    local total_loss, total_nonzeros, batch_loss, batch_nonzeros
    for epoch = opt.start_epoch, opt.epochs do
      generator:training()
      if opt.num_shards > 0 then
        total_loss = 0
        total_nonzeros = 0	 
        local shard_order = torch.randperm(opt.num_shards)
        for s = 1, opt.num_shards do
          local fn = train_data .. '.' .. shard_order[s] .. '.hdf5'
          logging:info('loading shard #' .. shard_order[s])
          local shard_data = data.new(opt, fn)
          batch_loss, batch_nonzeros = train_batch(shard_data, epoch)
          total_loss = total_loss + batch_loss
          total_nonzeros = total_nonzeros + batch_nonzeros
        end
      else
        total_loss, total_nonzeros = train_batch(train_data, epoch)
      end
      local train_score = math.exp(total_loss/total_nonzeros)
      logging:info('Train ' .. train_score)
      logging:info(opt.train_perf)

      opt.train_perf[#opt.train_perf + 1] = train_score

      local score = eval(valid_data)
      opt.val_perf[#opt.val_perf + 1] = score
      logging:info(opt.val_perf)
      if opt.learning_method == 'sgd' then --unncessary with adagrad
        decay_lr(epoch)
      end      
      -- clean and save models
      local savefile = string.format('%s_epoch%d_%.2f.t7', opt.savefile, epoch, score)      
      if epoch % opt.save_every == 0 then
        logging:info('saving checkpoint to ' .. savefile)
        clean_layer(generator)
        save_checkpoint(savefile)
      end      

    end
    logging:info('done!')
  end

  function eval(data)
    encoder_clones[1]:evaluate()   
    decoder_clones[1]:evaluate() -- just need one clone
    decoder_attn_clones[1]:evaluate()
    generator:evaluate()
    if opt.brnn == 1 then
      encoder_bwd_clones[1]:evaluate()
    end
    if opt.hierarchical == 1 then
      if opt.no_bow == 0 then
        bow_encoder:evaluate()
      end
      if opt.bow_encoder_lstm == 1 then
        bow_encoder_lstm_clones[1]:evaluate()
      end
      if opt.pos_embeds == 1 then
        pos_embeds:evaluate()
      end
    end

    local nll = 0
    local total = 0
    for i = 1, data:size() do
      local d = data[i]
      local target, target_out, nonzeros, source = d[1], d[2], d[3], d[4]
      local batch_l, target_l, source_l, target_l_all = d[5], d[6], d[7], d[8]
      local source_char_l = d[9]
      local bow_source
      if opt.all_lstm == 1 then
        bow_source = source:transpose(1,2):reshape(batch_l, source_l, source_char_l)
      else
        bow_source = source:permute(2,3,1):contiguous()
      end
      local context = context_proto[{{1, batch_l}, {1, source_l}, {1, source_char_l}}]
      local context_bow = context_bow_proto[{{1, batch_l}, {1, source_l}}]
      local rnn_state_bow_enc
      if opt.hierarchical == 1 and opt.bow_encoder_lstm == 1 then
        rnn_state_bow_enc = reset_state(init_fwd_bow_enc, batch_l)
      end
      local rnn_state_mask
      if opt.no_bow == 1 then
        local source_sent_l = bow_source:ne(1):sum(3):cuda():squeeze(3) -- batch_l x source_l
        rnn_state_mask = torch.zeros(batch_l, source_l, source_char_l, opt.rnn_size):cudaByte()
        for j = 1, batch_l do
          for t = 1, source_l do
            local idx = source_sent_l[j][t]
            if idx == 0 then idx = 1 end
            rnn_state_mask[j][t][idx]:fill(1)
          end
        end
      end

      local rnn_state_enc
      if opt.all_lstm == 1 then
        rnn_state_enc = reset_state(init_fwd_enc, batch_l)
      else
        rnn_state_enc = reset_state(init_fwd_enc, batch_l*source_l)
      end

      -- pos embeds
      if opt.pos_embeds == 1 and opt.hierarchical == 1 then
        local pos = pos_proto[{{1,batch_l}, {1, source_l}}]:reshape(batch_l*source_l)
        local pos_states = pos_embeds:forward(pos) -- batch_l*source_l x num_layers*rnn_size*2

        for l = 1, opt.num_layers do
          rnn_state_enc[l*2-1]:copy(pos_states[{{},{(l*2-2)*opt.rnn_size+1, (l*2-1)*opt.rnn_size}}])
          rnn_state_enc[l*2]:copy(pos_states[{{},{(l*2-1)*opt.rnn_size+1, (l*2)*opt.rnn_size}}])
        end
      end

      -- TODO fix this... context_bow hasn't been initialized yet
      --local sent_context
      --if opt.add_sent_context_init == 1 then
        --sent_context = add_sent_context_init:forward(context_bow:contiguous())
        --if source_l >= 2 then
          --sent_context[{{}, {2,source_l}}]:copy(sent_context[{{}, {1,source_l-1}}])
        --end
        --sent_context[{{}, {1}}]:zero()
        --sent_context = sent_context:view(batch_l*source_l, opt.num_layers*opt.rnn_size*2)

        --for l = 1, opt.num_layers do
          --rnn_state_enc[l*2-1]:copy(sent_context[{{},{(l*2-2)*opt.rnn_size+1, (l*2-1)*opt.rnn_size}}])
          --rnn_state_enc[l*2]:copy(sent_context[{{},{(l*2-1)*opt.rnn_size+1, (l*2)*opt.rnn_size}}])
        --end
      --elseif opt.add_sent_context == 1 then
        --sent_context = torch.zeros(batch_l, source_l, opt.bow_size):cuda()
        --if source_l >= 2 then
          --sent_context[{{}, {2,source_l}}]:copy(context_bow[{{}, {1,source_l-1}}])
        --end
      --end

      -- forward prop encoder
      if opt.all_lstm == 1 then
        for t = 1, source_char_l*source_l do
          local encoder_input = {source[t], table.unpack(rnn_state_enc)}
          local out = encoder_clones[1]:forward(encoder_input)
          rnn_state_enc = out
          local t1, t2 = get_indices(t, source_char_l)
          context[{{},t1,t2}]:copy(out[#out])
        end
      else
        for t = 1, source_char_l do
          local encoder_input
          if opt.add_sent_context == 1 then
            encoder_input = {source[t], sent_context, table.unpack(rnn_state_enc)}
          else
            encoder_input = {source[t], table.unpack(rnn_state_enc)}
          end
          local out = encoder_clones[1]:forward(encoder_input)

          if opt.mask_padding == 1 then
            local cur_mask = source[t]:eq(1)
            cur_mask = cur_mask:view(batch_l*source_l, 1):expand(batch_l*source_l, opt.rnn_size)
            for L = 1, opt.num_layers do
              out[L*2-1]:maskedFill(cur_mask, 0)
              out[L*2]:maskedFill(cur_mask, 0)
            end
          end

          rnn_state_enc = out
          context[{{},{},t}]:copy(out[#out]:view(batch_l, source_l, opt.rnn_size))
        end	 
      end

      -- bow encoder
      local rnn_states
      local bow_out 
      if opt.hierarchical == 1 then
        if opt.no_bow == 1 then
          masked_selecter = make_last_state_selecter(opt, batch_l, source_l)
          rnn_states = masked_selecter:forward({context, rnn_state_mask})
          if opt.pos_embeds_sent == 1 then
            local pos = pos_proto[{{1,batch_l}, {1, source_l}}]
            bow_out = pos_embeds_sent:forward({rnn_states, pos})
          else
            bow_out = rnn_states
          end
        else
          if opt.pos_embeds_sent == 1 then
            local pos = pos_proto[{{1,batch_l}, {1, source_l}}]
            bow_out = bow_encoder:forward({bow_source, pos})
          else
            bow_out = bow_encoder:forward(bow_source)
          end
        end

        if opt.bow_encoder_lstm == 1 then
          -- pass bag of words through LSTM over sentences for context
          for t = 1, source_l do
            local bow_encoder_input = {bow_out[t], table.unpack(rnn_state_bow_enc)}
            local out = bow_encoder_lstm_clones[1]:forward(bow_encoder_input)
            rnn_state_bow_enc = out
            context_bow[{{}, t}]:copy(out[#out])
          end
        else
          context_bow:copy(bow_out)
        end
      end

      local rnn_state_dec = reset_state(init_fwd_dec, batch_l)
      if opt.init_dec == 1 then
        init_dec_module = make_init_dec_module(opt, batch_l, source_l)
        for L = 1, opt.num_layers do
          rnn_state_dec[L*2-1+opt.input_feed]:copy(init_dec_module:forward(rnn_state_enc[L*2-1]))
          rnn_state_dec[L*2+opt.input_feed]:copy(init_dec_module:forward(rnn_state_enc[L*2]))
        end	 
      end

      if opt.brnn == 1 then
        local rnn_state_enc = reset_state(init_fwd_enc, batch_l*source_l)
        for t = source_char_l, 1, -1 do
          local encoder_input = {source[t], table.unpack(rnn_state_enc)}
          local out = encoder_bwd_clones[1]:forward(encoder_input)
          rnn_state_enc = out
          context[{{},{},t}]:add(out[#out]:view(batch_l, source_l, opt.rnn_size))
        end
        if opt.init_dec == 1 then
          for L = 1, opt.num_layers do
            rnn_state_dec[L*2-1+opt.input_feed]:add(init_dec_module:forward(rnn_state_enc[L*2-1]))
            rnn_state_dec[L*2+opt.input_feed]:add(init_dec_module:forward(rnn_state_enc[L*2]))
          end
        end	 
      end      	 	 

      local inf_mask
      if opt.hierarchical == 1 and opt.inf_mask == 1 then
        inf_mask = source:permute(2,3,1):contiguous():eq(1):mul(-math.huge):cuda() -- batch_l x source_l x source_char_l
      end

      local loss = 0
      for t = 1, target_l do
        local decoder_input = {target[t], table.unpack(rnn_state_dec)}
          --decoder_input = {target[t], context, table.unpack(rnn_state_dec)}
        local out = decoder_clones[1]:forward(decoder_input)
        local decoder_attn_input
        if opt.hierarchical == 1 then
          if opt.coarse_attn_only == 1 then
            decoder_attn_input = {out[#out], context_bow}
          else
            decoder_attn_input = {out[#out], context, context_bow}
          end
          if opt.inf_mask == 1 then
            table.insert(decoder_attn_input, inf_mask)
          end
        else
          decoder_attn_input = {out[#out], context}
        end
        local attn_out = decoder_attn_clones[1]:forward(decoder_attn_input)

        rnn_state_dec = {}
        if opt.input_feed == 1 then
          table.insert(rnn_state_dec, attn_out)
        end	 
        for j = 1, #out do
          table.insert(rnn_state_dec, out[j])
        end
        local pred = generator:forward(attn_out)
        loss = loss + criterion:forward(pred, target_out[t])
      end
      nll = nll + loss
      total = total + nonzeros
    end
    local valid = math.exp(nll / total)
    logging:info("Valid "..valid)
    collectgarbage()
    return valid
  end


function get_layer(layer)
   if layer.name ~= nil then
      if layer.name == 'word_vecs_dec' then
         table.insert(word_vec_layers, layer)
      elseif layer.name == 'word_vecs_enc' then
         table.insert(word_vec_layers, layer)
      elseif layer.name == 'word_vecs_bow' then
         word_vecs_bow = layer
      --elseif layer.name == 'charcnn_enc' or layer.name == 'mlp_enc' then
         --local p, gp = layer:parameters()
         --for i = 1, #p do
            --table.insert(charcnn_layers, p[i])
            --table.insert(charcnn_grad_layers, gp[i])
         --end	 
      end
   end
end

function get_RL_layer(layer)
   if layer.name ~= nil then
      if layer.name == 'decoder_attn' then
        table.insert(decoder_attn_layers, layer)
      elseif layer.name == 'sampler' then
        table.insert(sampler_layers, layer)
      --elseif layer.name == 'sampler_word' then
        --table.insert(sampler_word_layers, layer)
      --elseif layer.name == 'mul_constant' then
        --table.insert(mul_constant_layers, layer)
      end
   end
end

function copy_params(targ, src)
  local targ_params = targ:parameters()
  local src_params = src:parameters()

  if torch.isTensor(targ_params) then
    targ_params:copy(src_params) 
  else
    -- table
    for i,p in ipairs(targ_params) do
      p:copy(src_params[i])
    end
  end
end

function main() 
    -- parse input params
   opt = cmd:parse(arg)
   assert(opt.log_path ~= '', 'need to set logging')
   logging = logger(opt.log_path)
   logging:info("Command line args:")
   logging:info(arg)
   logging:info("End command line args")
   if opt.gpuid >= 0 then
      logging:info('using CUDA on GPU ' .. opt.gpuid .. '...')
      require 'cutorch'
      require 'cunn'
      if opt.cudnn == 1 then
	 logging:info('loading cudnn...')
	 require 'cudnn'
      end      
      cutorch.setDevice(opt.gpuid)
      cutorch.manualSeed(opt.seed)      
   end

   -- Create the data loader class.
   logging:info('loading data...')
   if opt.num_shards == 0 then
      train_data = data.new(opt, opt.data_file)
   else
      train_data = opt.data_file
   end

   valid_data = data.new(opt, opt.val_data_file)
   logging:info('done!')
   logging:info(string.format('Source vocab size: %d, Target vocab size: %d',
		       valid_data.source_size, valid_data.target_size))   
   opt.max_sent_l_src = valid_data.source:size(2)
   opt.max_sent_l_targ = valid_data.target:size(2)
   opt.max_sent_l = math.max(opt.max_sent_l_src, opt.max_sent_l_targ)
   if opt.max_batch_l == '' then
      opt.max_batch_l = valid_data.batch_l:max()
   end

   if opt.use_chars_enc == 1 or opt.use_chars_dec == 1 then
      opt.max_word_l = valid_data.source_char_l:max()
   end

   -- set output size of bow encoder
   opt.bow_size = 0
   if opt.bow_encoder_lstm == 1 then
     opt.bow_size = opt.rnn_size
   else
     if opt.no_bow == 1 then
       opt.bow_size = opt.rnn_size
     else
       if opt.conv_bow == 1 then
         opt.bow_size = opt.num_kernels
       else
         opt.bow_size = opt.word_vec_size
         if opt.linear_bow == 1 then
           opt.bow_size = opt.linear_bow_size
         end
       end
     end
     if opt.pos_embeds_sent == 1 then
       opt.bow_size = opt.bow_size + opt.pos_dim
     end
   end

   logging:info(string.format('Source max doc len: %d, Target max sent len: %d',
		       valid_data.source:size(2), valid_data.target:size(2)))   
   logging:info(string.format('Source max sent len: %d', opt.max_word_l))

   preallocateMemory(opt.prealloc)

   -- Build model
   if opt.train_from:len() == 0 or opt.start_soft == 1 then
      encoder = make_lstm(valid_data, opt, 'enc', opt.use_chars_enc)
      decoder = make_lstm(valid_data, opt, 'dec', opt.use_chars_dec)
      if opt.hierarchical == 1 then
        decoder_attn = make_hierarchical_decoder_attn(valid_data, opt)
        if opt.no_bow == 0 then
          bow_encoder = make_bow_encoder(valid_data, opt)
        end
        if opt.pos_embeds == 1 then
          pos_embeds = make_pos_embeds(valid_data, opt)
        end
        if opt.pos_embeds_sent == 1 and opt.no_bow == 1 then
          pos_embeds_sent = make_pos_embeds_sent(valid_data, opt)
        end
        if opt.bow_encoder_lstm == 1 then
          bow_encoder_lstm = make_lstm(valid_data, opt, 'bow_enc', 0)
        end
        if opt.add_sent_context_init == 1 then
          add_sent_context_init = make_add_sent_context_init(valid_data, opt)
        end
      else
        decoder_attn = make_decoder_attn(valid_data, opt)
      end
      generator, criterion = make_generator(valid_data, opt)
      if opt.brnn == 1 then
	 encoder_bwd = make_lstm(valid_data, opt, 'enc', opt.use_chars_enc)
      end      

      if opt.start_soft == 1 then
        -- Load params
        assert(path.exists(opt.train_from), 'checkpoint path invalid')
        logging:info('loading ' .. opt.train_from .. '...')
        local checkpoint = torch.load(opt.train_from)
        local model, model_opt = checkpoint[1], checkpoint[2]
        opt.num_layers = model_opt.num_layers
        opt.rnn_size = model_opt.rnn_size
        opt.input_feed = model_opt.input_feed
        opt.init_dec = model_opt.init_dec
        opt.brnn = model_opt.brnn
        opt.hierarchical = model_opt.hierarchical

        -- copy params
        local save_idx = model_opt.save_idx
        copy_params(encoder, model[1]:double())
        copy_params(decoder, model[2]:double())
        copy_params(generator, model[3]:double())
        copy_params(decoder_attn, model[4]:double())
        if model_opt.brnn == 1 then
          copy_params(encoder_bwd, model[save_idx['encoder_bwd']]:double())
        end      
        if model_opt.hierarchical == 1 and model_opt.no_bow == 0 then
          copy_params(bow_encoder, model[save_idx['bow_encoder']]:double())
        end
      end

      if opt.attn_type == 'hard' then
        if opt.baseline_method == 'learned' then
          logging:info('using learned baseline method')
          baseline_m, reward_criterion = make_reinforce(valid_data, opt)
        else
          _, reward_criterion = make_reinforce(valid_data, opt)
        end
        opt.baseline = 0 -- RL average

        if opt.moving_variance == 1 then
          opt.reward_variance = 0 -- RL stddev
        end
      end

      -- check for word2vec
      if opt.train_from == '' then
        if opt.synth_data == 0 then
          assert(opt.pre_word_vecs_enc ~= '', 'not using word2vec!')
        end
        opt.pre_word_vecs_dec = opt.pre_word_vecs_enc
      end
   else
      assert(path.exists(opt.train_from), 'checkpoint path invalid')
      logging:info('loading ' .. opt.train_from .. '...')
      local checkpoint = torch.load(opt.train_from)
      local model, model_opt = checkpoint[1], checkpoint[2]
      opt.num_layers = model_opt.num_layers
      opt.rnn_size = model_opt.rnn_size
      opt.input_feed = model_opt.input_feed
      opt.init_dec = model_opt.init_dec
      opt.brnn = model_opt.brnn
      opt.hierarchical = model_opt.hierarchical

      local save_idx = model_opt.save_idx
      encoder = model[1]:double()
      decoder = model[2]:double()      
      generator = model[3]:double()
      if model_opt.coarse_attn_only == 1 and opt.coarse_attn_only == 0 then
        -- copy weights
        decoder_attn = make_hierarchical_decoder_attn(valid_data, opt)
        local decoder_attn_params = model[4]:double():parameters()
        local targ_params = decoder_attn:parameters()

        -- copy only 1 for coarse attn bilinear
        targ_params[1]:copy(decoder_attn_params[1])
      else
        decoder_attn = model[4]:double()
      end
      if model_opt.brnn == 1 then
        encoder_bwd = model[save_idx['encoder_bwd']]:double()
      end      
      if model_opt.hierarchical == 1 then
        if model_opt.pos_embeds == 1 then
          pos_embeds = model[save_idx['pos_embeds']]:double()
        end
        if model_opt.no_bow == 0 then
          bow_encoder = model[save_idx['bow_encoder']]:double()
        end
      end

      if model_opt.attn_type == 'hard' then
        if model_opt.baseline_method == 'learned' then
          baseline_m = model[save_idx['baseline_m']]:double()
        end
        if model_opt.baseline_method == 'average' then
          opt.baseline_method = 'average'
          opt.baseline = model_opt.baseline
        end
        if model_opt.moving_variance == 1 then
          opt.moving_variance = 1 
          opt.reward_variance = model_opt.reward_variance
        end
      end

      _, criterion = make_generator(valid_data, opt)
      if opt.attn_type == 'hard' then
        _, reward_criterion = make_reinforce(valid_data, opt)
      end
   end   

   -- print options
   logging:info('init dec: ' .. opt.init_dec)
   logging:info('input feed: ' .. opt.input_feed)
   logging:info('attention on sentences:', opt.attn_type)
   if opt.hierarchical == 1 then
     logging:info('doing hierarchical model')
   else
     logging:info('doing full attention over doc')
   end
   if opt.mask_padding == 1 then
     logging:info('masking padding for encoder hidden states')
   end
   if opt.baseline_method == 'learned' then
     logging:info('using learned baseline method')
   end
   if opt.no_bow == 1 then
     logging:info('using no bow')
   end
   if opt.bow_encoder_lstm == 1 then
     logging:info('using bow encoder lstm')
   end
   if opt.conv_bow == 1 then
     assert(opt.cudnn == 1, 'use cudnn!')
     logging:info('using convolution instead of bag of words')
   end
   --assert(opt.multisampling > 0, 'please use multisampling')
   if opt.multisampling > 0 then
     --assert(opt.hop_attn > 1 or opt.multisampling > 1, 'please do more than one sample')
     logging:info(string.format('sampling attn %d instead of once', opt.multisampling))
   else
     logging:info('NOT multisampling')
   end

   layers = {encoder, decoder, generator, decoder_attn}
   layers_idx = {encoder=1, decoder=2, generator=3, decoder_attn=4}
   idx = 5
   if opt.hierarchical == 1 then
     if opt.no_bow == 0 then
       table.insert(layers, bow_encoder)
       layers_idx['bow_encoder'] = idx
       idx = idx + 1
     end
     if opt.pos_embeds == 1 then
       table.insert(layers, pos_embeds)
       layers_idx['pos_embeds'] = idx
       idx = idx + 1
     end
     if opt.pos_embeds_sent == 1 and opt.no_bow == 1 then
       table.insert(layers, pos_embeds_sent)
       layers_idx['pos_embeds_sent'] = idx
       idx = idx + 1
     end
     if opt.bow_encoder_lstm == 1 then
       table.insert(layers, bow_encoder_lstm)
       layers_idx['bow_encoder_lstm'] = idx
       idx = idx + 1
     end
     if opt.add_sent_context_init == 1 then
       table.insert(layers, add_sent_context_init)
       layers_idx['add_sent_context_init'] = idx
       idx = idx + 1
     end
   end
   if opt.attn_type == 'hard' and opt.baseline_method == 'learned' then
     table.insert(layers, baseline_m)
     layers_idx['baseline_m'] = idx
     idx = idx + 1
   end
   if opt.brnn == 1 then
      table.insert(layers, encoder_bwd)
      layers_idx['encoder_bwd'] = idx
      idx = idx + 1
   end

   assert(opt.learning_method == 'sgd' or opt.learning_method == 'adagrad' or opt.learning_method == 'adam', 'unsupported learning method!')
   layer_etas = {} -- different learning rates
   if opt.learning_method == 'sgd' then
      for i = 1, #layers do
         layer_etas[i] = opt.learning_rate
      end
      -- lower for bow_encoder
      layer_etas[layers_idx['bow_encoder']] = opt.sent_learning_rate
      if opt.baseline_method == 'learned' then
          layer_etas[layers_idx['bow_encoder']] = opt.baseline_learning_rate
      end
   else
      -- adagrad and the rest
      optStates = {}
      for i = 1, #layers do
         layer_etas[i] = opt.learning_rate
         optStates[i] = {}
      end     
   end

   if opt.gpuid >= 0 then
      for i = 1, #layers do	 
	 layers[i]:cuda()
      end
      criterion:cuda()      
      if opt.attn_type == 'hard' then
        reward_criterion:cuda()
      end
   end

   -- these layers will be manipulated during training
   word_vec_layers = {}
   if opt.use_chars_enc == 1 then
      charcnn_layers = {}
      charcnn_grad_layers = {}
   end
   encoder:apply(get_layer)   
   decoder:apply(get_layer)
   if opt.hierarchical == 1 and opt.no_bow == 0 then
     bow_encoder:apply(get_layer)
   end
   if opt.brnn == 1 then
      --if opt.use_chars_enc == 1 then
	 --charcnn_offset = #charcnn_layers
      --end      
      encoder_bwd:apply(get_layer)
   end   

   train(train_data, valid_data)
end

main()
