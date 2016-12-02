--
-- Manages encoder/decoder data matrices.
--

local data = torch.class("data")

function data:__init(opt, data_file)
   local f = hdf5.open(data_file, 'r')
   
   self.source  = f:read('source'):all()
   self.target  = f:read('target'):all()   
   self.target_output = f:read('target_output'):all()
   self.target_l = f:read('target_l'):all() --max target length each batch
   self.target_l_all = f:read('target_l_all'):all()
   self.target_l_all:add(-1)
   self.batch_l = f:read('batch_l'):all()
   self.source_l = f:read('batch_w'):all() --max source length each batch
   if opt.start_symbol == 0 then
     if opt.no_pad == 0 then
        self.source_l:add(-2)
        self.source = self.source[{{},{2, self.source:size(2)-1}}]
     end
   end   
   self.batch_idx = f:read('batch_idx'):all()
   
   self.target_size = f:read('target_size'):all()[1]
   --self.source_size = f:read('source_size'):all()[1]
   self.source_size = f:read('char_size'):all()[1]
   self.target_nonzeros = f:read('target_nonzeros'):all()
  
   if opt.use_chars_enc == 1 then
      self.source_char = f:read('source_char'):all() -- N x max_sent_l x max_word_l
      self.source_char_l = f:read('source_char_l'):all() -- max source word each batch
      self.char_size = f:read('char_size'):all()[1]
      --self.char_length = self.source_char:size(3)
      if opt.start_symbol == 0 then
         if opt.no_pad == 0 then
           self.source_char = self.source_char[{{}, {2, self.source_char:size(2)-1}}] -- doc

           -- assumes end padding
           self.source_char_l:add(-2)
           self.source_char = self.source_char[{{},{},{2, self.source_char:size(3)-1}}] -- get rid of start,end token
           self.source_char[self.source_char:eq(4)] = 1 -- replace EOS with pad
         else
           logging:info('using no_pad = 1')
         end
      end      
   end
   
   if opt.use_chars_dec == 1 then
      self.target_char = f:read('target_char'):all()
      self.char_size = f:read('char_size'):all()[1]
      self.char_length = self.target_char:size(3)      
   end   
   
   self.length = self.batch_l:size(1)
   self.seq_length = self.target:size(2) 
   self.batches = {}
   local max_source_l = self.source_l:max()   
   local source_l_rev = torch.ones(max_source_l):long()
   for i = 1, max_source_l do
      source_l_rev[i] = max_source_l - i + 1
   end   
   for i = 1, self.length do
      local source_i, target_i
      local target_output_i = self.target_output:sub(self.batch_idx[i],self.batch_idx[i]
							+self.batch_l[i]-1, 1, self.target_l[i])
      local target_l_i = self.target_l_all:sub(self.batch_idx[i],
					       self.batch_idx[i]+self.batch_l[i]-1)
      if opt.use_chars_enc == 1 then
         source_i = self.source_char:sub(self.batch_idx[i],
                                              self.batch_idx[i] + self.batch_l[i]-1, 1,
                                              self.source_l[i], 1, self.source_char_l[i])
         if opt.all_lstm == 1 then
           source_i = source_i:reshape(source_i:size(1), self.source_l[i]*self.source_char_l[i])
           source_i = source_i:transpose(1,2):contiguous()
         else
           source_i = source_i:permute(3,1,2):contiguous()
                                                -- permute to get words, batch, sents
          end
      else
	 source_i =  self.source:sub(self.batch_idx[i], self.batch_idx[i]+self.batch_l[i]-1,
				     1, self.source_l[i]):transpose(1,2)
      end
      if opt.reverse_src == 1 then
	 source_i = source_i:index(1, source_l_rev[{{max_source_l-self.source_l[i]+1,
						     max_source_l}}])
      end      

      if opt.use_chars_dec == 1 then
	 target_i = self.target_char:sub(self.batch_idx[i],
					      self.batch_idx[i] + self.batch_l[i]-1, 1,
					      self.target_l[i]):transpose(1,2):contiguous()
      else
	 target_i = self.target:sub(self.batch_idx[i], self.batch_idx[i]+self.batch_l[i]-1,
				    1, self.target_l[i]):transpose(1,2)
      end
      table.insert(self.batches,  {target_i,
				  target_output_i:transpose(1,2),
				  self.target_nonzeros[i], 
				  source_i,
				  self.batch_l[i],
				  self.target_l[i],
				  self.source_l[i],
				  target_l_i,
                                  self.source_char_l[i]})
   end

end

function data:size()
   return self.length
end

function data.__index(self, idx)
   if type(idx) == "string" then
      return data[idx]
   else
      local target_input = self.batches[idx][1]
      local target_output = self.batches[idx][2]
      local nonzeros = self.batches[idx][3]
      local source_input = self.batches[idx][4]      
      local batch_l = self.batches[idx][5]
      local target_l = self.batches[idx][6]
      local source_l = self.batches[idx][7]
      local target_l_all = self.batches[idx][8]
      local source_char_l = self.batches[idx][9]
      if opt.gpuid >= 0 then --if multi-gpu, source lives in gpuid1, rest on gpuid2
	 --cutorch.setDevice(opt.gpuid)
	 source_input = source_input:cuda()
	 target_input = target_input:cuda()
	 target_output = target_output:cuda()
	 target_l_all = target_l_all:cuda()
      end
      return {target_input, target_output, nonzeros, source_input,
	      batch_l, target_l, source_l, target_l_all, source_char_l}
   end
end

function data:get_worst()
  local worst_idx = 1
  local worst = 0
  local worst_stats = {}
  for idx = 1, self:size() do
      local batch_l = self.batches[idx][5]
      local target_l = self.batches[idx][6]
      local source_l = self.batches[idx][7]
      local source_char_l = self.batches[idx][9]
      local ans = batch_l*target_l*source_l*source_char_l
      if ans > worst then
        worst_idx = idx
        worst = ans
        worst_stats = {batch_l, target_l, source_l, source_char_l}
      end
  end
  return worst_idx, table.unpack(worst_stats)
end

return data
