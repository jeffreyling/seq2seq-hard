-- From dpnn library
--
require 'nn'

function nn.Module:toBatch(tensor, nDim, batchDim)
  local batchDim = batchDim or 1
  if tensor:dim() == nDim then
    self.dpnn_online = true
    local size = tensor:size():totable()
    table.insert(size, batchDim, 1)
    tensor = tensor:view(table.unpack(size))
  else
    self.dpnn_online = false
  end
  return tensor
end

function nn.Module:reinforce(reward)
  if self.modules then
    for i, module in ipairs(self.modules) do
      module:reinforce(reward)
    end
  end
end

nn.Criterion.toBatch = nn.Module.toBatch

--
------------------------------------------------------------------------
--[[ Reinforce ]]--
-- Ref A. http://incompleteideas.net/sutton/williams-92.pdf
-- Abstract class for modules that use the REINFORCE algorithm (ref A).
-- The reinforce(reward) method is called by a special Reward Criterion.
-- After which, when backward is called, the reward will be used to 
-- generate gradInputs. The gradOutput is usually ignored.
------------------------------------------------------------------------
local Reinforce, parent = torch.class("nn.Reinforce", "nn.Module")

function Reinforce:__init()
   parent.__init(self)
end

-- a Reward Criterion will call this
function Reinforce:reinforce(reward)
   self.reward = reward
end

function Reinforce:updateOutput(input)
   self.output:set(input)
end

function Reinforce:updateGradInput(input, gradOutput)
   local reward = self:rewardAs(input)
   self.gradInput:resizeAs(reward):copy(reward)
end

-- this can be called by updateGradInput
function Reinforce:rewardAs(input)
   assert(self.reward:dim() == 1)
   if input:isSameSizeAs(self.reward) then
      return self.reward
   else
      if self.reward:size(1) ~= input:size(1) then
         -- assume input is in online-mode
         input = self:toBatch(input, input:dim())
         assert(self.reward:size(1) == input:size(1), self.reward:size(1).." ~= "..input:size(1))
      end
      self._reward = self._reward or self.reward.new()
      self.__reward = self.__reward or self.reward.new()
      local size = input:size():fill(1):totable()
      size[1] = self.reward:size(1)
      self._reward:view(self.reward, table.unpack(size))
      self.__reward:expandAs(self._reward, input)
      return self.__reward
   end
end

------------------------------------------------------------------------
--[[ ReinforceCategorical ]]-- 
-- Ref A. http://incompleteideas.net/sutton/williams-92.pdf
-- Inputs are a vector of categorical prob : (p[1], p[2], ..., p[k]) 
-- Ouputs are samples drawn from this distribution.
-- Uses the REINFORCE algorithm (ref. A sec 6. p.230-236) which is 
-- implemented through the nn.Module:reinforce(r,b) interface.
-- gradOutputs are ignored (REINFORCE algorithm).
------------------------------------------------------------------------
local ReinforceCategorical, parent = torch.class("nn.ReinforceCategorical", "nn.Reinforce")

function ReinforceCategorical:__init(semi_sampling_p, entropy_scale)
  parent.__init(self)
  self.semi_sampling_p = semi_sampling_p or 1
  self.entropy_scale = entropy_scale or 0
  self.through = false -- pass prob weights through

  self.time_step = 0
  self.oracle = false -- stupid hack
end

function ReinforceCategorical:_doArgmax(input)
   _, self._index = input:max(2)
   self.output:zero()
   self.output:scatter(2, self._index, 1)
end

function ReinforceCategorical:_doSample(input)
   self._do_sample = (torch.uniform() < self.semi_sampling_p)
   if self._do_sample == false then
      -- use p
      self.output:copy(input)
   else
      -- sample from categorical with p = input
      self._input = self._input or input.new()
      -- prevent division by zero error (see updateGradInput)
      self._input:resizeAs(input):copy(input):add(0.00000001) 
      input.multinomial(self._index, input, 1)
      -- one hot encoding
      self.output:zero()
      self.output:scatter(2, self._index, 1)
   end
end

function ReinforceCategorical:updateOutput(input)
   self.output:resizeAs(input)
   self._index = self._index or ((torch.type(input) == 'torch.CudaTensor') and torch.CudaTensor() or torch.LongTensor())
   if self.through == true then
     -- identity
     self.output:copy(input)
   else
     if self.oracle and self.time_step <= self.output:size(2) then -- stupid hack
       self.output:zero()
       self.output:select(2,self.time_step):fill(1) -- very stupid hack
     else
       if self.train then
          --sample
          self:_doSample(input)
       else
         assert(self.train == false)
         -- do argmax at test time
         self:_doArgmax(input)
       end
     end
   end
   return self.output
end

function ReinforceCategorical:updateGradInput(input, gradOutput)
   -- Note that gradOutput is ignored
   -- f : categorical probability mass function
   -- x : the sampled indices (one per sample) (self.output)
   -- p : probability vector (p[1], p[2], ..., p[k]) 
   -- derivative of log categorical w.r.t. p
   -- d ln(f(x,p))     1/p[i]    if i = x  
   -- ------------ =   
   --     d p          0         otherwise
   self.gradInput:resizeAs(input):zero()
   if self.through then
     -- identity function
     self.gradInput:copy(gradOutput)
   else 
     self.gradInput:copy(self.output)
     self._input = self._input or input.new()
     -- prevent division by zero error
     self._input:resizeAs(input):copy(input):add(0.00000001) 
     self.gradInput:cdiv(self._input)
     
     -- multiply by reward 
     self.gradInput:cmul(self:rewardAs(input))
     -- add entropy term
     self._gradEnt = self._input:clone()
     self._gradEnt:log():add(1)
     self.gradInput:add(self.entropy_scale, self._gradEnt)

     -- multiply by -1 ( gradient descent on input )
     self.gradInput:mul(-1)
   end
   return self.gradInput
end

function ReinforceCategorical:type(type, tc)
   self._index = nil
   return parent.type(self, type, tc)
end

-- stupid hack
function nn.MulConstant:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(gradOutput)
  self.gradInput:copy(gradOutput)
  return self.gradInput
end

-- Modified ClassNLLCriterion
local ReinforceNLLCriterion, parent = torch.class("nn.ReinforceNLLCriterion", "nn.Criterion")

function ReinforceNLLCriterion:__init(scale, criterion)
   parent.__init(self)
   self.scale = scale or 1 -- scale of reward
   self.sizeAverage = true
   self.criterion = criterion or nn.MSECriterion() -- baseline criterion

   self.gradInput = {torch.Tensor()}
end

function ReinforceNLLCriterion:updateOutput(inputTable, target)
   assert(torch.type(inputTable) == 'table')
   local input = inputTable[1]
   local b = inputTable[2]
   local mask = inputTable[3]

   if type(target) == 'number' then
     if input:type() ~= 'torch.CudaTensor' then
       self.target = self.target:long()
     end
     self.target[1] = target
   elseif target:type() == 'torch.CudaTensor' then
     self.target = target
   else
     self.target = target:long()
   end

   self.reward = self.reward or input.new()
   self.reward = input:gather(2,target:view(target:size(1), 1))
   self.reward:resize(input:size(1))

   -- subtract baseline
   self.vrReward = self.vrReward or self.reward.new()
   self.vrReward:resizeAs(self.reward):copy(self.reward)
   if type(b) == 'number' then
     self.vrReward:add(-b)
   else
     self.vrReward:add(-1, b)
   end
   self.vrReward:mul(self.scale)
   if self.sizeAverage then
      self.vrReward:div(input:size(1))
   end
   self.vrReward:maskedFill(mask, 0)

   -- loss = -sum(reward) aka NLL
   -- this actually doesn't matter, we won't use it
   self.output = -self.reward:sum()
   return self.output
end

function ReinforceNLLCriterion:updateGradInput(inputTable, target)
  local input = inputTable[1]
  local b = inputTable[2]
  local mask = inputTable[3]

  -- zero gradInput
  self.gradInput[1]:resizeAs(input):zero()
  -- baseline grad
  self.criterion:forward(b, self.reward)
  self.gradInput[2] = self.criterion:backward(b, self.reward)
  self.gradInput[2]:maskedFill(mask, 0)

   -- gradInput
   --self.gradInput:resizeAs(input):zero()
   --local ones = input.new():resize(target:size(1), 1):fill(1)
   --if input:type() == 'torch.CudaTensor' then
     --ones = ones:cuda()
   --end
   --self.gradInput:scatter(2, target:view(target:size(1), 1), -ones)
   --self.gradInput:maskedFill(mask:view(mask:size(1),1):expand(self.gradInput:size()), 0) -- zero out padding samples

   return self.gradInput
end

function ReinforceNLLCriterion:type(type)
   local module = self.module
   self.module = nil
   local ret = parent.type(self, type)
   self.module = module
   return ret
end
